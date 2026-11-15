"""Chain tests. Gemini is replaced by tests.fakes.FakeLLM - no network, no cost."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.chains import (
    STRATEGY_MAP_REDUCE,
    STRATEGY_SINGLE,
    SummarizationError,
    build_model,
    plan_summary,
    stream_summary,
    summarize_document,
)
from src.pdf_utils import extract_document
from tests.fakes import FakeLLM, make_settings

BIG_LIMIT = 10**9
TINY_LIMIT = 10


@pytest.fixture
def document(sample_pdf: Path):
    return extract_document(sample_pdf)


def test_plan_single_pass(document) -> None:
    plan = plan_summary(document, settings=make_settings(single_pass_token_limit=BIG_LIMIT))
    assert plan.strategy == STRATEGY_SINGLE
    assert plan.chunks == ()
    assert plan.token_count == document.token_count


def test_plan_map_reduce_when_over_limit(document) -> None:
    plan = plan_summary(document, settings=make_settings(single_pass_token_limit=TINY_LIMIT))
    assert plan.strategy == STRATEGY_MAP_REDUCE
    assert len(plan.chunks) >= 3
    assert all(chunk.page_start == chunk.page_end for chunk in plan.chunks)


def test_summarize_single_pass(document) -> None:
    fake = FakeLLM(usage=(500, 60))
    result = summarize_document(
        document, settings=make_settings(single_pass_token_limit=BIG_LIMIT), model=fake
    )

    assert result.strategy == STRATEGY_SINGLE
    assert result.markdown.startswith("## TL;DR")
    assert result.calls == 1
    assert len(fake.calls) == 1
    assert (result.input_tokens, result.output_tokens) == (500, 60)
    assert result.pages == 3
    assert result.chunks == 0
    assert result.elapsed_seconds >= 0


def test_summarize_map_reduce(document) -> None:
    fake = FakeLLM()
    settings = make_settings(single_pass_token_limit=TINY_LIMIT)
    stages: list[str] = []

    result = summarize_document(document, settings=settings, model=fake, progress=stages.append)

    expected_chunks = len(plan_summary(document, settings=settings).chunks)
    assert result.strategy == STRATEGY_MAP_REDUCE
    assert result.chunks == expected_chunks
    assert result.calls == expected_chunks + 1
    assert len(fake.calls) == result.calls
    assert any("Reading chunk" in stage for stage in stages)
    assert any("Combining" in stage for stage in stages)


def test_usage_accumulates_across_calls(document) -> None:
    fake = FakeLLM(usage=(100, 10))
    result = summarize_document(
        document, settings=make_settings(single_pass_token_limit=TINY_LIMIT), model=fake
    )
    assert result.input_tokens == 100 * result.calls
    assert result.output_tokens == 10 * result.calls


def test_empty_single_pass_reply_raises(document) -> None:
    fake = FakeLLM(reply="")
    with pytest.raises(SummarizationError, match="empty summary"):
        summarize_document(
            document, settings=make_settings(single_pass_token_limit=BIG_LIMIT), model=fake
        )


def test_map_reduce_without_any_notes_raises(document) -> None:
    fake = FakeLLM(reply="")
    with pytest.raises(SummarizationError, match="every chunk returned an empty response"):
        summarize_document(
            document, settings=make_settings(single_pass_token_limit=TINY_LIMIT), model=fake
        )


def test_bad_key_error_is_made_human_readable(document) -> None:
    fake = FakeLLM(fail_with=Exception("400 INVALID_ARGUMENT: API key not valid"))
    with pytest.raises(SummarizationError, match="rejected the API key"):
        summarize_document(
            document, settings=make_settings(single_pass_token_limit=BIG_LIMIT), model=fake
        )


def test_rate_limit_error_is_made_human_readable(document) -> None:
    fake = FakeLLM(fail_with=Exception("429 RESOURCE_EXHAUSTED: quota"))
    with pytest.raises(SummarizationError, match="rate limit"):
        summarize_document(
            document, settings=make_settings(single_pass_token_limit=BIG_LIMIT), model=fake
        )


def test_stream_summary_yields_full_text(document) -> None:
    fake = FakeLLM(reply="## TL;DR\nfirst line\nsecond line\n")
    pieces = list(
        stream_summary(
            document, settings=make_settings(single_pass_token_limit=BIG_LIMIT), model=fake
        )
    )
    assert "".join(pieces).strip().endswith("second line")


def test_stream_summary_wraps_provider_errors(document) -> None:
    fake = FakeLLM(fail_with=Exception("429 RESOURCE_EXHAUSTED"))
    with pytest.raises(SummarizationError, match="rate limit"):
        list(
            stream_summary(
                document, settings=make_settings(single_pass_token_limit=BIG_LIMIT), model=fake
            )
        )


def test_build_model_uses_settings() -> None:
    model = build_model(make_settings(model="gemini-something"))
    assert model.model == "gemini-something"
