"""CLI tests. Nothing here touches the network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.config as config
from scripts.inspect_pdf import main as inspect_main
from scripts.summarize import main as summarize_main


def test_inspect_json_output(sample_pdf: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert inspect_main([str(sample_pdf), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pages"] == 3
    assert payload["strategy"] == "single-pass"
    assert payload["chunks"] == 3
    assert payload["estimated_llm_calls"] == 1


def test_inspect_missing_file_exits_nonzero(tmp_path: Path) -> None:
    assert inspect_main([str(tmp_path / "nope.pdf")]) == 1


def test_inspect_show_chunks(sample_pdf: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert inspect_main([str(sample_pdf), "--show-chunks"]) == 0
    out = capsys.readouterr().out
    assert "#" in out
    assert "p.1" in out


def test_summarize_rejects_nonsense_length(sample_pdf: Path) -> None:
    assert summarize_main([str(sample_pdf), "--max-words", "10"]) == 1


def test_summarize_without_api_key_exits_2(
    sample_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert summarize_main([str(sample_pdf)]) == 2
