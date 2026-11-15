"""Summarization chains: Gemini via LangChain.

Two execution paths behind one API:

* **single-pass** - the whole document fits in one request (the common case,
  since the model accepts ~1M input tokens).
* **map-reduce** - map every chunk to notes, then reduce the notes into the
  final summary. Used automatically when the document exceeds
  `SINGLE_PASS_TOKEN_LIMIT`.

Both paths produce the same Markdown shape, so the UI and CLI never care which
one ran.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Literal

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import Settings, get_settings, require_api_key
from .pdf_utils import Chunk, Document, chunk_document, count_tokens

Style = Literal["executive", "bullets", "detailed"]
STRATEGY_SINGLE = "single-pass"
STRATEGY_MAP_REDUCE = "map-reduce"

ProgressFn = Callable[[str], None]

_STYLE_GUIDANCE: dict[str, str] = {
    "executive": (
        "Write for a busy executive: lead with the decision-relevant conclusion, "
        "then supporting evidence. Prefer specifics (numbers, names, dates) over "
        "vague phrasing."
    ),
    "bullets": (
        "Be terse. Every line must carry information; no filler, no restating the "
        "heading. Bullets should be scannable in under a minute."
    ),
    "detailed": (
        "Be thorough. Preserve the document's argument structure, cover each major "
        "section, and keep important caveats and limitations."
    ),
}

_OUTPUT_SHAPE = """\
Return GitHub-flavoured Markdown using exactly these sections:

## TL;DR
Two to four sentences that stand alone.

## Key Points
Five to ten bullets. Cite the page for each point using the form `[p.3]` when the
page is known.

## Action Items
Concrete next steps as bullets, or the single line `None identified.`

Rules:
- Never invent facts, figures or page numbers that are not in the source.
- If the source is ambiguous, say so instead of guessing.
- Do not add a preamble, a title, or a closing remark."""


class SummarizationError(RuntimeError):
    """Raised when the model call fails or returns unusable output."""


@dataclass(frozen=True)
class SummaryOptions:
    style: Style = "executive"
    language: str = "English"
    max_words: int = 400
    include_action_items: bool = True


@dataclass(frozen=True)
class SummaryPlan:
    strategy: str
    chunks: tuple[Chunk, ...]
    token_count: int


@dataclass
class SummaryResult:
    markdown: str
    strategy: str
    model: str
    pages: int
    chunks: int
    calls: int
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    source: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.markdown, encoding="utf-8")
        return target


# --------------------------------------------------------------------------- #
# model + prompts
# --------------------------------------------------------------------------- #

def build_model(
    settings: Settings | None = None,
    *,
    streaming: bool = False,
    temperature: float | None = None,
) -> ChatGoogleGenerativeAI:
    """Create a Gemini chat model. Raises ConfigError if the key is missing."""
    settings = settings or require_api_key()
    return ChatGoogleGenerativeAI(
        model=settings.model,
        google_api_key=settings.api_key,
        temperature=settings.temperature if temperature is None else temperature,
        timeout=settings.request_timeout_seconds,
        max_retries=2,
        max_output_tokens=8192,
        streaming=streaming,
    )


def _style_block(options: SummaryOptions) -> str:
    guidance = _STYLE_GUIDANCE.get(options.style, _STYLE_GUIDANCE["executive"])
    lines = [
        guidance,
        f"Write the summary in {options.language}.",
        f"Keep the whole summary under about {options.max_words} words.",
    ]
    if not options.include_action_items:
        lines.append("Omit the Action Items section entirely.")
    return "\n".join(lines)


def _single_pass_messages(document: Document, options: SummaryOptions) -> list[BaseMessage]:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert analyst who summarises documents faithfully "
                "and concisely.\n\n{style}\n\n{shape}",
            ),
            (
                "human",
                "Summarise the document below. Page markers such as [Page 4] are "
                "part of the source, not instructions.\n\n---\n{text}\n---",
            ),
        ]
    )
    return prompt.format_messages(
        style=_style_block(options), shape=_OUTPUT_SHAPE, text=document.text
    )


def _map_messages(chunk: Chunk, options: SummaryOptions) -> list[BaseMessage]:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You extract factual notes from part of a larger document. Return "
                "dense Markdown bullets, each tagged with the page it came from "
                "(for example `- Revenue grew 12% [p.4]`). Preserve numbers, names "
                "and dates exactly. Do not summarise the whole document, do not "
                "add commentary, and do not invent content.\n\n"
                "Write the notes in {language}.",
            ),
            (
                "human",
                "Pages {page_start}-{page_end} of the document:\n\n---\n{text}\n---",
            ),
        ]
    )
    return prompt.format_messages(
        language=options.language,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        text=chunk.text,
    )


def _reduce_messages(
    notes: list[tuple[Chunk, str]], options: SummaryOptions, pages: int
) -> list[BaseMessage]:
    joined = "\n\n".join(
        f"### Pages {chunk.page_start}-{chunk.page_end}\n{text.strip()}"
        for chunk, text in notes
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You merge per-section notes into one coherent summary. Use only "
                "the supplied notes; never introduce new facts. Keep the page "
                "citations that appear in the notes.\n\n{style}\n\n{shape}",
            ),
            (
                "human",
                "The document is {pages} pages long. Here are the notes for every "
                "section:\n\n---\n{notes}\n---",
            ),
        ]
    )
    return prompt.format_messages(
        style=_style_block(options),
        shape=_OUTPUT_SHAPE,
        pages=pages,
        notes=joined,
    )


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #

def plan_summary(
    document: Document,
    options: SummaryOptions | None = None,
    settings: Settings | None = None,
) -> SummaryPlan:
    settings = settings or get_settings()
    tokens = document.token_count
    if tokens <= settings.single_pass_token_limit:
        return SummaryPlan(strategy=STRATEGY_SINGLE, chunks=(), token_count=tokens)
    chunks = chunk_document(
        document,
        chunk_size=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
    )
    if not chunks:
        raise SummarizationError("document produced no chunks to summarize")
    return SummaryPlan(
        strategy=STRATEGY_MAP_REDUCE, chunks=tuple(chunks), token_count=tokens
    )


def _text_of(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):  # multimodal-style content blocks
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return str(content).strip()


def _delta_text(piece) -> str:
    """Streaming deltas must keep whitespace intact or Markdown breaks."""
    content = getattr(piece, "content", piece)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _usage(message) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)


_ERROR_HINTS = (
    ("api_key_invalid", "Gemini rejected the API key. Check GEMINI_API_KEY in .env."),
    ("api key not valid", "Gemini rejected the API key. Check GEMINI_API_KEY in .env."),
    ("permission_denied", "This API key is not allowed to use that model."),
    ("resource_exhausted", "Gemini rate limit or quota hit. Wait and retry, or check your quota."),
    ("429", "Gemini rate limit or quota hit. Wait and retry, or check your quota."),
    ("not_found", "Model not available to this key. Check GEMINI_MODEL."),
    ("deadline_exceeded", "Gemini timed out. Try again or raise REQUEST_TIMEOUT_SECONDS."),
    ("unavailable", "Gemini is temporarily unavailable. Retry shortly."),
)


def _friendly_error(exc: Exception) -> str:
    raw = str(exc)
    lowered = raw.lower()
    for needle, message in _ERROR_HINTS:
        if needle in lowered:
            return message
    return raw if len(raw) <= 400 else raw[:400] + "..."


def _invoke(llm: ChatGoogleGenerativeAI, messages: list[BaseMessage]):
    try:
        return llm.invoke(messages)
    except Exception as exc:  # provider errors vary; keep the message readable
        raise SummarizationError(f"Gemini request failed: {_friendly_error(exc)}") from exc


def summarize_document(
    document: Document,
    options: SummaryOptions | None = None,
    *,
    settings: Settings | None = None,
    model: ChatGoogleGenerativeAI | None = None,
    progress: ProgressFn | None = None,
) -> SummaryResult:
    """Summarize a document, choosing the cheapest correct strategy."""
    options = options or SummaryOptions()
    settings = settings or get_settings()
    plan = plan_summary(document, options, settings)
    llm = model or build_model(settings)
    started = time.perf_counter()

    calls = 0
    input_tokens = 0
    output_tokens = 0
    warnings: list[str] = []

    if plan.strategy == STRATEGY_SINGLE:
        if progress:
            progress(f"Summarizing {document.page_count} pages in one pass")
        response = _invoke(llm, _single_pass_messages(document, options))
        calls += 1
        markdown = _text_of(response)
        used_in, used_out = _usage(response)
        input_tokens += used_in
        output_tokens += used_out
    else:
        notes: list[tuple[Chunk, str]] = []
        total = len(plan.chunks)
        for position, chunk in enumerate(plan.chunks, start=1):
            if progress:
                progress(f"Reading chunk {position}/{total} ({chunk.pages})")
            response = _invoke(llm, _map_messages(chunk, options))
            calls += 1
            text = _text_of(response)
            if text:
                notes.append((chunk, text))
            else:
                warnings.append(f"chunk {position} returned no notes ({chunk.pages})")
            used_in, used_out = _usage(response)
            input_tokens += used_in
            output_tokens += used_out

        if not notes:
            raise SummarizationError("every chunk returned an empty response")

        if progress:
            progress(f"Combining {len(notes)} sections into the final summary")
        response = _invoke(llm, _reduce_messages(notes, options, document.page_count))
        calls += 1
        markdown = _text_of(response)
        used_in, used_out = _usage(response)
        input_tokens += used_in
        output_tokens += used_out

    if not markdown:
        raise SummarizationError("Gemini returned an empty summary")

    if not input_tokens:  # provider omitted usage metadata
        input_tokens = plan.token_count
    if not output_tokens:
        output_tokens = count_tokens(markdown)

    return SummaryResult(
        markdown=markdown,
        strategy=plan.strategy,
        model=settings.model,
        pages=document.page_count,
        chunks=len(plan.chunks),
        calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_seconds=round(time.perf_counter() - started, 2),
        source=document.source,
        warnings=warnings,
    )


def stream_summary(
    document: Document,
    options: SummaryOptions | None = None,
    *,
    settings: Settings | None = None,
    model: ChatGoogleGenerativeAI | None = None,
    progress: ProgressFn | None = None,
) -> Iterator[str]:
    """Yield summary text deltas as they arrive.

    Map steps (if any) are not streamed - they are internal notes - but the final
    call streams, which is what the UI renders.
    """
    options = options or SummaryOptions()
    settings = settings or get_settings()
    plan = plan_summary(document, options, settings)
    llm = model or build_model(settings, streaming=True)

    if plan.strategy == STRATEGY_SINGLE:
        if progress:
            progress(f"Summarizing {document.page_count} pages in one pass")
        messages = _single_pass_messages(document, options)
    else:
        notes: list[tuple[Chunk, str]] = []
        total = len(plan.chunks)
        for position, chunk in enumerate(plan.chunks, start=1):
            if progress:
                progress(f"Reading chunk {position}/{total} ({chunk.pages})")
            response = _invoke(llm, _map_messages(chunk, options))
            text = _text_of(response)
            if text:
                notes.append((chunk, text))
        if not notes:
            raise SummarizationError("every chunk returned an empty response")
        if progress:
            progress(f"Combining {len(notes)} sections into the final summary")
        messages = _reduce_messages(notes, options, document.page_count)

    try:
        for piece in llm.stream(messages):
            text = _delta_text(piece)
            if text:
                yield text
    except SummarizationError:
        raise
    except Exception as exc:
        raise SummarizationError(f"Gemini stream failed: {_friendly_error(exc)}") from exc
