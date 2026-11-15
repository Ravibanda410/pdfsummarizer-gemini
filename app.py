"""Streamlit UI for the PDF summarizer.

Run with:
    streamlit run app.py

Design notes:
* Extraction stats are computed offline, so you see pages/tokens/cost before
  spending anything.
* Results are cached in session state keyed by file hash + options, so tweaking
  a slider and switching back does not re-bill the API.
* Streaming writes into a placeholder so map-reduce progress and text deltas can
  share one screen.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import streamlit as st

from src.chains import (
    STRATEGY_MAP_REDUCE,
    SummaryOptions,
    SummarizationError,
    plan_summary,
    stream_summary,
    summarize_document,
)
from src.config import PROJECT_ROOT, Settings, get_settings
from src.pdf_utils import PdfIngestError, count_tokens, extract_document

SAMPLE_PDF = PROJECT_ROOT / "samples" / "sample.pdf"
STYLES = {
    "executive": "Executive — conclusion first",
    "bullets": "Bullets — terse and scannable",
    "detailed": "Detailed — section by section",
}


def _load_secrets() -> None:
    """Let Streamlit Cloud secrets act as environment variables."""
    for key in ("GEMINI_API_KEY", "GEMINI_MODEL"):
        if not os.environ.get(key):
            try:
                value = st.secrets.get(key)
            except Exception:
                value = None
            if value:
                os.environ[key] = str(value)


@st.cache_data(show_spinner=False, max_entries=16)
def _extract(data: bytes, name: str):
    return extract_document(data, name=name)


@st.cache_data(show_spinner=False, max_entries=16)
def _stats(data: bytes, name: str):
    document = _extract(data, name)
    plan = plan_summary(document)
    return {
        "pages": document.page_count,
        "empty_pages": len(document.empty_pages),
        "tokens": document.token_count,
        "chunks": len(plan.chunks),
        "strategy": plan.strategy,
        "calls": 1 if plan.strategy != STRATEGY_MAP_REDUCE else len(plan.chunks) + 1,
    }


def _result_key(file_hash: str, options: SummaryOptions) -> str:
    raw = (
        f"{file_hash}|{options.style}|{options.language}|{options.max_words}|"
        f"{options.include_action_items}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sidebar(settings: Settings) -> tuple[SummaryOptions, bool]:
    with st.sidebar:
        st.header("Options")
        style = st.selectbox(
            "Style",
            options=list(STYLES),
            format_func=lambda key: STYLES[key],
            index=0,
        )
        language = st.text_input("Output language", value="English")
        max_words = st.slider("Target length (words)", 100, 1200, 400, step=50)
        include_actions = st.checkbox("Include action items", value=True)
        stream = st.toggle("Stream output", value=True)

        st.divider()
        st.caption("Model")
        st.code(settings.model, language=None)
        st.caption(
            f"Single-pass under {settings.single_pass_token_limit:,} tokens · "
            f"chunks of {settings.chunk_size_tokens:,}"
        )

        if st.button("Clear cached results", use_container_width=True):
            st.session_state.pop("results", None)
            st.toast("Cache cleared")

        st.divider()
        st.caption(
            "Extraction runs locally. Nothing is sent to Gemini until you press "
            "**Summarize**."
        )

    return (
        SummaryOptions(
            style=style,
            language=language.strip() or "English",
            max_words=max_words,
            include_action_items=include_actions,
        ),
        stream,
    )


def _run_summary(
    data: bytes,
    name: str,
    options: SummaryOptions,
    settings: Settings,
    stream: bool,
) -> dict:
    document = _extract(data, name)
    plan = plan_summary(document, options, settings)

    started = time.perf_counter()
    status = st.status("Working…", expanded=True)
    progress_line = status.empty()
    output = st.empty()

    def progress(message: str) -> None:
        progress_line.info(message)

    try:
        if stream:
            pieces: list[str] = []
            for delta in stream_summary(
                document, options, settings=settings, progress=progress
            ):
                pieces.append(delta)
                output.markdown("".join(pieces))
            markdown = "".join(pieces).strip()
            result = {
                "markdown": markdown,
                "strategy": plan.strategy,
                "model": settings.model,
                "pages": document.page_count,
                "chunks": len(plan.chunks),
                "calls": 1 if plan.strategy != STRATEGY_MAP_REDUCE else len(plan.chunks) + 1,
                "input_tokens": plan.token_count,
                "output_tokens": count_tokens(markdown),
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "estimated": True,
            }
        else:
            summary = summarize_document(
                document, options, settings=settings, progress=progress
            )
            output.markdown(summary.markdown)
            result = {
                "markdown": summary.markdown,
                "strategy": summary.strategy,
                "model": summary.model,
                "pages": summary.pages,
                "chunks": summary.chunks,
                "calls": summary.calls,
                "input_tokens": summary.input_tokens,
                "output_tokens": summary.output_tokens,
                "elapsed_seconds": summary.elapsed_seconds,
                "estimated": False,
            }
    except SummarizationError as exc:
        status.update(label="Failed", state="error")
        st.error(str(exc))
        return {}
    finally:
        progress_line.empty()

    status.update(label="Summary ready", state="complete", expanded=False)
    return result


def _render_result(result: dict, name: str) -> None:
    markdown = result["markdown"]
    st.markdown(markdown)

    meta = [
        f"**{result['strategy']}**",
        f"{result['calls']} call(s)",
        f"{result['pages']} pages",
        f"≈{result['input_tokens']:,} in / {result['output_tokens']:,} out tokens",
    ]
    if result.get("elapsed_seconds"):
        meta.append(f"{result['elapsed_seconds']}s")
    st.caption(" · ".join(meta))

    safe_name = Path(name).stem.replace(" ", "_")
    digest = hashlib.sha1(markdown.encode("utf-8")).hexdigest()[:8]
    st.download_button(
        "Download Markdown",
        data=markdown + "\n",
        file_name=f"{safe_name}_summary.md",
        mime="text/markdown",
        key=f"dl-{safe_name}-{digest}",
    )


def _render_card(name: str, data: bytes, options: SummaryOptions, settings: Settings, stream: bool) -> None:
    with st.container(border=True):
        st.subheader(name)

        try:
            stats = _stats(data, name)
        except PdfIngestError as exc:
            st.error(str(exc))
            return

        cols = st.columns(4)
        cols[0].metric("Pages", stats["pages"])
        cols[1].metric("Tokens", f"{stats['tokens']:,}")
        cols[2].metric("Strategy", stats["strategy"])
        cols[3].metric("API calls", stats["calls"])
        if stats["empty_pages"]:
            st.caption(
                f"{stats['empty_pages']} page(s) had no extractable text "
                "(likely images) and are skipped."
            )

        file_hash = hashlib.sha256(data).hexdigest()
        key = _result_key(file_hash, options)
        results = st.session_state.setdefault("results", {})
        cached = results.get(key)

        label = "Summarize" if not cached else "Regenerate"
        disabled = not settings.has_api_key
        if st.button(label, key=f"go-{file_hash[:12]}-{key[:8]}", disabled=disabled):
            if cached:
                results.pop(key, None)
            with st.spinner("Contacting Gemini…"):
                result = _run_summary(data, name, options, settings, stream)
            if result:
                results[key] = result
            st.rerun()

        if cached:
            st.divider()
            _render_result(cached, name)


def main() -> None:
    st.set_page_config(page_title="PDF Summarizer · Gemini", page_icon="📄", layout="wide")
    _load_secrets()
    settings = get_settings()

    st.title("📄 PDF Summarizer")
    st.caption(
        "LangChain + Gemini · extraction and token budgeting run locally, "
        "summaries cite page numbers."
    )

    if not settings.has_api_key:
        st.error(
            "GEMINI_API_KEY is not set. Copy `.env.example` to `.env` and add your "
            "key (or set it in Streamlit secrets), then reload."
        )

    options, stream = _sidebar(settings)

    uploaded = st.file_uploader(
        "Drop one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    documents: list[tuple[str, bytes]] = []
    if uploaded:
        for item in uploaded:
            documents.append((item.name, item.getvalue()))
    elif SAMPLE_PDF.is_file() and st.checkbox(
        "Use the bundled sample PDF", value=not uploaded
    ):
        documents.append((SAMPLE_PDF.name, SAMPLE_PDF.read_bytes()))

    if not documents:
        st.info("Upload a PDF to see page count, token estimate and summarization cost.")
        return

    for name, data in documents:
        _render_card(name, data, options, settings, stream)


if __name__ == "__main__":
    main()
