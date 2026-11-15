"""Summarize PDFs from the command line.

Usage:
    python scripts/summarize.py samples/sample.pdf
    python scripts/summarize.py report.pdf --style detailed --max-words 600
    python scripts/summarize.py a.pdf b.pdf --out output --no-stream --json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chains import (  # noqa: E402
    STRATEGY_MAP_REDUCE,
    SummarizationError,
    SummaryOptions,
    plan_summary,
    stream_summary,
    summarize_document,
)
from src.config import ConfigError, get_settings, require_api_key  # noqa: E402
from src.pdf_utils import PdfIngestError, extract_document  # noqa: E402


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]


def _resolve_out(out: Path | None, pdf: Path) -> Path | None:
    if out is None:
        return Path("output") / f"{pdf.stem}.md"
    if out.suffix.lower() == ".md":
        return out
    return out / f"{pdf.stem}.md"


def summarize_one(
    pdf: Path,
    options: SummaryOptions,
    *,
    stream: bool,
    out: Path | None,
    as_json: bool,
) -> int:
    settings = require_api_key()
    started = time.perf_counter()

    try:
        document = extract_document(pdf)
    except PdfIngestError as exc:
        print(f"{pdf.name}: {exc}", file=sys.stderr)
        return 1

    plan = plan_summary(document, options, settings)
    print(
        f"{pdf.name}: {document.page_count} pages, ~{document.token_count:,} tokens, "
        f"{plan.strategy}",
        file=sys.stderr,
    )
    if plan.strategy == STRATEGY_MAP_REDUCE:
        print(f"  map-reduce over {len(plan.chunks)} chunks", file=sys.stderr)

    def progress(message: str) -> None:
        print(f"  {message}", file=sys.stderr, flush=True)

    try:
        if stream:
            pieces: list[str] = []
            for delta in stream_summary(document, options, settings=settings, progress=progress):
                pieces.append(delta)
                if not as_json:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
            markdown = "".join(pieces).strip()
            if not as_json:
                sys.stdout.write("\n")
        else:
            result = summarize_document(document, options, settings=settings, progress=progress)
            markdown = result.markdown
            if not as_json:
                print(markdown)
    except SummarizationError as exc:
        print(f"{pdf.name}: {exc}", file=sys.stderr)
        return 1

    target = _resolve_out(out, pdf)
    saved = None
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown + "\n", encoding="utf-8")
        saved = str(target)

    elapsed = round(time.perf_counter() - started, 2)
    if as_json:
        print(
            json.dumps(
                {
                    "file": str(pdf),
                    "strategy": plan.strategy,
                    "pages": document.page_count,
                    "tokens": document.token_count,
                    "chunks": len(plan.chunks),
                    "elapsed_seconds": elapsed,
                    "saved_to": saved,
                    "markdown": markdown,
                },
                indent=2,
            )
        )
    else:
        print(
            f"  done in {elapsed}s" + (f" -> {saved}" if saved else ""),
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    # fail fast on malformed env config before doing any work
    get_settings()

    parser = argparse.ArgumentParser(description="Summarize PDFs with Gemini + LangChain.")
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument(
        "--style",
        choices=("executive", "bullets", "detailed"),
        default="executive",
    )
    parser.add_argument("--language", default="English")
    parser.add_argument("--max-words", type=int, default=400)
    parser.add_argument(
        "--no-action-items", action="store_true", help="omit the Action Items section"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output .md file or directory (default: output/<name>.md)",
    )
    parser.add_argument("--no-stream", action="store_true", help="wait for the full answer")
    parser.add_argument("--json", action="store_true", help="emit JSON metadata + markdown")
    args = parser.parse_args(argv)

    if args.max_words < 50:
        print("--max-words must be at least 50", file=sys.stderr)
        return 1

    options = SummaryOptions(
        style=args.style,
        language=args.language,
        max_words=args.max_words,
        include_action_items=not args.no_action_items,
    )

    failures = 0
    for pdf in args.pdfs:
        try:
            failures += summarize_one(
                pdf,
                options,
                stream=not args.no_stream,
                out=args.out,
                as_json=args.json,
            )
        except ConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
            return 130

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
