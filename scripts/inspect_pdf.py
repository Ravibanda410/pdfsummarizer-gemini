"""Dry-run PDF inspector: extraction stats, token budget and chunk plan.

Makes no API calls, so it is safe to run against any document before spending
tokens. Use it to confirm a PDF actually has selectable text and to see whether
the summarizer will take the single-pass or map-reduce route.

Usage:
    python scripts/inspect_pdf.py samples/sample.pdf
    python scripts/inspect_pdf.py report.pdf --chunk-size 8000 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings  # noqa: E402
from src.pdf_utils import (  # noqa: E402
    PdfIngestError,
    chunk_document,
    extract_document,
)


def inspect(path: Path, chunk_size: int, overlap: int, limit: int) -> dict:
    document = extract_document(path)
    chunks = chunk_document(document, chunk_size=chunk_size, overlap=overlap)
    tokens = document.token_count
    chunk_tokens = [c.token_count for c in chunks]

    return {
        "file": path.name,
        "pages": document.page_count,
        "pages_with_text": document.page_count - len(document.empty_pages),
        "empty_pages": list(document.empty_pages),
        "characters": document.char_count,
        "tokens": tokens,
        "chunks": len(chunks),
        "chunk_tokens_min": min(chunk_tokens) if chunk_tokens else 0,
        "chunk_tokens_max": max(chunk_tokens) if chunk_tokens else 0,
        "chunk_tokens_avg": round(sum(chunk_tokens) / len(chunk_tokens)) if chunk_tokens else 0,
        "strategy": "single-pass" if tokens <= limit else "map-reduce",
        "estimated_llm_calls": 1 if tokens <= limit else len(chunks) + 1,
        "chunk_plan": [
            {"chunk": c.index + 1, "pages": c.pages, "tokens": c.token_count}
            for c in chunks
        ],
    }


def print_report(report: dict, chunk_size: int, overlap: int, limit: int) -> None:
    print(f"{report['file']}")
    print(f"  pages              : {report['pages']} ({report['pages_with_text']} with text)")
    if report["empty_pages"]:
        shown = ", ".join(str(p) for p in report["empty_pages"][:10])
        extra = " ..." if len(report["empty_pages"]) > 10 else ""
        print(f"  empty pages        : {shown}{extra}")
    print(f"  characters         : {report['characters']:,}")
    print(f"  tokens (estimate)  : {report['tokens']:,}")
    print(
        f"  chunks             : {report['chunks']} "
        f"(size {chunk_size}, overlap {overlap})"
    )
    if report["chunks"]:
        print(
            f"  tokens per chunk   : min {report['chunk_tokens_min']} / "
            f"avg {report['chunk_tokens_avg']} / max {report['chunk_tokens_max']}"
        )
    print(
        f"  strategy           : {report['strategy']} "
        f"(single-pass limit {limit:,} tokens)"
    )
    print(f"  estimated API calls: {report['estimated_llm_calls']}")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Inspect PDFs offline: page text, token budget, chunk plan."
    )
    parser.add_argument("pdfs", nargs="+", type=Path, help="one or more PDF files")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size_tokens,
        help=f"chunk size in tokens (default {settings.chunk_size_tokens})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=settings.chunk_overlap_tokens,
        help=f"chunk overlap in tokens (default {settings.chunk_overlap_tokens})",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--show-chunks", action="store_true", help="list every chunk and its pages"
    )
    args = parser.parse_args(argv)

    reports: list[dict] = []
    failures = 0
    for path in args.pdfs:
        try:
            report = inspect(
                path,
                args.chunk_size,
                args.overlap,
                settings.single_pass_token_limit,
            )
        except PdfIngestError as exc:
            failures += 1
            if args.json:
                reports.append({"file": path.name, "error": str(exc)})
            else:
                print(f"{path.name}\n  ERROR: {exc}", file=sys.stderr)
            continue
        reports.append(report)
        if not args.json:
            print_report(report, args.chunk_size, args.overlap, settings.single_pass_token_limit)
            if args.show_chunks:
                for chunk in report["chunk_plan"]:
                    print(f"    #{chunk['chunk']:>3}  {chunk['pages']:>10}  {chunk['tokens']:>6} tokens")
            print()

    if args.json:
        print(json.dumps(reports if len(reports) > 1 else reports[0], indent=2))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
