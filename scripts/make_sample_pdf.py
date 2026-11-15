"""Generate a small text-based sample PDF for offline demos and smoke tests.

No third-party dependency: writes the PDF byte structure directly so the repo
can ship a deterministic fixture without a binary blob checked in by hand.

Usage:
    python scripts/make_sample_pdf.py                 # -> samples/sample.pdf
    python scripts/make_sample_pdf.py out/demo.pdf --pages 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PAGE_HEADINGS = [
    "1. Introduction and Scope",
    "2. Method and Data",
    "3. Results",
    "4. Discussion",
    "5. Conclusion",
]

BODY = (
    "This document exists so the summarizer can be exercised without needing a "
    "real report. It is generated locally, contains only plain ASCII text, and "
    "is therefore trivially safe to commit. Each page carries a different "
    "heading and a paragraph of prose so that page-aware chunking, token "
    "counting and citation rendering all have something meaningful to work "
    "with. A summarizer that cites the wrong page will be obvious immediately, "
    "because every heading names its own page number. "
)


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _wrap(text: str, width: int = 92) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _content_stream(heading: str, body: str, page_no: int) -> bytes:
    lines = [heading, ""] + _wrap(body) + ["", f"Page {page_no}"]
    parts = ["BT", "/F1 11 Tf", "50 760 Td", "15 TL"]
    for line in lines[:46]:
        parts.append(f"({_escape(line)}) Tj T*")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1")


def build_pdf(page_texts: list[str], path: Path) -> Path:
    objects: dict[int, bytes] = {}
    total_pages = len(page_texts)

    page_ids = [4 + 2 * i for i in range(total_pages)]
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {total_pages} >>".encode("latin-1")
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for index, text in enumerate(page_texts):
        page_id = page_ids[index]
        content_id = page_id + 1
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("latin-1")
        stream = _content_stream(PAGE_HEADINGS[index % len(PAGE_HEADINGS)], text, index + 1)
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = len(out)
        out += f"{obj_id} 0 obj\n".encode("latin-1") + objects[obj_id] + b"\nendobj\n"

    xref_pos = len(out)
    count = max(objects) + 1
    out += f"xref\n0 {count}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for obj_id in range(1, count):
        if obj_id in offsets:
            out += f"{offsets[obj_id]:010d} 00000 n \n".encode("latin-1")
        else:
            out += b"0000000000 65535 f \n"
    out += f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
        "latin-1"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a sample text PDF.")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "samples" / "sample.pdf",
    )
    parser.add_argument("--pages", type=int, default=3)
    args = parser.parse_args(argv)

    if args.pages < 1:
        print("--pages must be at least 1", file=sys.stderr)
        return 1

    texts = [BODY * 2 for _ in range(args.pages)]
    target = build_pdf(texts, args.output)
    print(f"wrote {target} ({args.pages} pages, {target.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
