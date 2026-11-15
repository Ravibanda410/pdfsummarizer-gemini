"""PDF text extraction, token counting and chunking.

Everything here is offline: no API calls, no API key required. That keeps
ingestion testable on its own and lets the UI show cost estimates before the
user spends tokens.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

# Chunks smaller than this (in tokens) get merged into the previous chunk so we
# never pay for a request that only summarises a heading or a page number.
MIN_CHUNK_TOKENS = 40

_WHITESPACE_RE = re.compile(r"[ \t\u00a0]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


class PdfIngestError(RuntimeError):
    """Raised when a PDF cannot be read or contains no extractable text."""


@dataclass(frozen=True)
class Page:
    number: int
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True)
class Document:
    source: str
    pages: tuple[Page, ...] = field(default_factory=tuple)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text(self) -> str:
        """Full text with `[Page N]` markers, matching what the LLM will see."""
        return "\n\n".join(f"[Page {p.number}]\n{p.text}" for p in self.pages)

    @property
    def char_count(self) -> int:
        return sum(p.char_count for p in self.pages)

    @property
    def empty_pages(self) -> tuple[int, ...]:
        return tuple(p.number for p in self.pages if p.is_empty)

    @property
    def token_count(self) -> int:
        return count_tokens(self.text)


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    token_count: int
    page_start: int
    page_end: int

    @property
    def pages(self) -> str:
        if self.page_start == self.page_end:
            return f"p.{self.page_start}"
        return f"p.{self.page_start}-{self.page_end}"


# --------------------------------------------------------------------------- #
# tokenisation
# --------------------------------------------------------------------------- #

_ENCODING = None
_ENCODING_FAILED = False


def _encoding():
    """tiktoken cl100k_base, or None if it cannot be loaded (offline)."""
    global _ENCODING, _ENCODING_FAILED
    if _ENCODING is not None or _ENCODING_FAILED:
        return _ENCODING
    try:
        import tiktoken

        _ENCODING = tiktoken.get_encoding("cl100k_base")
    except Exception:  # network/model download problems - degrade gracefully
        _ENCODING_FAILED = True
        _ENCODING = None
    return _ENCODING


def count_tokens(text: str) -> int:
    """Token count using tiktoken; falls back to a ~4-chars-per-token estimate.

    Gemini's own tokenizer differs slightly, so treat this as a budgeting
    estimate rather than an exact bill.
    """
    if not text:
        return 0
    enc = _encoding()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return math.ceil(len(text) / 4)


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

def _normalise(raw: str) -> str:
    text = raw.replace("\x00", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    lines = [line.strip() for line in text.splitlines()]
    return _BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()


def _open_reader(source: str | Path | bytes | BinaryIO, password: str | None) -> PdfReader:
    try:
        if isinstance(source, bytes):
            reader = PdfReader(io.BytesIO(source), strict=False)
        else:
            reader = PdfReader(source, strict=False)
    except FileNotFoundError as exc:
        raise PdfIngestError(f"file not found: {source}") from exc
    except (PdfReadError, ValueError, OSError) as exc:
        raise PdfIngestError(f"could not read PDF: {exc}") from exc

    if reader.is_encrypted:
        # Many PDFs are "encrypted" with an empty user password (owner-locked only).
        try:
            unlocked = reader.decrypt(password or "")
        except Exception as exc:  # pragma: no cover - pypdf internals
            raise PdfIngestError(f"encrypted PDF could not be opened: {exc}") from exc
        if not unlocked:
            raise PdfIngestError(
                "PDF is password protected; provide the password to read it"
            )
    return reader


def extract_document(
    source: str | Path | bytes | BinaryIO,
    *,
    password: str | None = None,
    name: str | None = None,
) -> Document:
    """Extract page-level text from a PDF.

    Raises PdfIngestError when the file is unreadable, encrypted without a
    usable password, or yields no text at all (typically a scanned document).
    """
    reader = _open_reader(source, password)

    label = name or (str(source) if isinstance(source, (str, Path)) else "<uploaded>")
    pages: list[Page] = []
    for number, raw_page in enumerate(reader.pages, start=1):
        try:
            raw_text = raw_page.extract_text() or ""
        except Exception:  # one bad page should not kill the whole document
            raw_text = ""
        pages.append(Page(number=number, text=_normalise(raw_text)))

    document = Document(source=label, pages=tuple(pages))
    if document.char_count == 0:
        raise PdfIngestError(
            f"{Path(label).name} contains no extractable text. It is probably a "
            "scanned image PDF; OCR (e.g. ocrmypdf) is required first."
        )
    return document


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #

def _word_windows(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int]]:
    """Fallback chunker used when tiktoken is unavailable."""
    words = text.split()
    if not words:
        return []
    per_chunk = max(1, int(chunk_size * 0.75))
    step = max(1, per_chunk - int(overlap * 0.75))
    windows: list[tuple[str, int]] = []
    for start in range(0, len(words), step):
        piece = words[start : start + per_chunk]
        if not piece:
            break
        chunk_text = " ".join(piece)
        windows.append((chunk_text, count_tokens(chunk_text)))
        if start + per_chunk >= len(words):
            break
    return windows


def _page_windows(
    text: str, chunk_size: int, overlap: int
) -> list[tuple[str, int]]:
    enc = _encoding()
    if enc is None:
        return _word_windows(text, chunk_size, overlap)

    tokens = enc.encode(text, disallowed_special=())
    if len(tokens) <= chunk_size:
        return [(text, len(tokens))]

    step = max(1, chunk_size - overlap)
    windows: list[tuple[str, int]] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        windows.append((enc.decode(window).strip(), len(window)))
        if start + chunk_size >= len(tokens):
            break
    return windows


def chunk_document(
    document: Document,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split a document into token-bounded chunks that never span pages.

    Keeping chunks inside a single page means every summary can cite a page
    number without guesswork.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[Chunk] = []
    for page in document.pages:
        if page.is_empty:
            continue
        for text, tokens in _page_windows(page.text, chunk_size, overlap):
            if not text:
                continue
            if (
                chunks
                and tokens < MIN_CHUNK_TOKENS
                and chunks[-1].page_end == page.number
            ):
                previous = chunks[-1]
                merged_text = f"{previous.text}\n{text}"
                chunks[-1] = Chunk(
                    index=previous.index,
                    text=merged_text,
                    token_count=count_tokens(merged_text),
                    page_start=previous.page_start,
                    page_end=page.number,
                )
                continue
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=text,
                    token_count=tokens,
                    page_start=page.number,
                    page_end=page.number,
                )
            )
    return chunks


def iter_chunk_texts(chunks: Iterable[Chunk]) -> Iterator[str]:
    for chunk in chunks:
        yield chunk.text
