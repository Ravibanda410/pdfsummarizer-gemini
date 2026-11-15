"""Ingestion tests: extraction, token counting and page-bounded chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pdf_utils import (
    PdfIngestError,
    chunk_document,
    count_tokens,
    extract_document,
)


def test_extract_pages(sample_pdf: Path) -> None:
    document = extract_document(sample_pdf)
    assert document.page_count == 3
    assert document.empty_pages == ()
    assert document.char_count > 500
    assert document.token_count > 100
    assert "Page 1" in document.pages[0].text


def test_extract_from_bytes_uses_given_name(sample_bytes: bytes) -> None:
    document = extract_document(sample_bytes, name="upload.pdf")
    assert document.source == "upload.pdf"
    assert document.page_count == 3


def test_document_text_has_page_markers(sample_pdf: Path) -> None:
    text = extract_document(sample_pdf).text
    assert "[Page 1]" in text
    assert "[Page 3]" in text


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(PdfIngestError, match="not found"):
        extract_document(tmp_path / "nope.pdf")


def test_non_pdf_raises(tmp_path: Path) -> None:
    bad = tmp_path / "not-really.pdf"
    bad.write_text("this is plain text", encoding="utf-8")
    with pytest.raises(PdfIngestError):
        extract_document(bad)


def test_blank_pdf_raises_actionable_error(blank_pdf: Path) -> None:
    with pytest.raises(PdfIngestError, match="no extractable text"):
        extract_document(blank_pdf)


def test_chunks_never_span_pages(sample_pdf: Path) -> None:
    document = extract_document(sample_pdf)
    chunks = chunk_document(document, chunk_size=100, overlap=20)

    assert len(chunks) >= 3
    assert all(chunk.page_start == chunk.page_end for chunk in chunks)
    assert {chunk.page_start for chunk in chunks} == {1, 2, 3}
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.token_count > 0 for chunk in chunks)


def test_chunk_size_is_respected_within_overlap(sample_pdf: Path) -> None:
    document = extract_document(sample_pdf)
    chunks = chunk_document(document, chunk_size=100, overlap=20)
    # the last window of a page is merged into the previous chunk, so allow slack
    assert max(chunk.token_count for chunk in chunks) <= 100 + 60


def test_large_chunk_size_yields_one_chunk_per_page(sample_pdf: Path) -> None:
    document = extract_document(sample_pdf)
    chunks = chunk_document(document, chunk_size=6000, overlap=200)
    assert len(chunks) == 3


def test_chunks_preserve_page_headings(sample_pdf: Path) -> None:
    document = extract_document(sample_pdf)
    chunks = chunk_document(document, chunk_size=100, overlap=20)
    first_per_page = {}
    for chunk in chunks:
        first_per_page.setdefault(chunk.page_start, chunk.text)
    assert first_per_page[1].startswith("1. Introduction")
    assert first_per_page[2].startswith("2. Method")


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (100, 100), (100, 150)],
)
def test_invalid_chunk_parameters(sample_pdf: Path, chunk_size: int, overlap: int) -> None:
    document = extract_document(sample_pdf)
    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=chunk_size, overlap=overlap)


def test_count_tokens_edge_cases() -> None:
    assert count_tokens("") == 0
    assert count_tokens("hello world") > 0
    assert count_tokens("hello world" * 100) > count_tokens("hello world")
