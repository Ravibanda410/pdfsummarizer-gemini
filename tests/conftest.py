"""Shared fixtures. All tests are offline: no API key, no network."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.make_sample_pdf import build_pdf  # noqa: E402
from src.config import get_settings  # noqa: E402

BODY = (
    "This generated paragraph exists so that page-aware chunking, token counting "
    "and citation rendering all have meaningful text to work with. Each page "
    "carries a distinct heading so a wrong citation is obvious immediately. "
)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """get_settings() is lru_cached; keep tests independent of each other."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("fixtures") / "sample.pdf"
    return build_pdf([BODY * 2] * 3, target)


@pytest.fixture(scope="session")
def sample_bytes(sample_pdf: Path) -> bytes:
    return sample_pdf.read_bytes()


@pytest.fixture
def blank_pdf(tmp_path: Path) -> Path:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(2):
        writer.add_blank_page(width=612, height=792)
    target = tmp_path / "blank.pdf"
    with target.open("wb") as handle:
        writer.write(handle)
    return target
