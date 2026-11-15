"""Guards against committing real credentials.

`.env` is gitignored, but `.env.example` is tracked and public - which is exactly
how a live key ended up on GitHub once already. These tests fail the build instead
of waiting for someone to notice.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / ".env.example"

# Real Google AI Studio keys are ~39-53 chars and start with "AIza" or "AQ.".
SUSPICIOUS_PREFIXES = ("AIza", "AQ.")


def _values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def test_env_example_ships_a_placeholder() -> None:
    assert EXAMPLE.is_file()
    key = _values(EXAMPLE).get("GEMINI_API_KEY", "")
    assert key, ".env.example should show the variable name"
    assert not key.startswith(SUSPICIOUS_PREFIXES), "looks like a real API key"
    assert len(key) < 30, "looks like a real API key, not a placeholder"


def test_real_env_file_is_never_tracked() -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")

    git = shutil.which("git")
    if git is None:
        # Still assert the rule that keeps it out of git.
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
        assert any(line.strip("/") == ".env" for line in ignored), ".gitignore must list .env"
        return

    tracked = subprocess.run(
        [git, "ls-files", "--error-unmatch", ".env"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert tracked.returncode != 0, ".env must never be committed"
