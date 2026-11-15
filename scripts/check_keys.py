"""Phase 0 gate: prove the Gemini API key and model actually work.

Runs three checks against the live API and exits non-zero if any fail:
  1. the key authenticates (models.list)
  2. the configured model exists and supports generateContent
  3. a tiny generateContent call returns the expected text

Stdlib only, so it works before `pip install -r requirements.txt`.

Usage:
    python scripts/check_keys.py
    python scripts/check_keys.py --no-generate   # skip the billed call
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
TIMEOUT = 60
EXPECTED_REPLY = "PHASE0_OK"


def load_dotenv(path: Path) -> None:
    """Minimal .env loader: existing environment variables always win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def mask(secret: str) -> str:
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-2:]} (len {len(secret)})"


def request(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method="POST" if data else "GET",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"network error: {exc.reason}") from exc


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("GEMINI_MODEL", "").strip()

    failures: list[str] = []
    print(f"repo root : {root}")
    print(f"api key   : {mask(key) if key else '<missing>'}")
    print(f"model     : {model or '<missing>'}")
    print()

    if not key:
        print("FAIL  GEMINI_API_KEY is not set (copy .env.example to .env)")
        return 1
    if not model:
        print("FAIL  GEMINI_MODEL is not set (copy .env.example to .env)")
        return 1

    # 1. key authenticates
    try:
        listing = request(f"{API_ROOT}/models?key={key}")
        models = listing.get("models", [])
        print(f"PASS  key authenticates - {len(models)} models visible")
    except RuntimeError as exc:
        print(f"FAIL  key rejected - {exc}")
        return 1

    # 2. configured model is usable for generation
    target = next((m for m in models if m.get("name") == f"models/{model}"), None)
    if target is None:
        failures.append(f"model '{model}' not available to this key")
        print(f"FAIL  model '{model}' not in the visible model list")
        flash = [m["name"].removeprefix("models/") for m in models if "flash" in m.get("name", "")]
        if flash:
            print("      available flash models: " + ", ".join(sorted(flash)[:12]))
    elif "generateContent" not in target.get("supportedGenerationMethods", []):
        failures.append(f"model '{model}' does not support generateContent")
        print(f"FAIL  model '{model}' does not support generateContent")
    else:
        print(f"PASS  model '{model}' supports generateContent")

    # 3. real generation
    if "--no-generate" in sys.argv:
        print("SKIP  generation check (--no-generate)")
    elif target is not None:
        payload = {"contents": [{"parts": [{"text": f"Reply with exactly: {EXPECTED_REPLY}"}]}]}
        try:
            result = request(f"{API_ROOT}/models/{model}:generateContent?key={key}", payload)
            candidate = (result.get("candidates") or [{}])[0]
            text = "".join(
                part.get("text", "") for part in candidate.get("content", {}).get("parts", [])
            ).strip()
            usage = result.get("usageMetadata", {})
            if EXPECTED_REPLY in text:
                print(
                    "PASS  generation works - reply "
                    f"'{text[:40]}' ({usage.get('totalTokenCount', '?')} tokens)"
                )
            else:
                failures.append(f"unexpected generation reply: {text[:80]!r}")
                print(f"FAIL  unexpected reply: {text[:80]!r}")
        except RuntimeError as exc:
            failures.append(f"generateContent failed - {exc}")
            print(f"FAIL  generateContent failed - {exc}")

    print()
    if failures:
        print(f"RESULT: FAILED ({len(failures)} problem(s))")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("RESULT: OK - API key, model and generation all verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
