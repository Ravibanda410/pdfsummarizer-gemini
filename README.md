# PDF Summarizer with Gemini + LangChain

Summarize PDF documents with Google Gemini, orchestrated through LangChain, in a Streamlit UI.

## Status

**Phase 0 complete** - environment, API key and GitHub delivery pipeline are verified.
The application itself is built in the phases below.

| Phase | Scope | State |
| --- | --- | --- |
| 0 | Toolchain, API key validation, repo scaffold, first push | done |
| 1 | PDF ingestion (pypdf extraction, token counting, chunking) | next |
| 2 | LangChain + Gemini chains (single-pass, map-reduce fallback) | planned |
| 3 | Streamlit UI (upload, options, streaming, download) | planned |
| 4 | Chat with PDF, multi-document comparison | planned |
| 5 | Tests, lint, CI, deployment | planned |

## Requirements

- Python 3.10+
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
- Git and GitHub CLI (only needed to publish the repo)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

Copy the environment template and add your key:

```bash
copy .env.example .env         # Windows
# cp .env.example .env         # macOS / Linux
```

```ini
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-3.1-flash-lite
```

## Verify your key before running anything

```bash
python scripts/check_keys.py
```

The script checks that the key authenticates, that the configured model exists and
supports `generateContent`, and that a real generation succeeds. It exits non-zero
on any failure, so it can gate CI.

```
PASS  key authenticates - 50 models visible
PASS  model 'gemini-3.1-flash-lite' supports generateContent
PASS  generation works - reply 'PHASE0_OK' (13 tokens)
RESULT: OK - API key, model and generation all verified
```

Use `--no-generate` to skip the billed call.

## Repository layout

```
app.py                  Streamlit entry point (Phase 3)
src/                    application package (Phase 1+)
scripts/check_keys.py   API key + model validation gate
tests/                  unit tests (Phase 5)
requirements.txt        runtime dependencies
.env.example            environment template - copy to .env
```

## Notes

- `.env` is gitignored; only `.env.example` is committed.
- `gemini-3.1-flash-lite` accepts ~1,048,576 input tokens, so most documents fit in a
  single request. Documents above the configured threshold fall back to chunked
  map-reduce summarization.

## License

MIT - see [LICENSE](LICENSE).
