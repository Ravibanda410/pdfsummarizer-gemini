# PDF Summarizer with Gemini + LangChain

Summarize PDF documents with Google Gemini, orchestrated through LangChain, in a Streamlit UI.

## Status

**Phase 1 complete** - PDF ingestion, token budgeting and page-aware chunking work
offline. Summarization chains land in Phase 2.

| Phase | Scope | State |
| --- | --- | --- |
| 0 | Toolchain, API key validation, repo scaffold, first push | done |
| 1 | PDF ingestion (pypdf extraction, token counting, chunking) | done |
| 2 | LangChain + Gemini chains (single-pass, map-reduce fallback) | next |
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

## Inspect a PDF before summarizing it

Ingestion is fully offline and makes no API calls, so you can check a document
for free:

```bash
python scripts/inspect_pdf.py samples/sample.pdf
```

```
sample.pdf
  pages              : 3 (3 with text)
  characters         : 2,942
  tokens (estimate)  : 612
  chunks             : 3 (size 6000, overlap 200)
  tokens per chunk   : min 197 / avg 198 / max 199
  strategy           : single-pass (single-pass limit 300,000 tokens)
  estimated API calls: 1
```

Useful flags:

| Flag | Effect |
| --- | --- |
| `--show-chunks` | list every chunk with its page range and token count |
| `--json` | machine-readable output for scripts |
| `--chunk-size` / `--overlap` | override the token windowing |
| `--no-generate` | (on `check_keys.py`) skip the billed call |

If a PDF has no selectable text - typically a scan - the inspector fails with a
clear message instead of sending an empty prompt to the model:

```
ERROR: scan.pdf contains no extractable text. It is probably a scanned image
PDF; OCR (e.g. ocrmypdf) is required first.
```

No sample file handy? Generate one:

```bash
python scripts/make_sample_pdf.py --pages 5
```

## Repository layout

```
app.py                     Streamlit entry point (Phase 3)
src/config.py              settings from env / .env
src/pdf_utils.py           extraction, token counting, page-aware chunking
scripts/check_keys.py      API key + model validation gate
scripts/inspect_pdf.py     offline PDF inspector (dry run)
scripts/make_sample_pdf.py generates a text PDF fixture
samples/sample.pdf         generated 3-page fixture
tests/                     unit tests (Phase 5)
requirements.txt           runtime dependencies
.env.example               environment template - copy to .env
```

## Notes

- `.env` is gitignored; only `.env.example` is committed.
- `gemini-3.1-flash-lite` accepts ~1,048,576 input tokens, so most documents fit in a
  single request. Documents above the configured threshold fall back to chunked
  map-reduce summarization.
- Optional tuning (environment variables or `.env`): `CHUNK_SIZE_TOKENS` (6000),
  `CHUNK_OVERLAP_TOKENS` (200), `SINGLE_PASS_TOKEN_LIMIT` (300000),
  `GEMINI_TEMPERATURE` (0.2), `REQUEST_TIMEOUT_SECONDS` (180).
- Token counts use tiktoken's `cl100k_base` as a budget estimate; Gemini's own
  tokenizer differs slightly. If tiktoken cannot download its encoding, counting
  falls back to a ~4-chars-per-token heuristic instead of failing.

## License

MIT - see [LICENSE](LICENSE).
