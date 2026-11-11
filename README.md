# AI PDF Assistant

Ask questions about one or more PDFs and get answers with page citations.

It uses **hybrid search**: FAISS (meaning) and BM25 (keywords) are merged
with Reciprocal Rank Fusion, then `gpt-4.1-mini` writes the answer from
the retrieved passages only. Follow-up questions ("why is it useful?")
are rewritten into standalone search queries using the conversation.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create a `.env` file in the project folder:

```
OPENAI_API_KEY=sk-...
APP_PASSWORD=choose-a-long-password   # optional locally, required online
```

When `APP_PASSWORD` is set, the web app asks for it before showing
anything. After 5 wrong attempts the form is locked for a minute.
Without it the app is open to anyone who can reach it, which is fine on
your own computer but **never put the app online without a password**:
visitors would be using your OpenAI key. On Streamlit Community Cloud,
set `APP_PASSWORD` and `OPENAI_API_KEY` under *App settings → Secrets*.

## Run

**Web app**

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Pick one or more PDFs from the `pdf/` folder, or upload new ones, in the
sidebar. With several PDFs selected, they are searched together and
citations name the file, e.g. `[notes.pdf, page 60]`.
Turn on "Show retrieval details" to see how each passage was ranked and
what each API call cost.

Conversations are saved automatically in `conversations/` (one JSON file
each) and listed in the sidebar, so you can reopen and continue them after
a restart. Changing the selected PDFs starts a new conversation. Each
answer shows its token count and estimated cost; the sidebar shows the
total for the conversation and for all saved conversations. Prices are
set in `PRICES_PER_MILLION` in `pdf_assistant/config.py`.

**Terminal**

```powershell
.\.venv\Scripts\python.exe app.py                                   # default PDF
.\.venv\Scripts\python.exe app.py pdf\other.pdf                     # another PDF
.\.venv\Scripts\python.exe app.py pdf\ISLP_website.pdf pdf\other.pdf # several PDFs
.\.venv\Scripts\python.exe app.py --debug --rerank                  # details, reranking
```

Terminal conversations are saved too and appear in the web app. Each
answer prints its token count and estimated cost; `exit` prints the
session total.

Several PDFs share one combined FAISS index and one BM25 index, so scores
are comparable and the best passages win whichever file they come from.

The first run on a PDF creates embeddings (a few cents with
`text-embedding-3-small`); they are cached in `cache/` so later runs
start in about a second.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

The tests use small generated PDFs and a fake OpenAI client, so they run
in about a second and cost nothing. Any accidental call to the real API
fails the test.

## Evaluation

Measure how often the search finds the right page:

```powershell
# 1. Create test questions from random passages (once per PDF)
.\.venv\Scripts\python.exe -m pdf_assistant.evaluation generate --n 40

# 2. Score the search, with and without reranking
.\.venv\Scripts\python.exe -m pdf_assistant.evaluation run --compare
```

Results on `ISLP_website.pdf` (40 questions):

| Setting         | Page hit@5 | Page MRR | Avg search time |
|-----------------|-----------:|---------:|----------------:|
| Hybrid only     | 97.5%      | 0.804    | 1.2s            |
| Hybrid + rerank | 92.5%      | 0.863    | 2.5s            |

Reranking is therefore **off by default**. Turn it on with `--rerank`
in the terminal or the "Rerank results" toggle in the web app.
Re-run the evaluation after changing chunking, models or search settings.

To check that other PDFs don't crowd out the right answers, add them to
the search; a hit must then come from the right file *and* page:

```powershell
.\.venv\Scripts\python.exe -m pdf_assistant.evaluation run --with pdf\other.pdf
```

With a second, unrelated 333-page PDF added, page hit@5 stayed at 97.5% (MRR 0.825).

## Project layout

```
app.py                  terminal entry point
streamlit_app.py        web app
pdf_assistant/
  config.py             settings (models, chunk size, paths)
  ingest.py             PDF reading, chunking, embeddings, caching, indexes
  library.py            combine several PDFs into one searchable collection
  retrieval.py          FAISS + BM25 hybrid search with RRF
  rerank.py             optional model-based reranking
  llm.py                query rewriting and streamed answers
  usage.py              token counting and cost estimates
  conversations.py      saving and loading conversations
  cli.py                terminal chat loop
  evaluation.py         test-question generation and retrieval metrics
eval/                   generated test questions (commit these)
tests/                  pytest suite (no API calls)
```
