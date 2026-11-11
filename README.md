# AI PDF Assistant

Ask questions about a PDF and get answers with page citations.

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
```

## Run

**Web app**

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Pick a PDF from the `pdf/` folder or upload one in the sidebar.
Turn on "Show retrieval details" to see how each passage was ranked.

**Terminal**

```powershell
.\.venv\Scripts\python.exe app.py                  # default PDF
.\.venv\Scripts\python.exe app.py pdf\ccnp.pdf     # another PDF
.\.venv\Scripts\python.exe app.py --debug          # show retrieval details
```

The first run on a PDF creates embeddings (a few cents with
`text-embedding-3-small`); they are cached in `cache/` so later runs
start in about a second.

## Project layout

```
app.py                  terminal entry point
streamlit_app.py        web app
pdf_assistant/
  config.py             settings (models, chunk size, paths)
  ingest.py             PDF reading, chunking, embeddings, caching, indexes
  retrieval.py          FAISS + BM25 hybrid search with RRF
  llm.py                query rewriting and streamed answers
  cli.py                terminal chat loop
```
