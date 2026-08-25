# Two-Agent Grounded Q&A Assistant

A retrieval-augmented question answering system over the official **LangChain** and
**Qdrant** documentation. Two agents hand off to each other: a **Researcher** that
retrieves passages from a remote Qdrant collection, and a **Reviewer** that verifies
every claim in the drafted answer and sends it back once if anything is unsupported.

If the retrieved documentation does not support an answer, the system refuses instead
of guessing.

---

## Architecture

```
                    ┌──────────────┐
    question ─────► │  Researcher  │  embeds the question, searches Qdrant,
                    └──────┬───────┘  returns top-k passages + source URLs
                           │
                           ▼
                    ┌──────────────┐
              ┌───► │   Drafter    │  answers using ONLY those passages,
              │     └──────┬───────┘  cites [1] [2], or refuses
              │            │
              │            ▼
              │     ┌──────────────┐
              │     │   Reviewer   │  checks every claim against the passages
              │     └──────┬───────┘  returns APPROVED / REJECTED + reason
              │            │
              │      ┌─────┴─────┐
              └──────┤ REJECTED? │──── APPROVED ────► final answer
                 (max 1 revision)
```

The **Reviewer → Drafter** edge is a LangGraph *conditional edge*. That is what makes
this a genuine handoff loop rather than a single sequential pipeline. It fires at most
once (`MAX_REVISIONS = 1`), so the graph always terminates.

### Design decisions

| Decision | Value | Why |
|---|---|---|
| Chunk size | 800 chars | Roughly one coherent idea per chunk. Larger chunks dilute the embedding; smaller ones cut code examples in half. |
| Chunk overlap | 120 chars | A sentence that straddles a boundary would otherwise be lost to both chunks. |
| Retrieval `k` | 5 | Enough context for the Drafter without burying the relevant passage in noise. |
| Distance | Cosine | Embeddings are L2-normalised, so cosine similarity is the correct metric. |
| Embeddings | `all-MiniLM-L6-v2`, run locally | 384-dim, fast, free. No embedding API cost or key required. |
| LLM | `gemini-2.0-flash` | Generous free tier; low latency for an interactive chat. |
| Max revisions | 1 | The brief specifies the reviewer loops back once. Also guarantees termination. |

---

## Project structure

```
.
├── app.py                  Streamlit chat interface
├── ingest.py               fetch docs → chunk → embed → upsert to Qdrant
├── requirements.txt
├── .env.example            every required variable, no real values
├── data/
│   └── urls.txt            the corpus: LangChain + Qdrant doc URLs
├── src/
│   ├── config.py           all settings, loaded from .env
│   ├── llm.py              thin Gemini wrapper
│   ├── retriever.py        Qdrant search + passage formatting
│   ├── prompts.py          Drafter and Reviewer prompts
│   └── graph.py            the LangGraph state machine
├── tests/
│   ├── questions.py        100 documented test cases
│   └── run_tests.py        runs them all, writes logs + summary
└── logs/                   generated: test_runs.jsonl, results.md
```

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd <repo>
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create a Qdrant Cloud cluster

1. Sign up at <https://cloud.qdrant.io> and create a **free** cluster.
2. Open the cluster, go to the **Connect / API keys** panel in the Web UI.
3. Copy the cluster **URL** and generate an **API key**.

Reference: <https://qdrant.tech/documentation/web-ui/>

### 3. Get a Gemini API key

Free key from <https://aistudio.google.com/apikey>.

### 4. Configure

```bash
cp .env.example .env
```

Then fill in `QDRANT_URL`, `QDRANT_API_KEY` and `GOOGLE_API_KEY`.
`.env` is git-ignored — **no credentials are ever committed**.

### 5. Ingest the corpus

```bash
python ingest.py --recreate
```

This fetches every URL in `data/urls.txt`, strips navigation and boilerplate,
chunks the text, embeds it locally, and upserts it into your Qdrant collection.
Pages that fail are reported at the end and do not stop the run.

Smoke test first if you like:

```bash
python ingest.py --limit 5
```

### 6. Run the app

```bash
streamlit run app.py
```

Open <http://localhost:8501>.

---

## Using it

Ask a question about LangChain or Qdrant. The interface shows:

- **The grounded answer**, with `[n]` citations pointing at the passages used
- **Sources** — every retrieved passage with its similarity score and source URL
- **Reviewer verdict** — APPROVED or REJECTED with a one-line reason
- **Revision notice** — if the Reviewer sent the draft back

Try a question the docs cannot answer (`What is the capital of Japan?`) to see the
refusal path.

---

## Test suite

100 documented cases:

| Category | Count | Expected behaviour |
|---|---|---|
| `in_corpus` | 70 | Answer, with citations |
| `out_of_corpus` | 20 | Refuse |
| `ambiguous` | 10 | Refuse or answer strictly from the text — never invent |

```bash
python -m tests.run_tests                 # all 100
python -m tests.run_tests --limit 10      # quick check
python -m tests.run_tests --delay 2       # slower, for free-tier rate limits
```

Outputs:

- **`logs/test_runs.jsonl`** — one record per question containing the question, every
  retrieved chunk with its score and source URL, every draft attempt, the reviewer's
  verdict and reason, the final output, and latency.
- **`logs/results.md`** — pass rate, refusal rate per category, revision count,
  average and p95 latency, and a table of every failure.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Missing environment variables` | You have not created `.env` from `.env.example`. |
| `Collection not found` | Run `python ingest.py --recreate` first. |
| Ingestion reports many failed pages | A doc site changed its URLs. Edit `data/urls.txt`. |
| Rate-limit errors during tests | Re-run with `--delay 2`. |
| First query is slow | The embedding model loads on first use, then is cached. |

---

## Known limitations

- Retrieval is dense-only. Hybrid search (dense + sparse) would improve recall on exact
  API names — Qdrant supports it natively and it is the obvious next step.
- The corpus is a curated page list, not a full sitemap crawl, so coverage is deliberately
  narrow and current only as of ingestion time.
- No reranking. A cross-encoder over the top 20 would likely raise answer quality.
- The Reviewer is the same model as the Drafter, so the two share blind spots. A different
  model for review would make verification more independent.
