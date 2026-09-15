# Agentic RAG Assistant

A multi-document, self-correcting RAG (Retrieval-Augmented Generation) system built with **LangChain** and **LangGraph**, featuring hybrid search, cross-encoder reranking, and two independent self-correction loops — retrieval retry and groundedness-based regeneration.

Built over a 14-day guided learning sprint covering RAG fundamentals through to a working agentic pipeline with a Streamlit UI.

---

## What makes this "agentic," not just RAG

A naive RAG pipeline is a fixed sequence: retrieve → generate. This project goes further — it's a **LangGraph StateGraph** where the agent makes decisions at each step:

```
User query
    │
    ▼
Retrieve  (hybrid search: vector + BM25, cross-encoder reranked)
    │
    ▼
Grade relevance  (LLM judges: do these chunks actually answer the question?)
    │
    ├── not relevant ──► Rewrite query ──► back to Retrieve   (max 3 attempts)
    │
    ▼ (relevant)
Generate answer  (strictly grounded in retrieved context, with citations)
    │
    ▼
Check groundedness  (LLM verifies: is every claim actually supported by context?)
    │
    ├── not grounded ──► Regenerate   (max 2 attempts)
    │
    ▼ (grounded)
Final answer
```

Both loops were independently stress-tested with real failure cases (see [Findings](#key-findings--limitations) below) — not just demonstrated on the happy path.

---

## Tech stack

| Component | Tool | Why |
|---|---|---|
| Orchestration | **LangGraph** | StateGraph, conditional edges, loops — the actual agentic layer |
| LLM calls, prompts, structured output | **LangChain** + **Groq** (`openai/gpt-oss-20b`) | Fast, free-tier LLM with structured output support |
| Document loading | `pypdf` via `langchain-community` | PDF text + metadata extraction |
| Chunking | Custom structure-aware splitter + `RecursiveCharacterTextSplitter` fallback | See [Chunking strategy](#chunking-strategy) below |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free, no API key) | 384-dim embeddings, runs on CPU |
| Vector store | **ChromaDB** (persistent, cosine distance) | Local, simple, metadata filtering support |
| Keyword search | `rank-bm25` | Combined with vector search via Reciprocal Rank Fusion |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precise relevance scoring on top of fused candidates |
| UI | **Streamlit** | Upload PDFs, ask questions, inspect the full reasoning trace |

---

## Architecture

```
rag-project-1/
├── data/
│   └── documents/          # uploaded PDFs (gitignored: chroma_db/)
├── src/
│   ├── ingestion.py         # PDF loading + SHA-256 content dedup
│   ├── chunking.py          # structure-aware chunking + fallback splitter
│   ├── embeddings.py        # local embedding model
│   ├── vector_store.py      # ChromaDB storage, rebuild-from-scratch each time
│   ├── query_analyzer.py    # LLM-based metadata filter extraction
│   ├── retriever.py         # hybrid search (BM25 + vector) + reranking
│   ├── grader.py            # relevance grading + groundedness checking + query rewriting
│   ├── generator.py         # grounded answer generation with citations
│   ├── evaluation.py        # small hand-built eval set
│   ├── state.py             # shared LangGraph state schema
│   ├── nodes.py              # LangGraph node functions
│   ├── graph.py              # the compiled StateGraph — the real entry point
│   └── app.py                 # Streamlit UI
├── docs/                     # day-by-day build log (14 days, bugs found/fixed, findings)
├── requirements.txt
└── .env                     # GROQ_API_KEY (gitignored)
```

**Why this split matters:** each file has one job (single responsibility). `retriever.py` doesn't know about LangGraph; `nodes.py` doesn't know about ChromaDB internals. This meant new pipeline stages (grading, generation, groundedness) could be added without ever touching retrieval code again.

---

## Setup

```powershell
git clone <this-repo>
cd rag-project-1
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

Run the UI:
```powershell
streamlit run src/app.py
```

Or run the graph directly from the command line:
```powershell
python src/graph.py
```

---

## Chunking strategy

The main test document (a 24-page course roadmap) has explicit `WEEK`/`MONTH` structure, so chunking is **structure-aware**: each week becomes one chunk, carrying month/week/section metadata forward — enabling precise metadata filtering later.

Any page (or uploaded PDF) **without** that structure falls back to standard fixed-size splitting (`chunk_size=1000`, `overlap=200`), tagged with `section: "General"`. This means the system handles both the primary structured document and arbitrary unstructured uploads (resumes, articles, etc.) without silently dropping content.

---

## Retrieval pipeline

1. **LLM-based filter extraction** — the query is analyzed for explicit month/week mentions using structured LLM output (`query_analyzer.py`), producing a Chroma `where` filter when applicable
2. **Hybrid search** — vector search (ChromaDB, cosine distance) and BM25 keyword search run in parallel, both respecting the same metadata filters, then combined via **Reciprocal Rank Fusion**
3. **Cross-encoder reranking** — the fused top-10 candidates are re-scored by directly comparing (query, chunk) pairs, producing a much more confident final ranking than fusion scores alone

---

## Self-correction loops

**Retrieval loop:** if the grader judges retrieved chunks as not relevant, the query is rewritten by the LLM and retrieval is retried (up to 3 attempts) before falling through to generation anyway with a best-effort answer.

**Generation loop:** if the groundedness checker detects the generated answer contains claims not supported by the retrieved context, the answer is regenerated (up to 2 attempts) before falling through with the best available answer.

Both loops fail toward **honesty over fabrication** — an unanswerable question produces "the context does not contain the answer," never a guessed response.

---

## Key findings & limitations

- **Metadata filtering matters more than better embeddings for structured queries.** A question like "Week 1 of Month 2" reliably failed under pure semantic search — even after fixing the distance metric and cleaning the data — until LLM-extracted metadata filters were added. Full before/after evidence in `docs/day5-log.md` and `docs/day6-log.md`.
- **Reranking significantly improves confidence, but isn't infallible.** On one ambiguous test question, the cross-encoder favored title-level keyword overlap over actual content relevance, dropping the correct chunk out of the top 3. Documented in `docs/day12-log.md`.
- **This project's LLM proved resistant to fabrication** even under deliberate adversarial prompting (raised temperature + explicit instructions to speculate) — the groundedness safety net's *logic* was independently verified with a synthetic fabricated answer, even though triggering a real fabrication naturally proved difficult. See `docs/day11-log.md`.
- **Six real bugs were found through hands-on UI testing** in a single session (Day 13) that wouldn't have surfaced from re-running the same known-good test question: silent content-dropping on non-structured PDFs, duplicate-file reprocessing, citation-on-non-answer, and unhandled LLM output-parsing crashes. Full writeup in `docs/day13-log.md`.
- **Evaluation results:** 80% top-1 retrieval accuracy, 100% generation relevance and groundedness on a 5-question hand-built eval set (`src/evaluation.py`). The one retrieval miss is a defensible near-ambiguity between two topically adjacent chunks, not a random failure.
- **Known, accepted limitation:** deduplication is exact-match only (SHA-256 file hash) — near-duplicate documents (same content, re-saved with minor formatting differences) are not caught.

---

## Full build log

Every day of this build — including every bug encountered, every debugging session, and every design decision — is documented in `docs/day1-log.md` through `docs/day13-log.md`. These are written as detailed engineering logs, not polished summaries, and are the most honest record of how this system was actually built.

---

## License
Personal learning project — not licensed for reuse.