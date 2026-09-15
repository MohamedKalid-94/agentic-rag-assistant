# Agentic RAG Assistant

A multi-document, self-correcting RAG (Retrieval-Augmented Generation) system built with **LangChain** and **LangGraph**, featuring hybrid search, cross-encoder reranking, and two independent self-correction loops — retrieval retry and groundedness-based regeneration.

Built over a 14-day guided learning sprint, going from zero hands-on experience with these libraries to a working, tested, multi-document agentic pipeline with a Streamlit UI.

---

## Table of Contents
- [What makes this "agentic"](#what-makes-this-agentic-not-just-rag)
- [The journey: what was built, when, and why](#the-journey-what-was-built-when-and-why)
- [Tech stack](#tech-stack)
- [Architecture](#architecture)
- [Setup](#setup)
- [Pipeline explained in detail](#pipeline-explained-in-detail)
- [Pros and cons](#pros-and-cons)
- [Key findings & limitations](#key-findings--limitations)
- [Future enhancements](#future-enhancements)
- [Reflection](#reflection)
- [Full build log](#full-build-log)

---

## What makes this "agentic," not just RAG

A naive RAG pipeline is a fixed sequence: retrieve → generate. This project goes further — it's a **LangGraph StateGraph** where the agent makes decisions at each step, not just executes a script:

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

Both loops were independently stress-tested with real failure cases, not just demonstrated on the happy path — see [Key findings](#key-findings--limitations).

---

## The journey: what was built, when, and why

This wasn't built top-down from a finished design — it was built incrementally, with each day's real findings shaping the next day's work. That history matters, because it's the honest story of *why* each piece exists.

**Days 1-4: Foundations.** Set up the environment, learned to make a basic LLM call through LangChain, loaded PDFs, and generated embeddings. Even here, real findings emerged — a "Week 1" query matched a "Week 3" chunk due to embeddings' weak grasp of literal, precise details. That single observation ended up justifying two major later decisions: metadata filtering (Day 6) and hybrid search (Day 12).

**Day 5: Vector storage, and a lesson in not trusting numbers blindly.** Built the ChromaDB store — and discovered Chroma's raw `distance` isn't cosine similarity unless you explicitly configure it. Fixing the metric improved the numbers, but the *underlying retrieval failure* (Week/Month confusion) persisted — proof that a correct distance metric alone doesn't fix a conceptual mismatch between semantic search and exact lookup.

**Day 6: Metadata filtering (an unplanned but necessary detour) + LangGraph fundamentals.** Built LLM-based extraction of month/week filters from natural language, combined with Chroma's `where` clause — this is what actually fixed the Week/Month retrieval problem, not better embeddings. Then learned LangGraph's core mechanics (StateGraph, nodes, conditional edges, loops) through small toy graphs before touching the real pipeline.

**Days 7-10: Building the real agent, one node at a time.** Refactored everything into proper modules (`retriever.py`, `nodes.py`, `graph.py`), then added grading (with a genuine retry loop, tested against an unanswerable question), generation (grounded, cited), and groundedness checking (tested against a deliberately fabricated answer about a fake "Dr. Andrew Ng" workshop — correctly caught). Day 10 was the milestone: the full 5-node loop worked end-to-end for the first time.

**Day 11: Trying to break it, on purpose.** Ran the retrieve-retry loop through the complete graph. Then deliberately tried to force a hallucination through two escalating stress tests (loosened prompt, then raised temperature + explicit instructions to speculate). The model stayed grounded both times — a genuinely positive, if inconclusive, finding about the system's real-world resilience.

**Day 12: Hybrid search, reranking, and a real bug.** Added BM25 + vector fusion and cross-encoder reranking — then discovered BM25 wasn't respecting metadata filters, silently reintroducing the exact Week/Month bug that Day 6 had fixed, through a completely different path. Fixed it, built an eval set, and honestly diagnosed a remaining near-miss down to the reranker specifically favoring title-level keyword overlap over actual content relevance.

**Day 13: The UI, and six real bugs found by actually using it.** Building the Streamlit interface surfaced problems that had never shown up in code review or repeated single-question testing: a duplicate empty `app.py`, citations appearing on non-answers, the whole app crashing on a rare LLM output-parsing failure, a resume PDF silently contributing zero chunks (a real, dropped-content bug — not just a theoretical one), a missing variable, and duplicate files being reprocessed under different names. Every one of these was found through hands-on testing, not static analysis.

**Day 14: Making it presentable.** Wrote this README, reorganized the flat file layout into packages that mirror the actual architecture, and centralized every scattered constant into one config file — which surfaced *yet another* recurring bug pattern (constants imported but never actually used in 7 different files) that only became visible by going through every file individually.

---

## Tech stack

| Component | Tool | Why |
|---|---|---|
| Orchestration | **LangGraph** | StateGraph, conditional edges, loops — the actual agentic layer |
| LLM calls, prompts, structured output | **LangChain** + **Groq** (`openai/gpt-oss-20b`) | Fast, free-tier LLM with structured output support |
| Document loading | `pypdf` via `langchain-community` | PDF text + metadata extraction |
| Chunking | Custom structure-aware splitter + `RecursiveCharacterTextSplitter` fallback | Handles both the primary structured document and arbitrary uploads |
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
│   └── documents/               # uploaded PDFs (SHA-256 deduplicated)
│   # data/chroma_db/ auto-created — gitignored
├── docs/                        # full 14-day build log
├── src/
│   ├── ingestion/loader.py      # PDF loading + content dedup
│   ├── processing/chunking.py   # structure-aware + fallback chunking
│   ├── embeddings/embedder.py   # local embedding model
│   ├── vectorstore/store.py     # ChromaDB storage (rebuild from scratch each time)
│   ├── retrieval/
│   │   ├── query_analyzer.py    # LLM-based metadata filter extraction
│   │   └── retriever.py         # hybrid search + reranking
│   ├── prompting/
│   │   ├── grader.py            # relevance + groundedness grading, query rewriting
│   │   └── generator.py         # grounded answer generation with citations
│   ├── agent/
│   │   ├── state.py             # shared AgentState schema
│   │   ├── nodes.py             # LangGraph node functions
│   │   └── graph.py             # compiled StateGraph — the real entry point
│   ├── evaluation/evaluation.py # hand-built eval set
│   ├── config.py                # centralized constants, absolute paths
│   └── app.py                   # Streamlit UI
├── requirements.txt
└── .env                         # GROQ_API_KEY (gitignored)
```

**Why organized this way:** each package maps to one stage of the RAG/agent pipeline. This means a new pipeline stage (e.g. a future "summarize" node) can be added without touching retrieval code, and a retrieval bug can be fixed without risking the agent orchestration layer.

---

## Setup

```powershell
git clone <this-repo>
cd rag-project-1
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
```
Free key at [console.groq.com](https://console.groq.com).

Run the UI:
```powershell
streamlit run src/app.py
```

Run the graph directly:
```powershell
cd src
python -m agent.graph
```

---

## Pipeline explained in detail

**1. Ingestion** — PDFs are loaded and SHA-256 hashed; identical content under a different filename is skipped, preventing duplicate storage.

**2. Chunking** — pages with explicit `WEEK`/`MONTH` headings are split into one chunk per week, carrying month/week/section as metadata. Pages without that structure (title pages, or entirely different document types like a resume) fall back to standard fixed-size chunking instead of being silently dropped.

**3. Embedding** — each chunk is converted to a 384-dimension vector using a local, free embedding model.

**4. Storage** — ChromaDB stores vectors + text + metadata, using cosine distance explicitly (not the default L2). The store is fully rebuilt (not incrementally patched) on every "rebuild" action, so stale data never lingers when the document set changes.

**5. Query analysis** — the LLM extracts explicit month/week mentions from the question into a structured filter, when present.

**6. Hybrid retrieval** — vector search and BM25 keyword search both run (respecting the same metadata filters), and their rankings are combined via Reciprocal Rank Fusion.

**7. Reranking** — the fused top-10 candidates are re-scored by a cross-encoder, which directly compares the question against each candidate for much higher-confidence final ranking.

**8. Grading** — the LLM judges whether the top retrieved chunks actually contain the answer. If not, the query is rewritten and retrieval retries (up to 3 times).

**9. Generation** — an answer is produced strictly from the retrieved context, with an explicit instruction to say "I don't know" rather than guess, and to cite sources.

**10. Groundedness check** — the LLM verifies the generated answer doesn't contain any claim unsupported by the retrieved context. If it does, the answer is regenerated (up to 2 times).

---

## Pros and cons

**Pros:**
- Genuinely agentic — the graph makes real decisions (retry, regenerate, give up gracefully) rather than following one fixed path
- Fails toward honesty, not fabrication — verified with adversarial testing, not just claimed
- Multi-document capable — content-deduplicated, handles both structured and unstructured PDFs
- Every design decision is backed by an observed, documented finding (not just following a tutorial)
- Modular architecture — each pipeline stage is independently testable and swappable
- Runs entirely on free tiers (Groq + local embeddings) — no cost to operate at this scale

**Cons:**
- Small-corpus tested only (16-58 chunks) — behavior at real-world document scale (thousands of chunks) is untested
- BM25 index is rebuilt from scratch on every retrieval call — fine at this scale, would need proper indexing/caching at larger scale
- Chunking is still fundamentally document-structure-dependent — the "good" path (WEEK/MONTH aware chunking) only applies to one document; everything else uses generic fallback chunking
- No conversation memory — each question is answered independently, no multi-turn context
- Reranker demonstrated to have a real, known failure mode (lexical title-overlap bias) — not corrected, only documented
- No authentication or access control on the Streamlit UI — not production-ready as-is

---

## Key findings & limitations

- **Metadata filtering mattered more than better embeddings** for structured queries — full before/after evidence across Days 5, 6, and 12.
- **Reranking significantly improves confidence but isn't infallible** — documented a specific case where it favored keyword overlap over true content relevance.
- **The model proved resistant to fabrication** under deliberate adversarial prompting (Day 11) — though this made the groundedness *regenerate* loop hard to trigger naturally; its correctness was independently verified with a synthetic fabricated answer instead.
- **Six real bugs surfaced only through hands-on UI testing** (Day 13) that static code review never caught.
- **Evaluation: 80% top-1 retrieval accuracy, 100% generation relevance/groundedness** on a 5-question hand-built set — with an honest, explainable reason for the one miss.
- **Deduplication is exact-match only** (file hash) — near-duplicate documents aren't caught.

---

## Future enhancements

- **Conversation memory** — support multi-turn follow-up questions using LangGraph's checkpointing
- **Larger-scale testing** — validate retrieval quality and BM25 performance on a corpus of thousands of chunks, not just dozens
- **Persistent BM25 index** — avoid rebuilding it from scratch on every single retrieval call
- **Near-duplicate detection** — move beyond exact file-hash matching to content-similarity-based deduplication
- **Broader document type support** — chunking strategies for other structured formats (tables, code, slides) beyond the current WEEK/MONTH-specific logic
- **Automated regression testing** — turn the manual eval set into a CI-style test that runs automatically on every change
- **Authentication** — basic access control before this could be shared beyond local/personal use

---

## Reflection

This project's real value wasn't writing code that worked on the first try — almost nothing here did. It was the pattern, repeated across 14 days: build something, test it against a case specifically designed to break it, find that it breaks in an interesting way, understand *why*, fix it, and write down what was learned. The Week/Month metadata bug alone got rediscovered and re-fixed three separate times across three different layers (raw vector search, hybrid fusion, and — nearly — reranking) precisely because each layer was tested independently rather than assumed to inherit the previous layer's correctness. That's the habit most worth carrying forward from this project: trusting evidence over assumption, and treating every "it works" as a claim to verify, not a fact to accept.

---

## Full build log

Every day of this build — every bug, every debugging session, every design decision, in full detail — is documented in `docs/day1-log.md` through `docs/day14-log.md`. These are real engineering logs, not polished summaries, and are the most honest record of how this system was actually built.

---

## License

Personal learning project — not licensed for reuse.