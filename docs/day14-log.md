# Day 14 Log — Polish, Restructure, README, Ship
### Agentic RAG Project (rag-project-1)

---

## Goal for the day
Final polish before calling the project done: write a proper README, reorganize the codebase to reflect its actual architecture, centralize scattered configuration, and confirm everything still works end-to-end.

---

## Part 1: README.md

Wrote a comprehensive README covering:
- What makes the project "agentic" (the two self-correction loops), with a text diagram of the full pipeline
- Tech stack table with justification for each choice
- Architecture / folder structure
- Setup instructions
- Chunking strategy explanation (structure-aware + fallback)
- Retrieval pipeline explanation (filter extraction → hybrid search → reranking)
- Self-correction loop explanation
- **Key findings & limitations** — pulled the most interesting, honest findings from all 13 prior day-logs into one place (the Month/Week metadata-filtering necessity, the reranker's lexical-overlap limitation, the model's resistance to fabrication under adversarial testing, the six Day-13 bugs, the eval results)
- Pointer to the full day-by-day build log in `docs/`

---

## Part 2: Restructured `src/` into architecture-based packages

### Before: flat structure
All ~15 files sitting directly in `src/`, with only filenames distinguishing their role.

### After: 8 packages by architectural layer
```
src/
├── ingestion/      (loader.py)
├── processing/     (chunking.py)
├── embeddings/     (embedder.py)
├── vectorstore/    (store.py)
├── retrieval/      (query_analyzer.py, retriever.py)
├── prompting/      (grader.py, generator.py)
├── agent/          (state.py, nodes.py, graph.py)
├── evaluation/     (evaluation.py)
├── config.py        # stays at src/ root — shared across all layers
└── app.py           # stays at src/ root — entry point, not one layer
```

### Deleted: `toy_graph_1.py`, `toy_graph_2.py`, `toy_graph_3.py`
All three were Day 6 learning artifacts, fully superseded by the real `agent/` package. Removed rather than fixed, since keeping broken/unused files around after a restructure is worse than not having them.

### Centralized `config.py`
Moved every previously-scattered constant (Groq model name, embedding model name, reranker model name, Chroma path, collection name, chunk size/overlap, max retry/regenerate attempts) into one file. Critically, made the path constants **absolute**, anchored to the project root via `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` — this was necessary because scripts are now run from different working directories (`src/` for `python -m agent.graph`, project root for `streamlit run src/app.py`), and relative paths like `"data/documents"` resolve differently depending on where the command is launched from.

---

## Part 3: The recurring bug pattern — found and fixed across 7 files

While reviewing each restructured file individually, the same bug shape kept appearing: **a config constant would be correctly imported at the top of a file, but the old hardcoded literal value was still being used in the actual function body** — meaning the import was present but silently doing nothing.

Found and fixed in:
| File | What was still hardcoded |
|---|---|
| `agent/nodes.py` | `3` and `2` instead of `MAX_RETRIEVAL_ATTEMPTS` / `MAX_GENERATION_ATTEMPTS` |
| `processing/chunking.py` | `chunk_size=1000, chunk_overlap=200` instead of `CHUNK_SIZE` / `CHUNK_OVERLAP`; `"data/documents"` in the `__main__` block |
| `vectorstore/store.py` | `"data/documents"` default param, `"data/chroma_db"` path, `"roadmap_collection"` name (3 separate spots) |
| `prompting/grader.py` | `"openai/gpt-oss-20b"` in 3 separate functions instead of `GROQ_MODEL`; broken flat imports in `__main__` |
| `prompting/generator.py` | `"openai/gpt-oss-20b"` instead of `GROQ_MODEL` |
| `retrieval/query_analyzer.py` | `"openai/gpt-oss-20b"` instead of `GROQ_MODEL` |
| `retrieval/retriever.py` | `"data/chroma_db"`, `"roadmap_collection"`, `"cross-encoder/ms-marco-MiniLM-L-6-v2"`, and `"data/documents"` — 4 separate hardcoded values, plus an entirely unused `GROQ_MODEL` import (this file makes no direct LLM calls) |

**This file (`retriever.py`) was the one that actually crashed the pipeline** — `FileNotFoundError: data/documents` when running `python -m agent.graph` from inside `src/`, since the hardcoded relative path no longer resolved correctly from that working directory. Fixed by replacing with `DOCUMENTS_FOLDER` from config (which is now absolute).

**Also found: `embeddings/embedder.py`'s `__main__` block** still had pre-restructure flat imports (`from ingestion import load_documents`) that would break if run standalone — fixed to `from ingestion.loader import load_documents` and `from processing.chunking import clean_text, split_documents`.

---

## Part 4: Cleanup

- Removed stray `src/data/chroma_db/` — a duplicate vector store folder accidentally created during earlier testing (before the absolute-path fix), when relative paths resolved into the wrong location depending on the working directory
- Removed multiple orphaned UUID-named segment folders inside the real `data/chroma_db/` — leftover from repeated `delete_collection()`/`create_collection()` cycles during Day 13's testing that weren't fully cleaned up on disk (cosmetic only — gitignored, no functional impact)
- Cleared `__pycache__` folders across the new package structure (harmless, auto-regenerating, already gitignored)

---

## Verification

Two full end-to-end tests, confirming the restructure changed nothing about actual behavior:

**1. Command line:**
```powershell
python -m agent.graph
```
Same correct result as every previous test — Chunk 4, Month 2/Week 1, fully grounded, correctly cited.

**2. Streamlit UI:**
```powershell
streamlit run src/app.py
```
Two live questions tested through the browser, both correct:
- "Week 2 Structured Query Language" → correctly answered, cited Month 1/Week 2
- A Month 2/Week 4 question → correctly answered, filters extracted as `{'month': 2, 'week': 4}`, retrieved `chunk_9`, grounded

---

## Errors / issues encountered

| # | Issue | Cause | Resolution |
|---|---|---|---|
| 1 | `FileNotFoundError: data/documents` crashing `agent/graph.py` | `retriever.py`'s `get_bm25_index()` used a relative path that no longer resolved correctly once the working directory changed (running from `src/` instead of project root) | Made all path constants in `config.py` absolute; replaced every hardcoded relative path across the codebase with the config constant |
| 2 | `ImportError: cannot import name 'GROQ_MODEL' from 'config'` | `config.py` wasn't fully saved with all constants on a prior edit | Rewrote the complete `config.py` from scratch, visually confirmed the save before re-running |
| 3 | Recurring "config imported but not used" pattern across 7 files (see Part 3 table) | Manual restructure work — imports were added correctly but the corresponding hardcoded literals in function bodies weren't swapped out in the same pass | Went through every restructured file individually, cross-checked each import against its actual usage, fixed all 7 |
| 4 | Stray `src/data/chroma_db/` folder | Created during earlier testing before the absolute-path fix, when relative paths pointed to the wrong location depending on working directory | Deleted; confirmed the real `data/` only exists at the project root going forward |

---

## Key lesson from Day 14
A restructure isn't finished when the folders are moved — it's finished when every file that *references* those moved pieces has been checked, not just the files that were physically relocated. The most dangerous bugs today weren't import errors (which fail loudly and immediately) — they were **silent no-ops**: a config constant correctly imported but never actually used, which doesn't error at all, just quietly ignores the "fix" and keeps the old broken behavior. Going file-by-file and manually cross-checking imports against actual usage — rather than trusting that "it imported successfully" means "it's being used correctly" — was what caught all seven instances of this pattern.

---

## Confirmed state after Day 14
- Full package-based architecture: `ingestion/`, `processing/`, `embeddings/`, `vectorstore/`, `retrieval/`, `prompting/`, `agent/`, `evaluation/`
- `config.py` centralizes every shared constant, using absolute paths robust to working-directory changes
- Comprehensive `README.md` covering architecture, setup, design decisions, and honest findings/limitations
- Three redundant scratch files removed
- Verified working end-to-end via both the command-line graph and the Streamlit UI
- All 14 days' work committed and pushed to GitHub with a clean commit history (Git correctly tracked file moves as renames, not delete+recreate)

---

## Project status: complete
14-day Agentic RAG build finished — full pipeline from PDF ingestion through hybrid retrieval, self-correcting generation, evaluation, and a working multi-document UI, with a properly organized codebase and complete documentation of every design decision, bug, and finding along the way.