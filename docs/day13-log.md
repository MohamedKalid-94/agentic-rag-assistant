# Day 13 Log — Streamlit UI + Edge Case Testing
### Agentic RAG Project (rag-project-1)

---

## Goal for the day
Build a Streamlit UI with full-scope functionality: PDF upload, vector store rebuild, question answering, and a visible reasoning trace — then stress-test it with real edge cases (unanswerable questions, non-structured documents, duplicate files).

## Summary
This was the densest debugging day of the project. Six real, distinct bugs were found and fixed — all discovered through actually using the UI, not through code review alone. This is genuinely valuable material: it shows the difference between code that runs and code that's been stress-tested against real user behavior.

---

## Part 1: Building the UI

### Refactored `vector_store.py` into a reusable function
Original script ran everything at import time (unsafe for a UI to call on demand). Wrapped into `rebuild_vector_store(documents_folder)`, returning the chunk count.

### Built `src/app.py`
Streamlit interface with:
- Sidebar: PDF upload (multiple files), list of current documents, "Rebuild vector store" button
- Main area: question input, Ask button, answer display, expandable "🔍 Reasoning trace" showing filters, attempts, relevance, groundedness, and retrieved chunks

### Installed Streamlit
```
streamlit
```

---

## Bug 1: Duplicate `app.py` files (root vs `src/`)
**Symptom:** blank page, only the "Deploy" button visible, no error in terminal.
**Cause:** an empty placeholder `app.py` existed at the project root from Day 1 planning; the real working code had been pasted into a *different* `app.py` inside `src/`. Running `streamlit run app.py` launched the empty root file.
**Fix:** deleted the root-level `app.py`; run the app going forward with `streamlit run src/app.py` (from the project root, so relative paths like `data/documents/` still resolve correctly while imports still find sibling modules in `src/`).

---

## Bug 2: Citations appearing on honest "I don't know" answers
**Symptom:** unanswerable questions correctly said "the context does not contain the answer" but still cited a source, which is misleading (implies that source supports the non-answer).
**Fix:** updated the generator prompt to explicitly instruct: *"If the context doesn't contain the answer, say so clearly and do NOT include a source citation in that case."*
**Verified:** re-tested with "capital of India/Australia" — clean non-answers, no citation.

---

## Bug 3: App crashing on LLM structured-output parsing failures
**Symptom:** asking "What is the capital of India?" crashed the whole app with `Error code: 400 - Parsing failed`. The `failed_generation` field showed the LLM had written out a long chain-of-thought ramble instead of returning clean structured JSON for the groundedness grade.
**Cause:** `grade_relevance()` and `check_groundedness()` had no error handling — any malformed structured output propagated straight up and killed the request.
**Fix:** wrapped both grader calls in try/except, defaulting to a safe fallback (`is_relevant=False` / `is_grounded=False`) on failure — "fail safe, not fail silent." A relevance failure triggers a retry; a groundedness failure triggers a regenerate, rather than trusting unverified output.
**Follow-up bug caught during this fix:** the first version of the except block only had a `print()` with no `return`, so Python implicitly returned `None` — causing a *different* crash (`'NoneType' object has no attribute 'is_grounded'`) downstream. Fixed by adding the explicit `return GroundednessGrade(...)` inside the except block.

---

## Bug 4: Non-WEEK-structured PDFs silently contributing zero chunks
**Symptom:** uploaded a resume PDF, rebuilt the vector store, asked about the candidate's email — got "context does not contain an email address," even though the resume genuinely contained one.
**Cause:** `split_documents()`'s `if not matches: continue` behavior (known and deferred since Day 5) — any page without a literal `WEEK` heading was dropped entirely, not just deprioritized. The resume produced **zero** chunks; nothing from it was ever stored.
**Confirmed via UI:** chunk count stayed at exactly 16 (unchanged) both times the resume was uploaded and rebuilt — direct proof the resume contributed nothing.
**Fix:** added a `RecursiveCharacterTextSplitter` fallback path — any page without WEEK headings now gets chunked normally (`chunk_size=1000`, `overlap=200`, `section: "General"`, `month`/`week`: `None`) instead of being dropped.
**Bonus discovery:** this fix also recovered previously-missing content from the *roadmap PDF itself* — its title page, overview page, job-titles section, and "Is this material right for you?" section had never been chunked either, since they also lack WEEK headings. Chunk count for the roadmap alone went from 16 → 34 once this was fixed — a real, positive side effect of fixing what looked like an edge case.

---

## Bug 5: Missing `context` variable in `generate_answer()`
**Symptom:** would have caused a `NameError: name 'context' is not defined` — caught during code review before it could be tested live.
**Cause:** while updating the citation-fallback logic, the line joining `context_parts` into `context` was accidentally dropped.
**Fix:** re-added `context = "\n\n---\n\n".join(context_parts)` after the loop. Also improved filename extraction to handle both `/` and `\` path separators (Windows compatibility).

---

## Bug 6: Duplicate documents processed twice under different filenames
**Symptom:** uploaded a renamed copy of the roadmap PDF; chunk count jumped from 34 to 58 unexpectedly on the *next* upload (a resume), and a follow-up test with the renamed duplicate alone showed no reduction — content was being fully reprocessed under its new name.
**Cause:** `load_documents()` treats every file as unique by filename only — no check for identical content. A renamed copy of an already-loaded file is processed as if it were entirely new.
**Fix:** added SHA-256 content hashing in `ingestion.py`. Each file's raw bytes are hashed before loading; if the hash matches one already seen, the file is skipped with a logged message (`[load_documents] Skipping '...' — identical content to '...'`).
**Verified across a 4-step test sequence:**
| Step | Chunks | Result |
|---|---|---|
| Roadmap only | 34 | baseline |
| + renamed duplicate roadmap | 34 | ✅ duplicate skipped |
| + resume added | 58 | ✅ genuinely new content added |
| renamed duplicate re-checked (resume still present) | 58 | ✅ duplicate still correctly skipped |

**Known limitation, accepted as out of scope:** this only catches *exact* byte-for-byte duplicates. A near-duplicate (same content, re-saved with minor formatting differences) would produce a different hash and not be caught. True near-duplicate detection would require content similarity comparison — a larger feature, not pursued today.

---

## Also fixed along the way: `rebuild_vector_store()` now clears stale data
**Cause:** the original rebuild logic used `upsert()` with position-based IDs (`chunk_0`, `chunk_1`, ...). If a rebuild produced fewer chunks than before, old leftover IDs from the previous run would never be removed — stale content could linger in the collection indefinitely.
**Fix:** `rebuild_vector_store()` now explicitly deletes the collection (`client.delete_collection(...)`) before creating a fresh one on every rebuild, guaranteeing no stale chunks ever persist across rebuilds.

---

## Key lesson from Day 13
Every one of today's six bugs was found by actually using the application the way a real user would — uploading unexpected files, asking adversarial questions, renaming files, checking chunk counts before and after. None of these would have been caught by re-running the same known-good test question repeatedly. Hands-on UI testing surfaces a fundamentally different class of bugs than unit-testing individual functions in isolation — both are necessary, but this day proved neither is sufficient alone.

---

## Confirmed state after Day 13
- `src/app.py` — full Streamlit UI: upload, rebuild, ask, reasoning trace (run via `streamlit run src/app.py` from project root)
- `src/vector_store.py` — `rebuild_vector_store()`, now clears stale data before every rebuild
- `src/chunking.py` — fallback chunking recovers previously-dropped non-WEEK content (roadmap's own hidden sections + any non-structured uploaded PDF)
- `src/ingestion.py` — SHA-256 content-hash deduplication, verified across multiple test sequences
- `src/grader.py` — both grading functions now fail safe (default to retry-triggering values) instead of crashing on malformed LLM output
- `src/generator.py` — citation logic fixed (no citation on non-answers, filename fallback for non-roadmap sources), missing `context` variable bug fixed
- Genuinely functional multi-document RAG: verified the system can hold and search across multiple distinct PDFs simultaneously, with correct per-source citation

---

## Next up: Day 14 — Polish, README, GitHub push, interview prep