# Day 5 Log — Vector Database
### Agentic RAG Project (rag-project-1)

---

## Goal for the day
Store chunk embeddings in a vector database (Chroma), run raw similarity search, and understand the difference between vector similarity and exact structured lookup.

---

## Steps completed

### 1. Upgraded chunking strategy (carried over from end of Day 4 work)
`chunking.py` was rewritten from fixed-size splitting (`RecursiveCharacterTextSplitter`) to **structure-aware chunking** — splitting on `WEEK` headings while carrying `MONTH` context forward as metadata across pages. This produced 16 chunks (one per week across 4 months) instead of the previous 33 fixed-size chunks, with richer metadata (`month`, `week`, `week_type`, `section`).

### 2. Installed ChromaDB
```
chromadb
```
```powershell
pip install chromadb
```

### 3. Wrote `src/vector_store.py`
Loads documents → cleans → chunks → embeds → stores in a persistent Chroma collection → runs a raw similarity query.

Key setup:
```python
client = chromadb.PersistentClient(path="data/chroma_db")

collection = client.get_or_create_collection(
    name="roadmap_chunks",
    metadata={"hnsw:space": "cosine"}
)
```

### 4. First run — revealed two real bugs

**Bug A — Chroma distance ≠ cosine similarity.**
Rank 1 came back with `Distance: 0.9906`, while the Day 4 manual cosine calculation for the same chunk gave `Similarity: 0.5047`. These are not the same number and are not interchangeable — Chroma's default distance metric is squared L2, not cosine, unless explicitly configured.

**Bug B — cosine setting didn't take effect on first attempt.**
Even after adding `metadata={"hnsw:space": "cosine"}`, re-running still produced the exact same `Distance: 0.9906` as before. Cause: Chroma's `hnsw:space` is locked in at collection **creation** time and can't be changed on an existing collection — `get_or_create_collection` was silently reusing the old collection (created before the cosine setting was added) instead of applying the new setting.

**Fix:** deleted `data/chroma_db/` entirely and re-ran from scratch, forcing a fresh collection to be created with the cosine setting actually applied.

### 5. Also found: footer regex bug still active
The PDF footer text was still visible inside chunk previews (`"AI Coach John | PROITBRIDGE Page 5 of 24"`), confirming the `clean_text()` regex in `chunking.py` had a stray `\*` character breaking the match (carried over from an earlier edit). Fixed by removing the erroneous `\*` characters from the three regex segments.

### 6. Re-ran after both fixes
```powershell
python src/vector_store.py
```
Result:
- Chroma collection metadata confirmed: `{'hnsw:space': 'cosine'}`
- Distances now in a tight, sensible range (0.44–0.59) instead of the earlier 0.99–1.19 — consistent with `1 - cosine_similarity` for moderately related text
- Footer text no longer present in chunk previews — clean data confirmed

---

## Key finding: distance metric fix improved data quality, but not the core retrieval problem

Even after both fixes (correct cosine metric + clean text), the query *"What topics are covered in Week 1 of Month 2?"* **still** did not return the actually correct chunk (Month 2, Week 1 — "Deep Learning Fundamentals & Neural Networks") anywhere in the top 5 results.

This confirms the real lesson of Day 5: **fixing the distance metric and cleaning the data improves retrieval quality overall, but does not solve the fundamental mismatch between semantic similarity search and exact structured lookup.** The month/week information the question is asking for already exists explicitly in metadata (`{'month': 2, 'week': 1}`) — this kind of question should eventually be handled with **metadata filtering + vector search combined**, not vector search alone. This directly sets up Day 6.

---

## Errors / issues encountered

| # | Issue | Cause | Resolution |
|---|---|---|---|
| 1 | Confused Chroma's `distance` output with cosine similarity | Assumed Chroma always returns cosine similarity by default | Learned: for a distance metric, smaller = more similar; for cosine similarity, larger = more similar — they are not interchangeable numbers |
| 2 | Cosine metric setting had no effect after first applying it | `hnsw:space` is fixed at collection creation; `get_or_create_collection` was silently returning the old (pre-cosine) collection instead of creating a new one | Deleted `data/chroma_db/` entirely and re-ran to force fresh collection creation |
| 3 | PDF footer text still appearing in chunk content | The `clean_text()` regex still had a stray `\*` character from an earlier edit, causing the match to silently fail | Removed the stray `\*` characters from the regex pattern |
| 4 | Renamed vector store folder from `data/chroma` to `data/chroma_db` | Needed a clean folder for the fresh collection | Updated `.gitignore` to exclude `data/chroma_db/` |

---

## Known, deliberately deferred issues (not fixed today — will be addressed in later days)
- `chunking.py` still uses `if not matches: continue`, silently dropping any PDF page with no `WEEK` heading (title page, overview, job-titles section, etc.) — real content loss, intentionally left for a later chunking revision
- Week-based chunks have no upper size limit, so longer weeks risk truncation by MiniLM's ~256-token input limit during embedding — noted, not yet fixed
- `week_type` is `None` for weeks without a parenthetical type — turned out to be a non-issue: ChromaDB silently omits `None`-valued metadata keys rather than erroring on upsert, confirmed by inspecting actual stored metadata

---

## Key lesson from Day 5
A distance number alone means nothing without knowing which distance metric produced it — always confirm and explicitly set the metric when using a vector database. Separately: correct retrieval infrastructure (working DB, correct metric, clean data) does not automatically mean correct retrieval *results* — semantic search has real, structural limitations for exact/structured questions that no amount of data cleaning will fix. That gap is what metadata filtering exists to close.

---

## Confirmed state after Day 5
- `src/vector_store.py` — stores chunks in ChromaDB with explicit cosine distance metric, runs raw similarity search
- `data/chroma_db/` — persistent vector store (excluded from git via `.gitignore`)
- 16 chunks stored with full metadata (month, week, week_type, section, source, page)
- Confirmed retrieval pipeline works end-to-end: query → embed → search → top-K results with distance + metadata + content
- Confirmed (with real evidence) that vector similarity ≠ exact structured/metadata lookup

---

## Next up: Day 6 — Retrieval quality / metadata filtering
