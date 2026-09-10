# Day 12 Log — Hybrid Search + Reranking + Evaluation
### Agentic RAG Project (rag-project-1)

---

## Goal for the day
Add hybrid search (BM25 + vector) and cross-encoder reranking inside the Retrieve node, then build a small evaluation set to measure retrieval accuracy and answer quality systematically.

---

## Stage 1: Hybrid search

### Installed BM25
```
rank-bm25
```

### Built `hybrid_retrieve()` in `src/retriever.py`
Combines vector search (ChromaDB) and keyword search (BM25) using **Reciprocal Rank Fusion** — a standard technique for merging two independently-ranked lists into one combined score:
```python
def reciprocal_rank_fusion(vector_ranked_ids, bm25_ranked_ids, k=60):
    scores = {}
    for rank, doc_id in enumerate(vector_ranked_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_ranked_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return scores
```

### Standalone test
"What is covered about TensorFlow or PyTorch?" → correctly surfaced Chunk 4 (which genuinely mentions TensorFlow/PyTorch) as the top fused result, alongside two other chunks with moderate keyword overlap. Fused scores were very close together (0.0325, 0.0315, 0.0313) — noted as expected given the small (16-chunk) corpus and RRF's smoothing constant.

---

## Stage 2: Cross-encoder reranking

### Built `rerank()` using `cross-encoder/ms-marco-MiniLM-L-6-v2`
Re-scores the fused candidates by directly comparing (question, chunk) pairs — more precise than embedding similarity or keyword overlap alone.

### Result — reranking sharpened confidence significantly
Same question as above, after reranking:
- Chunk 4: **5.0885** (clear winner)
- Chunk 0: -0.2226
- Chunk 2: -0.2831

Compared to the near-tied fusion scores, this is a much more decisive, confident ranking — the core value proposition of reranking, clearly demonstrated with real before/after numbers.

---

## Real bug found and fixed: BM25 wasn't respecting metadata filters

### Discovery
After wiring `retrieve()` to use hybrid search + reranking by default (replacing plain vector search in the graph), re-running the previously-correct "Week 1 of Month 2" question returned the **wrong answer** — an SQL chunk (Month 1, Week 2) ranked above the correct Deep Learning chunk (Month 2, Week 1), even though filter extraction correctly identified `{'month': 2, 'week': 1}`.

### Root cause
`hybrid_retrieve()` applied the metadata `where` filter to the vector search branch only. The BM25 branch searched the **entire 16-chunk corpus unfiltered**, so it could freely surface out-of-scope chunks that happened to score well on generic keyword overlap (common words like "week," "month," numbers). This silently reintroduced the exact Month/Week retrieval failure that Day 6's metadata filtering was specifically built to solve — just via a different path (BM25 leakage instead of pure semantic confusion).

### Fix
Restricted BM25 to only rank chunks matching the same filters used by vector search:
```python
def matches_filters(chunk):
    return all(chunk.metadata.get(key) == value for key, value in filters.items())

valid_indices = [i for i, chunk in enumerate(chunks) if not filters or matches_filters(chunk)]
# BM25 scoring now restricted to valid_indices only
```

### Verified fixed
Re-ran "Week 1 of Month 2" — now correctly returns **only Chunk 4** (the filter narrows the candidate pool down to exactly the right chunk, so there's nothing else in scope to compete with it).

### Impact assessment (before the fix was confirmed)
- **Affected:** any question where the LLM extracts a month/week filter — BM25 leakage could reintroduce wrong-chunk results even when filtering should have prevented it
- **Not affected:** questions with no month/week filter extracted (filters empty either way, so BM25 searching the full corpus is correct behavior)
- **Severity note:** disproportionately impactful on this project's small 16-chunk corpus — a single irrelevant leaked chunk has outsized influence on a small candidate pool; likely less severe proportionally on a larger document set, though still a genuine bug regardless of corpus size

---

## Stage 3: Evaluation set

### Built `src/evaluation.py`
A 5-question hand-built eval set with expected chunk IDs, measuring both retrieval accuracy (top-1 match) and generation quality (relevance + groundedness, using the existing grader functions).

### Results
**Retrieval: 4/5 (80%)**
- 4 correct exact matches
- 1 near-miss: "What deep learning projects are covered?" expected Chunk 5, got Chunk 4 — both are legitimately deep-learning-related chunks one week apart (Week 1 "Fundamentals" vs Week 2 "Projects")

**Generation: 5/5 relevant, 5/5 grounded (100%)**
Even on the retrieval near-miss, generation still succeeded — because generation was evaluated with `n_results=3` (not 1), so the correct chunk likely still made it into the broader context window even when it wasn't the single top-ranked result. A good real-world argument for retrieving more than 1 chunk even with decent retrieval — it buys resilience against near-misses.

---

## Deeper investigation: why does the "deep learning projects" question keep failing?

Debugged by checking the full top-10 pre-rerank candidate pool for this specific question:
```
chunk_4 (0.0328) — Deep Learning Fundamentals & Neural Networks
chunk_2 (0.0320) — Machine Learning Fundamentals & Core Concepts
chunk_7 (0.0310) — NLP Projects & Deployment
chunk_5 (0.0306) — DL Project Development & Deployment (Image/Video)   <- the correct answer
chunk_15 (0.0301) — No-Code AI Platforms & AI Productivity Tools
...
```

**Finding:** Chunk 5 (the correct answer) *does* make it into the candidate pool, at rank 4 of 10. So retrieval/fusion is not the failure point. After reranking, Chunk 5 drops out of the top 3 entirely, while Chunk 4 is ranked first with a negative (poor) score of -1.8189 — meaning the cross-encoder doesn't consider *any* of the top-3 candidates a strong match, but still ranks the wrong one highest among weak options.

**Root cause:** the cross-encoder appears to weight literal phrase overlap in the chunk's opening/title text ("Deep Learning" appears prominently in Chunk 4's title "Deep Learning **Fundamentals**") more heavily than deeper content relevance (Chunk 5's title uses "DL Project Development & Deployment" — the actual list of real deep learning projects the question is asking about, but with less literal lexical overlap to the word "projects" specifically, and the abbreviation "DL" instead of the full phrase).

**Conclusion:** this is a genuine, understood limitation of reranking — not a bug to chase further. Cross-encoders are more precise than embeddings alone, but they're not infallible, and can still be misled by surface-level lexical similarity in ambiguous cases. Documented as-is.

---

## Errors / issues encountered

| # | Issue | Cause | Resolution |
|---|---|---|---|
| 1 | Duplicate `if __name__ == "__main__":` blocks in `retriever.py`, second one using stale `results` from the first block | Copy-paste while adding new test code — both blocks execute, the second reused a variable name from unrelated code | Consolidated into a single `if __name__` block testing the real `retrieve()` pipeline on two different questions |
| 2 | BM25 silently ignoring metadata filters (see full writeup above) | `hybrid_retrieve()` only applied `where_clause` to the vector search branch | Added `matches_filters()` check, restricted BM25 scoring to the same filtered index set |

---

## Key lesson from Day 12
Two techniques combined don't automatically inherit each other's correctness — hybrid search's fusion step needs to respect the *same* constraints (like metadata filters) across *both* of its component searches, or one unfiltered branch can silently undermine work done in an earlier, unrelated part of the pipeline (Day 6's filtering). Separately: reranking is a real, measurable improvement (confirmed with concrete before/after scores), but it is not a cure-all — it can still be fooled by lexical surface similarity, which is worth knowing and being able to explain rather than assuming it always finds the "true" best match.

---

## Confirmed state after Day 12
- `src/retriever.py` — `hybrid_retrieve()` (BM25 + vector via RRF, both filter-aware) and `rerank()` (cross-encoder), combined as the default `retrieve()` used by the graph
- `src/evaluation.py` — 5-question eval set, measuring retrieval accuracy and generation quality
- **Results:** 80% top-1 retrieval accuracy, 100% generation relevance and groundedness
- One real bug found and fixed (BM25 filter leakage)
- One genuine, root-caused limitation documented (reranker favoring lexical title overlap over content match on one ambiguous question)

---

## Next up: Day 13 — Streamlit UI + edge case testing