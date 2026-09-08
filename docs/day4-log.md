# Day 4 Log — Embeddings
### Agentic RAG Project (rag-project-1)

---

## Goal for the day
Generate embeddings for chunked documents, confirm the embedding model works correctly, and validate embedding quality before building the vector database (Day 5).

---

## Embedding provider decision

Considered three options: local `sentence-transformers` (free, no API key), Google Gemini embeddings (free tier, needs a key), or OpenAI (paid). Chose **local sentence-transformers** — no extra account to manage, no rate limits, good enough quality for this project.

**Confirmed this choice doesn't affect the agentic/LangGraph layer:** embeddings only feed the vector store's similarity search. LangGraph's orchestration (state, nodes, conditional edges, loops) and the agentic reasoning (grading, generating, checking groundedness) are handled entirely by the LLM (Groq) — completely separate concerns. LangChain's common `Embeddings` interface also means swapping providers later would be a one-line change, not a redesign.

---

## Steps completed

### 1. Installed packages
Added to `requirements.txt`:
```
sentence-transformers
langchain-huggingface
numpy
```
```powershell
pip install sentence-transformers langchain-huggingface numpy
```

### 2. Wrote `src/embeddings.py`
```python
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np


def get_embedding_model():
    """Returns a local embedding model (free, runs on CPU, no API key needed)."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### 3. First test — 3 sample chunks
Embedded chunk 0, 1, and 20 to sanity-check the model:
- Vector dimension: 384 (fixed size regardless of text length)
- Similarity chunk 0 vs chunk 1 (adjacent, related topic): **0.80**
- Similarity chunk 0 vs chunk 20 (far apart, different topic): **0.38**

Confirmed embeddings capture real semantic meaning — related content scores meaningfully higher than unrelated content.

### 4. Full test — all 33 chunks
Embedded every chunk and built a full 33×33 cosine similarity matrix:
- Average pairwise similarity: **0.3643**
- Similarity range: **0.0323 to 0.8215**
- Most similar pair: Chunk 22 & 23 — both about job title categories (genuinely related)
- Least similar pair: Chunk 11 & 31 — RAG/technical content vs. a marketing-style section header (genuinely unrelated)

A moderate average with a wide range indicated the document has genuinely distinct topics, not repetitive content — a good sign for retrieval quality.

### 5. Query embedding test
Tested `embedder.embed_query()` (the method used for real user questions, as opposed to `embed_documents()` used for chunks) with the question *"What topics are covered in Week 1?"*:
- Best matching chunk: **Chunk 4 — Week 3 content**, not Week 1 (similarity 0.4260)

---

## Key finding: a real limitation of semantic search

The Week 1 question matching a Week 3 chunk is **not a bug** — it's a genuine, expected weakness of dense embedding models. Small embedding models are good at capturing general topic/meaning ("this is curriculum content") but weak at distinguishing precise literal details like specific numbers ("Week 1" vs "Week 3" look nearly identical semantically). 

This directly motivates **Day 12's planned hybrid search** (combining this semantic search with keyword-based BM25 search) — BM25 would catch the literal "Week 1" match that pure embeddings missed. Good to have observed this firsthand rather than just reading about why hybrid search exists.

---

## Errors / issues encountered

| # | Issue | Cause | Resolution |
|---|---|---|---|
| 1 | `UserWarning: huggingface_hub cache-system uses symlinks...` and `unauthenticated requests to HF Hub` warnings on first run | Windows doesn't support symlinks by default; no HF_TOKEN set | Harmless — ignored. Model still downloaded and cached correctly (~90MB, one-time download) |
| 2 | "Least similar pair" incorrectly showed Chunk 0 vs Chunk 0 (similarity -1.0000) | Bug in the min-similarity search: the diagonal was set to `-1` to exclude self-comparison from the *max* search, but that same `-1` then became the global minimum, so `argmin` kept picking the diagonal | Used two separate copies of the similarity matrix — one with the diagonal set to `-1` for the max search, another with the diagonal set to `2` (above any real cosine similarity) for the min search |

---

## Key lesson from Day 4
Embeddings capture *general meaning*, not exact keywords or numbers. A model can correctly cluster topically related content while still failing on a specific literal detail (like "Week 1" vs "Week 3"). This is exactly why production RAG systems combine semantic search with keyword-based search (hybrid search) rather than relying on embeddings alone — a lesson that's much more concrete after seeing it fail on a real query than it would be from just reading about it.

---

## Confirmed state after Day 4
- `src/embeddings.py` — local embedding model (`all-MiniLM-L6-v2`, 384 dimensions, free, no API key)
- Validated on all 33 chunks: meaningful similarity spread (0.03–0.82, avg 0.36)
- Confirmed and documented a real semantic-search limitation, directly justifying a later step in the roadmap (hybrid search, Day 12)

---

## Next up: Day 5 — Vector database