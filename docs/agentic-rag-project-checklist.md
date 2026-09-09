# Agentic RAG Project Checklist
### Multi-PDF Agentic RAG Assistant — LangChain + LangGraph

---

## Part 1 — Theory to be strong on before you start

### LLM fundamentals
- [x] Embeddings — text mapped to a vector space where semantic similarity = geometric closeness
- [x] Tokens & context windows — why chunking exists at all
- [x] How LLMs generate text (next-token prediction) — enough to understand why hallucination happens

### Retrieval theory
- [x] Similarity search — cosine similarity vs Euclidean distance, what "top-k" means
- [x] Why chunk size/overlap matters — too small loses context, too large dilutes relevance
- [x] Sparse vs dense retrieval — BM25 (keyword) vs vector search (semantic), why hybrid combines both
- [x] What reranking solves — fast/approximate initial retrieval, then slower/precise reranking

### RAG-specific concepts
- [x] Grounding / faithfulness — every claim traces back to retrieved context
- [x] Hallucination — claims not supported by given context
- [x] Why naive RAG fails — no check on retrieval quality, no check on answer quality

### Agentic AI concepts
- [x] Chain (fixed sequence) vs agent (makes decisions about what to do next)
- [x] What "state" means in LangGraph — the shared data object every node reads/writes
- [x] Conditional edges — how a graph decides which node runs next
- [x] Why loops matter — self-correction is what makes it "agentic," not just a pipeline

### Evaluation theory
- [x] Precision/recall applied to retrieval
- [x] Why retrieval and generation are evaluated separately

---

## Part 2 — 14-Day Build Plan

### Week 1 — Core RAG fundamentals + LangGraph basics
- [x] **Day 1:** RAG + agentic RAG concepts overview, environment setup (langchain, langgraph, chromadb, pypdf, LLM + embedding API keys)
- [x] **Day 2:** Document loading & text extraction from PDFs — inspect what gets lost (tables, headers)
- [x] **Day 3:** Chunking strategies — fixed-size vs recursive vs semantic, test overlap settings
- [x] **Day 4:** Embeddings — generate vectors for chunks, inspect similarity manually
- [x] **Day 5:** Vector database setup (Chroma) — store chunks with metadata, run raw similarity search
- [x] **Day 6:** LangGraph fundamentals — StateGraph, nodes, edges, conditional edges (build 2-3 toy graphs)
- [x] **Day 7:** Build the Retrieve node using your vector store as a graph node

### Week 2 — Agentic layer (routing, self-correction) + polish
- [ ] **Day 8:** Build the Grade relevance node (LLM judges retrieved chunks) + conditional edge back to Retrieve with a rewritten query
- [ ] **Day 9:** Build the Generate answer node — prompt template that answers strictly from context and cites sources
- [ ] **Day 10:** Build the Check groundedness node (LLM verifies no hallucination) + conditional edge back to Generate on failure
- [ ] **Day 11:** Wire the full graph end-to-end (Query → Retrieve → Grade → Generate → Check → Finalize), test with real questions including ones that should trigger both loops
- [ ] **Day 12:** Add hybrid search (BM25 + vector) and reranking inside the Retrieve node; build a small eval set, measure retrieval relevance + answer faithfulness
- [ ] **Day 13:** Streamlit UI — upload PDFs, ask questions, show the agent's reasoning trace; test edge cases (no answer in docs, vague questions)
- [ ] **Day 14:** README explaining the architecture and why it's agentic (not naive RAG), clean GitHub push, prepare a full interview walkthrough

---

## Part 3 — Tech stack

| Component | Tool |
|---|---|
| Orchestration | `langgraph` (StateGraph, conditional edges) |
| LLM calls, prompts, tool wrapping | `langchain` |
| Document loading | `pypdf` / `langchain-community` loaders |
| Embeddings + vector store | OpenAI/Gemini embeddings + `chromadb` |
| Web search fallback (optional) | Tavily API |
| UI | `streamlit` |

---