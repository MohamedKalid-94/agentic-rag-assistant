# Day 7 Log — Real Retrieve Node as a Proper Module
### Agentic RAG Project (rag-project-1)

---

## Goal for the day
Refactor the inline retrieval logic from the Day 6 toy graph into clean, reusable modules that match the planned project architecture — and wire it into a real LangGraph `StateGraph` entry point.

---

## Why this refactor mattered
Up to this point, retrieval logic (embed query → extract filters → query Chroma) lived inline inside `toy_graph_3.py`. That worked for learning, but wasn't reusable — both `app.py` and the growing agent graph need the same retrieval logic, and duplicating it would mean fixing bugs in multiple places. This day moved that logic into dedicated modules matching the originally planned structure (`state.py`, `retriever.py`, `nodes.py`, `graph.py`).

---

## Steps completed

### 1. `src/state.py` — shared state schema
```python
from typing import TypedDict


class AgentState(TypedDict):
    """Shared state for the agentic RAG graph."""
    question: str
    filters: dict
    retrieved_chunks: list          # list of (id, text, distance, metadata)
    attempts: int
```
Kept minimal intentionally — will grow in later days (relevance grade, generated answer, groundedness check).

### 2. `src/retriever.py` — reusable retrieval logic
Extracted the retrieval logic (vector store connection, query embedding, filter extraction, Chroma query) into a standalone `retrieve()` function that returns `(chunks, filters_used)`, callable from anywhere in the project — not just from inside a graph node.

Tested standalone:
```powershell
python src/retriever.py
```
Result: correctly returned Chunk 4 (Month 2, Week 1, "Deep Learning Fundamentals & Neural Networks") with filters `{'month': 2, 'week': 1}` — same correct result as the Day 6 toy graph, now from clean reusable code.

### 3. `src/nodes.py` — the LangGraph node wrapping the retriever
```python
from state import AgentState
from retriever import retrieve


def retrieve_node(state: AgentState) -> AgentState:
    """LangGraph node: retrieves chunks for the current question."""
    chunks, filters = retrieve(state["question"])
    state["retrieved_chunks"] = chunks
    state["filters"] = filters
    state["attempts"] += 1
    print(f"[retrieve_node] Attempt {state['attempts']} — {len(chunks)} chunks found (filters: {filters})")
    return state
```
This is the pattern every future node will follow: take `AgentState` in, call the real logic function, update state, return it.

### 4. `src/graph.py` — the real graph entry point
```python
from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import retrieve_node


builder = StateGraph(AgentState)
builder.add_node("retrieve", retrieve_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", END)  # grading/looping comes in later days

graph = builder.compile()
```
This replaces the toy graphs as the actual project graph going forward — future days add nodes here (`grade`, `generate`, `check_groundedness`) rather than rewriting retrieval logic.

Tested end-to-end:
```powershell
python src/graph.py
```
Result: same correct Chunk 4 result, now flowing through the full modular chain (`state.py` → `retriever.py` → `nodes.py` → `graph.py`).

### 5. Fixed a stale reference file along the way
`test_chroma.py` (a standalone sanity-check script, unrelated to the main pipeline) still pointed at the old path/collection name from before the Day 5 fixes:
- Old: `path="data/chroma"`, `collection_name="agentic_rag"`
- Fixed to: `path="data/chroma_db"`, `collection_name="roadmap_collection"`

Confirmed working: `Chunks stored: 16`

---

## Errors / issues encountered

| # | Issue | Cause | Resolution |
|---|---|---|---|
| 1 | `test_chroma.py` would have failed with a "collection not found" error if run | File predated the Day 5 path/collection-name fixes and was never updated afterward | Updated path to `data/chroma_db` and collection name to `roadmap_collection`, matching what `vector_store.py` actually creates |

No new bugs introduced during the refactor itself — the retrieval logic carried over cleanly from the working Day 6 toy graph version.

---

## Key lesson from Day 7
Moving from "one script that does everything" to "small modules that each do one job" doesn't change *what* the code does — it changes how easy it is to extend. Every future day (grading, generation, groundedness checking) now just means adding one function to `nodes.py` and one node + edge to `graph.py`, without touching retrieval logic again. This is the same separation-of-concerns principle from Day 3 (`ingestion.py` vs `chunking.py`), now applied one level up at the graph/agent level.

---

## Confirmed state after Day 7
- `src/state.py` — shared `AgentState` schema
- `src/retriever.py` — reusable `retrieve()` function, callable standalone or from a node
- `src/nodes.py` — `retrieve_node`, the first real LangGraph node
- `src/graph.py` — real project graph, replacing the toy graphs as the actual entry point
- `toy_graph_1.py`, `toy_graph_2.py`, `toy_graph_3.py` — kept as learning/reference scratch files, no longer part of the live pipeline
- Confirmed end-to-end: real graph correctly retrieves Chunk 4 for "Week 1 of Month 2"

---

## Next up: Day 8 — Build the Grade relevance node (LLM judges retrieved chunks) + conditional edge back to Retrieve with a rewritten query