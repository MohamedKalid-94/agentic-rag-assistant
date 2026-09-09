# Day 6 Log — Metadata Filtering + LangGraph Fundamentals
### Agentic RAG Project (rag-project-1)

---

## Note on scope
This day ended up covering two things: an unplanned detour into **metadata filtering** (a direct follow-up to Day 5's finding that pure vector search fails on structured questions), and then the **original Day 6 checklist item — LangGraph fundamentals**. Both are logged here together since they happened in the same working session, but they're conceptually distinct.

---

## Part 1: Metadata Filtering (unplanned detour)

### Goal
Fix the retrieval failure discovered in Day 5 — pure semantic search couldn't find the correct chunk for "What topics are covered in Week 1 of Month 2?" even with clean data and a correct distance metric.

### Step 1: Understand Chroma's `where` filter
Chroma supports pre-filtering which chunks are even considered before similarity ranking:
```python
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={"month": 2, "week": 1}
)
```

### Step 2: Manual filter test
Confirmed that with a hand-written filter, Rank 1 correctly returned Chunk 4 (Month 2, Week 1).

### Step 3: The real problem — where does the filter come from?
In a real system, users type plain questions, not filter dicts. Decided to build **LLM-based filter extraction** rather than regex, since it's more flexible and mirrors what a LangGraph "query analysis" node would do later.

### Step 4: Built `src/query_analyzer.py`
```python
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
from langchain_groq import ChatGroq

load_dotenv()


class QueryFilter(BaseModel):
    """Structured filters extracted from a user's natural language question."""
    month: Optional[int] = Field(default=None, description="Month number (1-4) if mentioned, else null")
    week: Optional[int] = Field(default=None, description="Week number (1-4) if mentioned, else null")


def get_filter_extractor():
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    return llm.with_structured_output(QueryFilter)


def extract_filters(question: str) -> dict:
    extractor = get_filter_extractor()
    result = extractor.invoke(
        f"Extract the month and week numbers mentioned in this question, if any: {question}"
    )
    where_clause = {}
    if result.month is not None:
        where_clause["month"] = result.month
    if result.week is not None:
        where_clause["week"] = result.week
    return where_clause
```

Tested standalone with 4 questions — all extracted correctly:
- "Week 1 of Month 2" → `{'month': 2, 'week': 1}`
- "deep learning projects" (no specifics) → `{}`
- "Week 3?" → `{'week': 3}`
- "Month 4?" → `{'month': 4}`

### Step 5: Wired into `vector_store.py`
Combined LLM-extracted filters with the vector query:
```python
filters = extract_filters(query)
if filters:
    where_clause = filters if len(filters) == 1 else {"$and": [{k: v} for k, v in filters.items()]}
    results = collection.query(query_embeddings=[query_embedding], n_results=5, where=where_clause)
else:
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
```

### Result — the fix confirmed working
Query: *"What topics are covered in Week 1 of Month 2?"*
- Extracted filters: `{'month': 2, 'week': 1}`
- **Rank 1: Chunk 4 — Month 2, Week 1, "Deep Learning Fundamentals & Neural Networks"** — the correct answer, finally.

Three-stage comparison across Days 5-6:
| Attempt | Result |
|---|---|
| Pure vector search (Day 5) | Wrong, not even in top 5 |
| Vector search + correct cosine metric (Day 5 retry) | Still wrong — confirmed it wasn't a metric issue |
| Vector search + LLM-extracted metadata filter (Day 6) | **Correct — Rank 1** |

### Key insight
Chunk 4's distance (0.6877) was actually *higher* (less similar) than some of the wrong answers returned in Day 5. The filter didn't make Chunk 4 more semantically similar — it restricted the search space so only the structurally correct chunk was even considered. Filtering and similarity ranking are separate mechanisms working together, not the same thing.

---

## Part 2: LangGraph Fundamentals (original Day 6 checklist item)

### Goal
Learn StateGraph, nodes, edges, and conditional edges by building small toy graphs before applying them to the real project.

### Step 1: Installed LangGraph
```
langgraph
```
```powershell
pip install langgraph
```

### Step 2: Toy Graph 1 — linear state flow (`src/toy_graph_1.py`)
A `TypedDict` state (`number`, `history`) flowing through two nodes (`add_five`, `multiply_by_two`) connected by plain edges.

Result: `{'number': 30, 'history': ['Added 5 -> 15', 'Multiplied by 2 -> 30']}` — confirmed state flows correctly through sequential nodes, each one reading and updating it.

### Step 3: Toy Graph 2 — conditional edge + loop (`src/toy_graph_2.py`)
A single node (`guess_number`) that increments a counter, paired with a conditional function (`is_big_enough`) that routes back to the same node (loop) or to `END`, based on state.

Result: looped through 7 attempts (0→3→6→9→12→15→18→21), stopping once `number >= 20`.
```
Attempt 1-7: number climbing by 3 each time
Final result: {'number': 21, 'attempts': 7}
```
Confirmed the core looping mechanism that the real agent's Grade/Retry and Groundedness-check loops will use.

### Step 4: Toy Graph 3 — real retriever wired into LangGraph (`src/toy_graph_3.py`)
Built a graph using the actual project components:
- `analyze_query` node — calls `extract_filters()` from `query_analyzer.py`
- `retrieve` node — runs real ChromaDB filtered/unfiltered search
- `check_results` conditional — retries without filters if nothing came back (only if under 2 attempts)

Result: ran end-to-end against the real vector database.
```
[analyze_query] Attempt 1 — filters: {'month': 2, 'week': 1}
[retrieve] Found 1 chunks

chunk_4 (distance: 0.6877)
MONTH 2
WEEK 1
Deep Learning Fundamentals & Neural Networks
```
Correct chunk returned on the first attempt — the retry path wasn't exercised since the filtered search succeeded immediately (correct behavior, not a bug; noted as optional follow-up to explicitly test the retry path with a query that would return no results, e.g. asking about a nonexistent "Month 5").

---

## Errors / issues encountered

| # | Issue | Cause | Resolution |
|---|---|---|---|
| 1 | `groq.GroqError: api_key client option must be set` in `query_analyzer.py` | Script never called `load_dotenv()` — each entry-point script needs its own call, it's not shared automatically across files | Added `from dotenv import load_dotenv` + `load_dotenv()` at the top of `query_analyzer.py` |
| 2 | `ConnectionRefusedError` / debugpy traceback when running `query_analyzer.py` | Ran via VS Code's Run/Debug button (F5) instead of the terminal — same recurring interpreter mismatch pattern from Day 1 | Ran directly from terminal: `python src/query_analyzer.py` |
| 3 | `vector_store.py` became structurally broken after inserting the filtered-search block in the wrong place | Filtered-search code was pasted before embeddings were generated/stored, and old unfiltered-search code was left in as a duplicate — caused `NameError`s and would have run the search twice | Rewrote the full file in the correct order: load → clean → chunk → embed → prepare data → generate embeddings → store → filtered search → display results |
| 4 | Vector store path had reverted to `data/chroma` instead of `data/chroma_db` in a pasted version | Copy-paste from an older version of the file | Corrected back to `data/chroma_db` to match the Day 5 cosine-metric fix and avoid creating a second, separate, unfiltered database |

---

## Key lesson from Day 6
Two lessons, one from each half of the day:
1. **Filtering and similarity search solve different problems.** Semantic search finds "what's conceptually related"; metadata filtering finds "what matches an exact known field." Real retrieval quality often needs both together, not one or the other.
2. **LangGraph's core pattern is small and consistent**: state in → node transforms it → conditional function decides the next node name → repeat or exit. Once this clicks with a toy example, it directly maps onto real components (a toy "is_big_enough" check becomes a real "is_relevant" or "is_grounded" check) — the structure doesn't change, only the logic inside each node does.

---

## Confirmed state after Day 6
- `src/query_analyzer.py` — LLM-based structured filter extraction (month/week), tested and working
- `src/vector_store.py` — combines filtered + unfiltered vector search correctly, in the right execution order
- `src/toy_graph_1.py`, `src/toy_graph_2.py`, `src/toy_graph_3.py` — three working LangGraph examples, the third one using real project components (embeddings, filters, ChromaDB)
- Confirmed end-to-end: a real LangGraph graph correctly answers "Week 1 of Month 2" using the actual retrieval pipeline

---

## Next up: Day 7 — Building the real Retrieve node as a proper module