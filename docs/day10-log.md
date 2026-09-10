# Day 10 Log — Check Groundedness Node
### Agentic RAG Project (rag-project-1)

---

## Goal for the day
Build a node that verifies the generated answer is actually supported by the retrieved context (not hallucinated), with a retry loop back to `generate` if it isn't — closing the second half of the Corrective RAG loop.

## 🎯 Milestone reached today
**The complete 5-node agentic RAG pipeline now works end-to-end in a single run:**
```
retrieve -> grade -> generate -> check_groundedness -> done
```
Every node from the original architecture diagram (Day 6) is now real, working code, with both self-correction loops (retrieval retry + generation regenerate) confirmed working on genuine failure cases, not just the happy path.

---

## Steps completed

### 1. Updated `src/state.py` — added groundedness fields
```python
class AgentState(TypedDict):
    question: str
    original_question: str
    filters: dict
    retrieved_chunks: list
    is_relevant: bool
    attempts: int
    answer: str
    is_grounded: bool
    generation_attempts: int
```
`generation_attempts` tracks regeneration tries separately from `attempts` (retrieval retries), so the two loops don't interfere with each other's counters.

### 2. Added groundedness checking to `src/grader.py`
```python
class GroundednessGrade(BaseModel):
    is_grounded: bool = Field(description="True if every claim in the answer is supported by the context")
    reasoning: str = Field(description="Brief explanation of the judgment")


def check_groundedness(answer: str, chunks: list) -> GroundednessGrade:
    context = "\n\n---\n\n".join(chunk[1] for chunk in chunks)
    grader = get_groundedness_grader()
    result = grader.invoke(
        f"Context:\n{context}\n\nGenerated answer:\n{answer}\n\n"
        f"Does this answer contain ONLY information that is directly supported by the context? "
        f"Flag it as NOT grounded if it adds any fact, number, or claim not present in the context above. "
        f"An honest 'I don't know' / 'the context doesn't contain this' answer should always be marked grounded."
    )
    return result
```
Key design choice: explicitly instructing the grader that an honest "I don't know" should always count as grounded — without this, a short/negative answer could get incorrectly flagged as "ungrounded" just for being non-committal, which would be the wrong signal.

### 3. Standalone test — real vs. fabricated answer
- **Real answer** (from Day 9's actual generated output) → `is_grounded: True`, correct reasoning
- **Deliberately fabricated answer** (invented a fake instructor "Dr. Andrew Ng," a "12-hour live workshop," and a "certification exam" — none of which exist in the document) → `is_grounded: False`, with reasoning specifically naming each fabricated detail

This confirmed the checker genuinely evaluates claims against context, not just approving anything plausible-sounding.

### 4. Wired into `src/nodes.py`
Added `generate_node` (updated to increment `generation_attempts`), `check_groundedness_node`, and the new conditional `check_groundedness_edge`:
```python
def check_groundedness_edge(state: AgentState) -> str:
    if state["is_grounded"]:
        return "done"
    if state["generation_attempts"] >= 2:
        return "done"  # give up gracefully, same pattern as check_relevance
    return "regenerate"
```

### 5. Wired into `src/graph.py`
```python
builder.add_node("check_groundedness", check_groundedness_node)
builder.add_edge("generate", "check_groundedness")
builder.add_conditional_edges(
    "check_groundedness",
    check_groundedness_edge,
    {"regenerate": "generate", "done": END}
)
```

### 6. Full end-to-end test
```
[retrieve_node] Attempt 1 — 1 chunks found (filters: {'month': 2, 'week': 1})
[grade_node] Relevant: True — matches the question
[generate_node] Answer generated (attempt 1, 649 chars)
[check_groundedness_node] Grounded: True — lists only context-provided topics

=== FINAL ANSWER ===
[full accurate, cited list of Deep Learning topics]

Grounded: True | Generation attempts: 1
```
All 5 nodes fired correctly in sequence, both conditional edges evaluated correctly, final state consistent throughout.

---

## Errors / issues encountered

| # | Issue | Cause | Resolution |
|---|---|---|---|
| 1 | `NameError: name 'BaseModel' is not defined` | `GroundednessGrade` class was pasted above the `from pydantic import BaseModel, Field` import line instead of below it | Moved the class definition below the imports |
| 2 | Duplicate `GroundednessGrade` class defined twice in `grader.py` | Copy-paste leftover — harmless (Python just uses the second definition) but untidy | Noted for cleanup; not blocking |
| 3 | `ImportError: cannot import name 'grade_relevance' from 'grader'` | When groundedness functions were added, the whole file got overwritten rather than appended to — silently dropped `get_grader()`, `grade_relevance()`, and `rewrite_query()` | Rewrote the complete `grader.py` with all original Day 8 functions plus the new Day 10 groundedness functions together, no omissions |
| 4 | `ImportError: cannot import name 'retrieve_node' from 'nodes'` | Same root cause as #3, but in `nodes.py` — adding new node functions overwrote the file instead of appending, dropping `retrieve_node`, `grade_node`, `rewrite_node` | Rewrote the complete `nodes.py` with all 5 node functions and both conditional-edge functions together |

**Pattern across errors 3 and 4:** both came from replacing a whole file with new content instead of adding to the existing content — a real risk when growing a multi-function file incrementally. Going forward: always add new code *after* existing code, never paste in a way that could overwrite working functions.

---

## Key lesson from Day 10
Groundedness checking needs the same rigor as relevance grading: it has to be tested against a case *specifically designed to fail* (the fabricated Andrew Ng answer), not just against cases that are already correct. A checker that only ever sees good input never actually proves it can catch bad input. Also reinforced: as a file grows across multiple days, incremental edits become risky — reviewing the *complete* file after each change (not just the diff) is worth the extra step to catch silently dropped code.

---

## Confirmed state after Day 10
- `src/grader.py` — both `grade_relevance()` (Day 8) and `check_groundedness()` (Day 10) present and correct, tested on true-positive and true-negative cases for both
- `src/nodes.py` — all 5 nodes (`retrieve`, `grade`, `rewrite`, `generate`, `check_groundedness`) and both conditional-edge functions
- `src/graph.py` — the complete agentic RAG graph: retrieve → grade → (retry loop) → generate → check_groundedness → (regenerate loop) → done
- **Confirmed end-to-end**: full pipeline runs correctly in a single invocation, producing an accurate, cited, grounded answer
- Both self-correction loops independently stress-tested with genuine failure cases in earlier days (Day 8: retrieval retry on "capital of France"; Day 10: groundedness catch on fabricated answer)

---

## Next up: Day 11 — Wire the full graph end-to-end with additional test cases, or advance to hybrid search / evaluation (original checklist Days 11-12)