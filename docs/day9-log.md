# Day 9 Log — Generate Answer Node
### Agentic RAG Project (rag-project-1)

---

## Goal for the day
Build a node that generates a real answer from retrieved chunks, strictly grounded in that context, with source citations — and wire it into the graph so it runs after a successful grade.

---

## Steps completed

### 1. Updated `src/state.py` — added the answer field
```python
class AgentState(TypedDict):
    question: str
    original_question: str
    filters: dict
    retrieved_chunks: list
    is_relevant: bool
    attempts: int
    answer: str
```

### 2. Wrote `src/generator.py`
```python
def generate_answer(question: str, chunks: list) -> str:
    """Generates an answer strictly grounded in the retrieved chunks, with source citations."""
    if not chunks:
        return "I don't have enough information in the document to answer this question."

    context_parts = []
    for chunk_id, doc, distance, metadata in chunks:
        source_label = f"[Month {metadata.get('month')}, Week {metadata.get('week')} — {metadata.get('section')}]"
        context_parts.append(f"{source_label}\n{doc}")
    context = "\n\n---\n\n".join(context_parts)

    llm = get_generator()
    prompt = (
        f"Answer the question using ONLY the information in the context below. "
        f"Do not use any outside knowledge. If the context doesn't contain the answer, say so clearly.\n\n"
        f"After your answer, cite which source(s) you used in this format: (Source: Month X, Week Y).\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    response = llm.invoke(prompt)
    return response.content
```
Key design choices: explicit "ONLY the information in context" instruction, explicit permission to say "I don't know," and a required citation format tied to the metadata already stored on each chunk.

### 3. Standalone test
"What topics are covered in Week 1 of Month 2?" → produced a complete, accurate list of the real topics from Chunk 4 (Deep Learning, Neural Networks, Perceptron, CNNs, Transfer Learning, etc.), correctly cited as `(Source: Month 2, Week 1)`.

### 4. Wired into `src/nodes.py` — added `generate_node`
```python
def generate_node(state: AgentState) -> AgentState:
    answer = generate_answer(state["original_question"], state["retrieved_chunks"])
    state["answer"] = answer
    return state
```

Also **updated `check_relevance`** to route to `"generate"` instead of `"done"` — both on success AND after exhausting retries. This was a deliberate design decision: even when the agent gives up trying to find better chunks, it should still attempt an answer (or honestly say it can't) rather than silently terminating with nothing.

### 5. Wired into `src/graph.py`
```python
builder.add_node("generate", generate_node)
builder.add_conditional_edges("grade", check_relevance, {"retry": "rewrite", "generate": "generate"})
builder.add_edge("generate", END)
```

---

## Testing — two scenarios through the full graph

### Scenario 1: Answerable question (success path)
"What topics are covered in Week 1 of Month 2?" → retrieve (1 chunk, correct filters) → grade (relevant) → generate.
**Result:** full, accurate, cited answer — matched the standalone generator test exactly, now flowing through the real graph.

### Scenario 2: Unanswerable question (honest failure path)
"What is the capital of France?" → 3 failed retrieve/grade/rewrite cycles (same pattern as Day 8) → max attempts reached → routed to generate anyway.
**Result:** *"The context does not contain the answer to that question."* — no hallucination, no crash, honest response even under a forced best-effort generation.

---

## Errors / issues encountered

| # | Issue | Cause | Resolution |
|---|---|---|---|
| 1 | `KeyError: 'done'` when running the updated graph | `graph.py`'s conditional edge mapping was updated to `{"retry": ..., "generate": ...}`, but `nodes.py`'s `check_relevance` function still had the old Day 8 version returning `"done"` on success — the two files got out of sync during editing | Replaced the stale `check_relevance` in `nodes.py` with the updated version that returns `"generate"` instead of `"done"` |

---

## Key finding: rewrite can repeat itself
On the unanswerable-question test, attempts 2 and 3 produced the **exact same rewritten question** ("What is the capital city of France?" both times) — the LLM ran out of meaningfully different phrasings for a question that's fundamentally outside the document's scope, and repeated itself rather than trying something new. Not a bug, but a real limitation: `rewrite_query` doesn't currently track previous rewrites, so it can spin on a duplicate rather than genuinely diversifying its attempts. Noted as a candidate improvement (e.g., pass prior rewrites into the prompt and ask for something meaningfully different) — not fixed today.

---

## Key lesson from Day 9
Grounding isn't just a nice-to-have instruction — it needs to be tested against a case specifically designed to break it. The "capital of France" test isn't just a retrieval-quality check anymore; run through the full generate step, it's now a direct test of whether the LLM will fabricate an answer when it has no real information. Confirming the honest "I don't know" response, even after being forced into best-effort generation, is the actual proof that the grounding constraint works under pressure — not just on paper.

---

## Confirmed state after Day 9
- `src/generator.py` — `generate_answer()`, tested standalone and through the full graph
- `src/nodes.py` — `generate_node` added; `check_relevance` updated to always route to generation (success or exhausted retries)
- `src/graph.py` — full pipeline now runs retrieve → grade → (retry loop) → generate → real final answer
- Confirmed on both a success case (accurate, cited answer) and a genuine-failure case (honest "I don't know," no hallucination)
- Noted, not yet fixed: rewrite can produce duplicate rewrites across attempts

---

## Next up: Day 10 — Check groundedness node (verifies the generated answer isn't hallucinated) + retry loop back to generate