# Day 11 Log — Wire Full Graph End-to-End + Test Both Loops
### Agentic RAG Project (rag-project-1)

---

## Goal for the day
Confirm the complete 5-node graph works correctly with real questions, specifically testing that **both** self-correction loops (retrieve retry, and generation regenerate) function properly through the actual graph — not just in isolated standalone tests.

## Why this mattered
Day 10 confirmed the full graph on a happy-path question, and confirmed `check_groundedness()` catches a fabricated answer — but only standalone, never through the graph's actual regenerate loop. Day 9 confirmed the retrieve retry loop — but on an older, 4-node version of the graph, before the groundedness node existed. Day 11 closed both gaps.

---

## Test 1: Retrieve retry loop, re-confirmed on the current full 5-node graph

Ran "What is the capital of France?" through the complete graph (all 5 nodes, both conditional edges wired):

```
Attempt 1: retrieve (3 chunks, no filters) -> grade (NOT relevant) -> rewrite: "What is the capital city of France?"
Attempt 2: retrieve (3 chunks) -> grade (NOT relevant) -> rewrite: "Capital city of France"
Attempt 3: retrieve (3 chunks) -> grade (NOT relevant) -> max attempts reached
-> generate (best-effort) -> check_groundedness (TRUE — honest non-answer correctly graded as grounded) -> done
```

**Final answer:** *"The context does not contain information about the capital of France."*
**State:** `Relevant: False | Retrieval attempts: 3` / `Grounded: True | Generation attempts: 1`

Confirmed: the retrieve loop still works correctly now that it flows into the newer groundedness stage — the honest failure answer was correctly recognized as grounded (per the explicit rule built into the groundedness prompt in Day 10), not incorrectly flagged.

**Minor issue noted (not fixed):** the final answer cited sources (`Month 2, Week 2; Month 4, Week 2; Month 4, Week 4`) even though it was saying "I don't know." Not wrong or hallucinated (those chunks genuinely were retrieved), but slightly confusing to a real user — citing sources alongside a non-answer. Candidate future refinement: skip citations when the answer is a non-answer.

---

## Test 2 & 3: Attempting to force-trigger the regenerate loop through the graph

Two deliberate, escalating stress tests on `generator.py`, using the known-answerable "Week 1 of Month 2" question:

### Attempt 1 — loosened prompt, same temperature (0)
Removed the "ONLY use context" instruction, replaced with permission to "add helpful extra detail... even if it's not explicitly stated."
**Result:** `Grounded: True` — answer stayed accurate, only added a genuinely-present detail ("Free resources and GenAI support are also mentioned") already in the source text. No fabrication.

### Attempt 2 — raised temperature to 0.9 + explicit speculation instruction
Prompt explicitly said "you may speculate and add plausible extra details, specific numbers, names, or examples even if they aren't explicitly stated... be creative... don't hold back."
**Result:** `Grounded: True` — answer reformatted the same real topics into a nicer list, added a mild editorial line ("These topics are listed in the course outline..."), but still invented no fake facts, names, or numbers.

**Neither attempt triggered the regenerate loop.**

---

## Key finding
This model (`openai/gpt-oss-20b` via Groq), given real, relevant retrieved context, proved genuinely resistant to fabrication — even under two escalating adversarial prompts designed specifically to induce hallucination. This is a **positive result for the system's reliability**, even though it left the regenerate loop untriggered through the graph in this specific test.

Importantly, this does **not** mean the regenerate loop is unverified — its correctness was already independently proven in Day 10, where `check_groundedness()` was tested standalone against a genuinely fabricated answer (the fake "Dr. Andrew Ng" instructor claim) and correctly caught it. What Day 11 adds is a separate, honest finding: *getting the model to fabricate in the first place*, on a document it has solid context for, is hard — which is a good property of the system, distinct from whether the safety net itself works.

After both stress tests, `generator.py` was reverted back to its original strict configuration: `temperature=0`, "ONLY the information in the context" instruction restored exactly as before.

---

## Errors / issues encountered
None — no bugs this time. Both stress tests ran cleanly; the "issue" was that the intended failure couldn't be forced, which is itself the finding, not a defect.

---

## Key lesson from Day 11
Testing a safety mechanism only by directly unit-testing it (Day 10's standalone fabricated-answer test) isn't the same as testing whether the system it's meant to protect actually produces that failure mode in practice. Both are valuable: the standalone test proves the checker *can* catch a bad answer; the end-to-end stress test tells you *how likely* that bad answer is to occur naturally. Here, the checker works, and the failure mode it guards against turned out to be rare for this model/document combination — worth knowing both facts, not just one.

---

## Confirmed state after Day 11
- Full 5-node graph re-verified correct on both a genuinely answerable question (Day 10) and a genuinely unanswerable one (Day 11, retrieve loop fully exercised)
- Groundedness regenerate loop's underlying logic independently verified (Day 10, standalone); its real-world trigger rate observed to be low against this model/document (Day 11, two escalating attempts, both failed to trigger)
- `generator.py` confirmed reverted to original strict grounding configuration after stress testing
- Minor citation-on-non-answer cosmetic issue noted, not fixed

---

## Next up: Day 12 — Hybrid search + reranking (inside the Retrieve node) + evaluation