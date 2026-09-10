from retriever import retrieve
from grader import grade_relevance, check_groundedness
from generator import generate_answer


# A small hand-built evaluation set: (question, expected_chunk_id)
EVAL_SET = [
    ("What topics are covered in Week 1 of Month 2?", "chunk_4"),
    ("What is covered in Week 2 Structured Query Language?", "chunk_1"),
    ("What deep learning projects are covered?", "chunk_5"),
    ("What is covered about LLM Engineering and Prompt Engineering?", "chunk_9"),
    ("What topics are in Week 4 of Month 1?", "chunk_3"),
]


def evaluate_retrieval():
    correct = 0
    print("=== RETRIEVAL EVALUATION ===\n")
    for question, expected_id in EVAL_SET:
        chunks, filters = retrieve(question, n_results=1)
        retrieved_id = chunks[0][0] if chunks else None
        is_correct = retrieved_id == expected_id
        correct += is_correct
        print(f"Q: {question}")
        print(f"Expected: {expected_id} | Got: {retrieved_id} | {'✅' if is_correct else '❌'}\n")

    accuracy = correct / len(EVAL_SET)
    print(f"Retrieval accuracy: {correct}/{len(EVAL_SET)} ({accuracy:.0%})")
    return accuracy


def evaluate_generation():
    print("\n=== GENERATION EVALUATION (grounded + relevant) ===\n")
    grounded_count = 0
    relevant_count = 0

    for question, expected_id in EVAL_SET:
        chunks, filters = retrieve(question, n_results=3)
        answer = generate_answer(question, chunks)

        relevance = grade_relevance(question, chunks)
        groundedness = check_groundedness(answer, chunks)

        relevant_count += relevance.is_relevant
        grounded_count += groundedness.is_grounded

        print(f"Q: {question}")
        print(f"Relevant: {relevance.is_relevant} | Grounded: {groundedness.is_grounded}\n")

    print(f"Relevance rate: {relevant_count}/{len(EVAL_SET)}")
    print(f"Groundedness rate: {grounded_count}/{len(EVAL_SET)}")


if __name__ == "__main__":
    evaluate_retrieval()
    evaluate_generation()