from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import (
    retrieve_node, grade_node, rewrite_node,
    generate_node, check_groundedness_node,
    check_relevance, check_groundedness_edge
)


builder = StateGraph(AgentState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("generate", generate_node)
builder.add_node("check_groundedness", check_groundedness_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "grade")

builder.add_conditional_edges(
    "grade",
    check_relevance,
    {"retry": "rewrite", "generate": "generate"}
)

builder.add_edge("rewrite", "retrieve")
builder.add_edge("generate", "check_groundedness")

builder.add_conditional_edges(
    "check_groundedness",
    check_groundedness_edge,
    {"regenerate": "generate", "done": END}
)

graph = builder.compile()


if __name__ == "__main__":
    question = "What topics are covered in Week 1 of Month 2?"

    result = graph.invoke({
        "question": question,
        "original_question": question,
        "filters": {},
        "retrieved_chunks": [],
        "is_relevant": False,
        "attempts": 0,
        "answer": "",
        "is_grounded": False,
        "generation_attempts": 0
    })

    print("\n=== FINAL ANSWER ===")
    print(result["answer"])
    print(f"\nGrounded: {result['is_grounded']} | Generation attempts: {result['generation_attempts']}")