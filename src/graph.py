from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import retrieve_node, grade_node, rewrite_node, check_relevance


builder = StateGraph(AgentState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_node)
builder.add_node("rewrite", rewrite_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "grade")

builder.add_conditional_edges(
    "grade",
    check_relevance,
    {
        "retry": "rewrite",
        "done": END
    }
)

builder.add_edge("rewrite", "retrieve")  # after rewriting, go back and retrieve again

graph = builder.compile()


if __name__ == "__main__":
    bad_question = "What is the capital of France?"

    result = graph.invoke({
        "question": bad_question,
        "original_question": bad_question,
        "filters": {},
        "retrieved_chunks": [],
        "is_relevant": False,
        "attempts": 0
    })

    print("\n=== FINAL STATE ===")
    print(f"Relevant: {result['is_relevant']}")
    print(f"Attempts: {result['attempts']}")