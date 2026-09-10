from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import retrieve_node, grade_node, rewrite_node, generate_node, check_relevance


builder = StateGraph(AgentState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("generate", generate_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "grade")

builder.add_conditional_edges(
    "grade",
    check_relevance,
    {
        "retry": "rewrite",
        "generate": "generate"
    }
)

builder.add_edge("rewrite", "retrieve")
builder.add_edge("generate", END)

graph = builder.compile()


if __name__ == "__main__":
    question = "What is the capital of France?"

    result = graph.invoke({
        "question": question,
        "original_question": question,
        "filters": {},
        "retrieved_chunks": [],
        "is_relevant": False,
        "attempts": 0,
        "answer": ""
    })

    print("\n=== FINAL ANSWER ===")
    print(result["answer"])