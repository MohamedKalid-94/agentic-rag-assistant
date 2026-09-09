from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import retrieve_node


builder = StateGraph(AgentState)
builder.add_node("retrieve", retrieve_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", END)  # grading/looping comes in later days

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({
        "question": "What topics are covered in Week 1 of Month 2?",
        "filters": {},
        "retrieved_chunks": [],
        "attempts": 0
    })

    print("\n=== FINAL STATE ===")
    for chunk_id, doc, distance, metadata in result["retrieved_chunks"]:
        print(f"\n{chunk_id} (distance: {distance:.4f})")
        print(f"Section: {metadata.get('section')}")