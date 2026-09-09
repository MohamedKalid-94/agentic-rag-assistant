from typing import TypedDict
from langgraph.graph import StateGraph, END


class GraphState(TypedDict):
    """The shared state every node can read and update."""
    number: int
    history: list[str]


def add_five(state: GraphState) -> GraphState:
    state["number"] += 5
    state["history"].append(f"Added 5 -> {state['number']}")
    return state


def multiply_by_two(state: GraphState) -> GraphState:
    state["number"] *= 2
    state["history"].append(f"Multiplied by 2 -> {state['number']}")
    return state


# Build the graph
builder = StateGraph(GraphState)

builder.add_node("add_five", add_five)
builder.add_node("multiply_by_two", multiply_by_two)

builder.set_entry_point("add_five")
builder.add_edge("add_five", "multiply_by_two")
builder.add_edge("multiply_by_two", END)

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({"number": 10, "history": []})
    print(result)