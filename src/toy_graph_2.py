from typing import TypedDict
from langgraph.graph import StateGraph, END


class GraphState(TypedDict):
    number: int
    attempts: int


def guess_number(state: GraphState) -> GraphState:
    state["number"] += 3
    state["attempts"] += 1
    print(f"Attempt {state['attempts']}: number is now {state['number']}")
    return state


def is_big_enough(state: GraphState) -> str:
    """A conditional function — returns the NAME of the next node to go to."""
    if state["number"] >= 20:
        return "done"
    else:
        return "retry"


builder = StateGraph(GraphState)
builder.add_node("guess_number", guess_number)

builder.set_entry_point("guess_number")

# Conditional edge: after guess_number, call is_big_enough to decide what's next
builder.add_conditional_edges(
    "guess_number",
    is_big_enough,
    {
        "retry": "guess_number",  # loop back to itself
        "done": END
    }
)

graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({"number": 0, "attempts": 0})
    print(f"\nFinal result: {result}")