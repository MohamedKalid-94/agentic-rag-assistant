from state import AgentState
from retriever import retrieve


def retrieve_node(state: AgentState) -> AgentState:
    """LangGraph node: retrieves chunks for the current question."""
    chunks, filters = retrieve(state["question"])
    state["retrieved_chunks"] = chunks
    state["filters"] = filters
    state["attempts"] += 1
    print(f"[retrieve_node] Attempt {state['attempts']} — {len(chunks)} chunks found (filters: {filters})")
    return state