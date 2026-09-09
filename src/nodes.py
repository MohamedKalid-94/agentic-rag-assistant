from state import AgentState
from retriever import retrieve
from grader import grade_relevance, rewrite_query


def retrieve_node(state: AgentState) -> AgentState:
    """LangGraph node: retrieves chunks for the current question."""
    chunks, filters = retrieve(state["question"])
    state["retrieved_chunks"] = chunks
    state["filters"] = filters
    state["attempts"] += 1
    print(f"[retrieve_node] Attempt {state['attempts']} — {len(chunks)} chunks found (filters: {filters})")
    return state


def grade_node(state: AgentState) -> AgentState:
    """LangGraph node: judges whether retrieved chunks actually answer the question."""
    grade = grade_relevance(state["original_question"], state["retrieved_chunks"])
    state["is_relevant"] = grade.is_relevant
    print(f"[grade_node] Relevant: {grade.is_relevant} — {grade.reasoning}")
    return state


def rewrite_node(state: AgentState) -> AgentState:
    """LangGraph node: rewrites the question if the previous attempt failed."""
    new_question = rewrite_query(state["original_question"])
    state["question"] = new_question
    print(f"[rewrite_node] Rewritten question: {new_question}")
    return state


def check_relevance(state: AgentState) -> str:
    """Conditional: decide whether to retry or finish, based on the grade and attempt count."""
    if state["is_relevant"]:
        return "done"
    if state["attempts"] >= 3:
        print("[check_relevance] Max attempts reached — giving up")
        return "done"
    return "retry"