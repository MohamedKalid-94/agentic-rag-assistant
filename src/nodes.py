from state import AgentState
from retriever import retrieve
from grader import grade_relevance, rewrite_query
from generator import generate_answer

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

def generate_node(state: AgentState) -> AgentState:
    """LangGraph node: generates the final answer from relevant chunks."""
    answer = generate_answer(state["original_question"], state["retrieved_chunks"])
    state["answer"] = answer
    print(f"[generate_node] Answer generated ({len(answer)} chars)")
    return state

def check_relevance(state: AgentState) -> str:
    if state["is_relevant"]:
        return "generate"
    if state["attempts"] >= 3:
        print("[check_relevance] Max attempts reached — generating best-effort answer anyway")
        return "generate"
    return "retry"