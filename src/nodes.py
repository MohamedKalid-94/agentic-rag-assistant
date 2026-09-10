from state import AgentState
from retriever import retrieve
from grader import grade_relevance, rewrite_query, check_groundedness
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
    state["generation_attempts"] += 1
    print(f"[generate_node] Answer generated (attempt {state['generation_attempts']}, {len(answer)} chars)")
    return state


def check_groundedness_node(state: AgentState) -> AgentState:
    """LangGraph node: verifies the generated answer is grounded in the retrieved context."""
    grounded = check_groundedness(state["answer"], state["retrieved_chunks"])
    state["is_grounded"] = grounded.is_grounded
    print(f"[check_groundedness_node] Grounded: {grounded.is_grounded} — {grounded.reasoning}")
    return state


def check_relevance(state: AgentState) -> str:
    """Conditional: decide whether to retry retrieval or move to generation."""
    if state["is_relevant"]:
        return "generate"
    if state["attempts"] >= 3:
        print("[check_relevance] Max attempts reached — generating best-effort answer anyway")
        return "generate"
    return "retry"


def check_groundedness_edge(state: AgentState) -> str:
    """Conditional: if the answer isn't grounded, regenerate — up to a max of 2 tries."""
    if state["is_grounded"]:
        return "done"
    if state["generation_attempts"] >= 2:
        print("[check_groundedness_edge] Max generation attempts reached — returning best-effort answer")
        return "done"
    return "regenerate"