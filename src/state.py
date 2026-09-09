from typing import TypedDict


class AgentState(TypedDict):
    """Shared state for the agentic RAG graph."""
    question: str
    original_question: str          # keep the user's original question, in case we rewrite
    filters: dict
    retrieved_chunks: list
    is_relevant: bool
    attempts: int