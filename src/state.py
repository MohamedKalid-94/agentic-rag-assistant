from typing import TypedDict


class AgentState(TypedDict):
    """Shared state for the agentic RAG graph."""
    question: str
    original_question: str
    filters: dict
    retrieved_chunks: list
    is_relevant: bool
    attempts: int
    answer: str
    is_grounded: bool
    generation_attempts: int