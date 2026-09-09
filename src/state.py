from typing import TypedDict


class AgentState(TypedDict):
    """Shared state for the agentic RAG graph."""
    question: str
    filters: dict
    retrieved_chunks: list          # list of (id, text, distance, metadata)
    attempts: int