from typing import TypedDict
import chromadb

from embeddings import get_embedding_model
from query_analyzer import extract_filters
from langgraph.graph import StateGraph, END


class RetrievalState(TypedDict):
    question: str
    filters: dict
    results: list
    attempts: int


embedder = get_embedding_model()
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_or_create_collection(
    name="roadmap_collection",
    metadata={"hnsw:space": "cosine"}
)


def analyze_query(state: RetrievalState) -> RetrievalState:
    """Node: extract structured filters from the question using the LLM."""
    state["filters"] = extract_filters(state["question"])
    state["attempts"] += 1
    print(f"\n[analyze_query] Attempt {state['attempts']} — filters: {state['filters']}")
    return state


def retrieve(state: RetrievalState) -> RetrievalState:
    """Node: run vector search, filtered if we have filters."""
    query_embedding = embedder.embed_query(state["question"])

    if state["filters"]:
        where_clause = (
            state["filters"] if len(state["filters"]) == 1
            else {"$and": [{k: v} for k, v in state["filters"].items()]}
        )
        results = collection.query(query_embeddings=[query_embedding], n_results=3, where=where_clause)
    else:
        results = collection.query(query_embeddings=[query_embedding], n_results=3)

    state["results"] = list(zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ))
    print(f"[retrieve] Found {len(state['results'])} chunks")
    return state


def check_results(state: RetrievalState) -> str:
    """Conditional: if nothing came back, retry without filters. Otherwise, done."""
    if not state["results"] and state["attempts"] < 2:
        print("[check_results] No results — retrying without filters")
        state["filters"] = {}  # drop filters and try again
        return "retry"
    return "done"


builder = StateGraph(RetrievalState)
builder.add_node("analyze_query", analyze_query)
builder.add_node("retrieve", retrieve)

builder.set_entry_point("analyze_query")
builder.add_edge("analyze_query", "retrieve")
builder.add_conditional_edges(
    "retrieve",
    check_results,
    {"retry": "retrieve", "done": END}
)

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({
        "question": "What topics are covered in Week 1 of Month 2?",
        "filters": {},
        "results": [],
        "attempts": 0
    })

    print("\n=== FINAL RESULTS ===")
    for chunk_id, doc, distance in result["results"]:
        print(f"\n{chunk_id} (distance: {distance:.4f})")
        print(doc[:200])