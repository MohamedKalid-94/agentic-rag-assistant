import chromadb
from embeddings import get_embedding_model
from query_analyzer import extract_filters


def get_vector_store():
    """Connects to the existing Chroma collection."""
    client = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_or_create_collection(
        name="roadmap_collection",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def retrieve(question: str, n_results: int = 3, use_filters: bool = True) -> tuple:
    """
    Runs retrieval for a question: extracts filters (optional),
    embeds the question, and queries Chroma.
    Returns (list of (id, document, distance, metadata) tuples, filters used).
    """
    embedder = get_embedding_model()
    collection = get_vector_store()

    query_embedding = embedder.embed_query(question)

    filters = extract_filters(question) if use_filters else {}

    if filters:
        where_clause = (
            filters if len(filters) == 1
            else {"$and": [{k: v} for k, v in filters.items()]}
        )
        results = collection.query(
            query_embeddings=[query_embedding], n_results=n_results, where=where_clause
        )
    else:
        results = collection.query(
            query_embeddings=[query_embedding], n_results=n_results
        )

    return list(zip(
        results["ids"][0],
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    )), filters


if __name__ == "__main__":
    question = "What topics are covered in Week 1 of Month 2?"
    chunks, filters = retrieve(question)

    print(f"Question: {question}")
    print(f"Filters used: {filters}")
    print(f"\nRetrieved {len(chunks)} chunks:\n")

    for chunk_id, doc, distance, metadata in chunks:
        print(f"{chunk_id} (distance: {distance:.4f})")
        print(f"Metadata: {metadata}")
        print(f"Preview: {doc[:200]}\n")