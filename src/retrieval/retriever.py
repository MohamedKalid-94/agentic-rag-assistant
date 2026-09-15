from rank_bm25 import BM25Okapi
import chromadb

from sentence_transformers import CrossEncoder
from embeddings.embedder import get_embedding_model
from retrieval.query_analyzer import extract_filters
from ingestion.loader import load_documents
from processing.chunking import clean_text, split_documents
from config import CHROMA_DB_PATH, COLLECTION_NAME, RERANKER_MODEL_NAME, DOCUMENTS_FOLDER

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def rerank(question: str, candidates: list, top_n: int = 3) -> list:
    """
    Re-scores candidates using a cross-encoder for more precise relevance ranking.
    candidates: list of (id, doc, score, metadata) tuples.
    """
    reranker = get_reranker()
    pairs = [(question, doc) for _, doc, _, _ in candidates]
    rerank_scores = reranker.predict(pairs)

    reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
    return [(c[0], c[1], score, c[3]) for c, score in reranked[:top_n]]

def get_vector_store():
    """Connects to the existing Chroma collection."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def get_bm25_index():
    """Builds a BM25 keyword index over all chunks (rebuilt fresh each call for simplicity)."""
    documents = load_documents(DOCUMENTS_FOLDER)
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    chunks = split_documents(documents)

    tokenized_corpus = [chunk.page_content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunks

def reciprocal_rank_fusion(vector_ranked_ids: list, bm25_ranked_ids: list, k: int = 60) -> dict:
    """
    Combines two ranked lists into one score using Reciprocal Rank Fusion.
    Standard, simple technique for merging rankings from different retrieval methods.
    """
    scores = {}
    for rank, doc_id in enumerate(vector_ranked_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_ranked_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return scores

def retrieve(question: str, n_results: int = 3, use_filters: bool = True) -> tuple:
    """
    Full retrieval pipeline: hybrid search (vector + BM25) -> cross-encoder rerank.
    This is what the graph actually uses.
    """
    candidates, filters = hybrid_retrieve(question, n_results=10, use_filters=use_filters)

    if not candidates:
        return [], filters

    reranked = rerank(question, candidates, top_n=n_results)
    return reranked, filters

def vector_retrieve(question: str, n_results: int = 3, use_filters: bool = True) -> tuple:
    """Original plain vector search — kept for comparison/testing."""
    embedder = get_embedding_model()
    collection = get_vector_store()
    query_embedding = embedder.embed_query(question)
    filters = extract_filters(question) if use_filters else {}

    if filters:
        where_clause = filters if len(filters) == 1 else {"$and": [{k: v} for k, v in filters.items()]}
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results, where=where_clause)
    else:
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    return list(zip(
        results["ids"][0], results["documents"][0], results["distances"][0], results["metadatas"][0]
    )), filters

def hybrid_retrieve(question: str, n_results: int = 3, use_filters: bool = True) -> tuple:
    """
    Combines vector search and BM25 keyword search using Reciprocal Rank Fusion.
    Both branches respect the same metadata filters.
    """
    embedder = get_embedding_model()
    collection = get_vector_store()
    bm25, chunks = get_bm25_index()

    query_embedding = embedder.embed_query(question)
    filters = extract_filters(question) if use_filters else {}

    # --- Vector search ---
    if filters:
        where_clause = filters if len(filters) == 1 else {"$and": [{k: v} for k, v in filters.items()]}
        vector_results = collection.query(query_embeddings=[query_embedding], n_results=10, where=where_clause)
    else:
        vector_results = collection.query(query_embeddings=[query_embedding], n_results=10)

    vector_ranked_ids = vector_results["ids"][0]

    # --- BM25 keyword search, restricted to the same filtered set ---
    def matches_filters(chunk):
        return all(chunk.metadata.get(key) == value for key, value in filters.items())

    valid_indices = [i for i, chunk in enumerate(chunks) if not filters or matches_filters(chunk)]

    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    filtered_scores = [(i, bm25_scores[i]) for i in valid_indices]
    bm25_ranked_indices = [i for i, _ in sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:10]]
    bm25_ranked_ids = [f"chunk_{i}" for i in bm25_ranked_indices]

    # --- Fuse rankings ---
    fused_scores = reciprocal_rank_fusion(vector_ranked_ids, bm25_ranked_ids)
    top_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:n_results]

    chunk_lookup = {f"chunk_{i}": chunk for i, chunk in enumerate(chunks)}
    results = []
    for doc_id in top_ids:
        chunk = chunk_lookup.get(doc_id)
        if chunk:
            results.append((doc_id, chunk.page_content, fused_scores[doc_id], chunk.metadata))

    return results, filters


if __name__ == "__main__":
    # Test 1: the full pipeline (what the graph actually uses)
    question = "What topics are covered in Week 1 of Month 2?"
    chunks, filters = retrieve(question)

    print(f"Question: {question}")
    print(f"Filters used: {filters}\n")
    for chunk_id, doc, score, metadata in chunks:
        print(f"{chunk_id} (rerank score: {score:.4f})")
        print(f"Section: {metadata.get('section')}")
        print(f"Preview: {doc[:200]}\n")

    # Test 2: the deep-learning-projects near-miss from Day 12's eval
    print("\n" + "=" * 70 + "\n")
        # --- Debug: check the pre-rerank candidate pool ---
    print("\n" + "=" * 70)

    print("DEBUG: pre-rerank candidates for 'deep learning projects' question")
    candidates, filters = hybrid_retrieve("What deep learning projects are covered?", n_results=10)
    for doc_id, doc, score, metadata in candidates:
        print(f"{doc_id} (fused score: {score:.4f}) — {metadata.get('section')}")

    question2 = "What deep learning projects are covered?"
    chunks2, filters2 = retrieve(question2)

    print(f"Question: {question2}")
    print(f"Filters used: {filters2}\n")
    for chunk_id, doc, score, metadata in chunks2:
        print(f"{chunk_id} (rerank score: {score:.4f})")
        print(f"Section: {metadata.get('section')}")
        print(f"Preview: {doc[:200]}\n")