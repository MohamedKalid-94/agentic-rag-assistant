from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np


def get_embedding_model():
    """Returns a local embedding model (free, runs on CPU, no API key needed)."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    from ingestion import load_documents
    from chunking import clean_text, split_documents

    documents = load_documents("data/documents")
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    chunks = split_documents(documents)

    embedder = get_embedding_model()

    # Embed ALL chunks
    all_texts = [chunk.page_content for chunk in chunks]
    all_vectors = embedder.embed_documents(all_texts)

    print(f"Total chunks embedded: {len(all_vectors)}")
    print(f"Embedding dimension: {len(all_vectors[0])}")

    # Build a full similarity matrix
    n = len(all_vectors)
    sims = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sims[i][j] = cosine_similarity(all_vectors[i], all_vectors[j])

    # Find the most similar pair (excluding self-comparison)
    sims_for_max = sims.copy()
    np.fill_diagonal(sims_for_max, -1)
    max_idx = np.unravel_index(np.argmax(sims_for_max), sims_for_max.shape)

    # Find the least similar pair (excluding self-comparison)
    sims_for_min = sims.copy()
    np.fill_diagonal(sims_for_min, 2)  # 2 is above any real cosine similarity (max is 1)
    min_idx = np.unravel_index(np.argmin(sims_for_min), sims_for_min.shape)

    print(f"\nMost similar pair: Chunk {max_idx[0]} & Chunk {max_idx[1]} (similarity: {sims[max_idx]:.4f})")
    print(f"Chunk {max_idx[0]} preview: {chunks[max_idx[0]].page_content[:150]}")
    print(f"Chunk {max_idx[1]} preview: {chunks[max_idx[1]].page_content[:150]}")

    print(f"\nLeast similar pair: Chunk {min_idx[0]} & Chunk {min_idx[1]} (similarity: {sims[min_idx]:.4f})")
    print(f"Chunk {min_idx[0]} preview: {chunks[min_idx[0]].page_content[:150]}")
    print(f"Chunk {min_idx[1]} preview: {chunks[min_idx[1]].page_content[:150]}")

    # Overall stats
    upper_triangle = sims[np.triu_indices(n, k=1)]
    print(f"\nAverage similarity across all chunk pairs: {upper_triangle.mean():.4f}")
    print(f"Similarity range: {upper_triangle.min():.4f} to {upper_triangle.max():.4f}")

    # --- New: query embedding test ---
    sample_question = "What topics are covered in Week 1?"
    query_vector = embedder.embed_query(sample_question)

    print(f"\nQuery embedding dimension: {len(query_vector)}")

    query_sims = [cosine_similarity(query_vector, vec) for vec in all_vectors]
    best_match_idx = np.argmax(query_sims)

    print(f"\nQuestion: '{sample_question}'")
    print(f"Best matching chunk: Chunk {best_match_idx} (similarity: {query_sims[best_match_idx]:.4f})")
    print(f"Chunk preview: {chunks[best_match_idx].page_content[:300]}")