from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np


def get_embedding_model():
    """Returns a local embedding model (free, runs on CPU, no API key needed)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


if __name__ == "__main__":

    from ingestion import load_documents
    from chunking import clean_text, split_documents

    # ---------------------------------------------------------
    # 1. LOAD DOCUMENTS
    # ---------------------------------------------------------

    documents = load_documents("data/documents")

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    print(f"Loaded {len(documents)} documents/pages")

    # ---------------------------------------------------------
    # 2. CHUNK DOCUMENTS
    # ---------------------------------------------------------

    chunks = split_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    # ---------------------------------------------------------
    # 3. LOAD EMBEDDING MODEL
    # ---------------------------------------------------------

    embedder = get_embedding_model()

    # ---------------------------------------------------------
    # 4. EMBED ALL CHUNKS
    # ---------------------------------------------------------

    all_texts = [chunk.page_content for chunk in chunks]

    all_vectors = embedder.embed_documents(all_texts)

    print(f"Total chunks embedded: {len(all_vectors)}")
    print(f"Embedding dimension: {len(all_vectors[0])}")

    # ---------------------------------------------------------
    # 5. BUILD SIMILARITY MATRIX
    # ---------------------------------------------------------

    n = len(all_vectors)

    sims = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            sims[i][j] = cosine_similarity(
                all_vectors[i],
                all_vectors[j]
            )

    # ---------------------------------------------------------
    # 6. MOST SIMILAR CHUNK PAIR
    # ---------------------------------------------------------

    sims_for_max = sims.copy()

    # Ignore self-similarity
    np.fill_diagonal(sims_for_max, -1)

    max_idx = np.unravel_index(
        np.argmax(sims_for_max),
        sims_for_max.shape
    )

    print(
        f"\nMost similar pair: "
        f"Chunk {max_idx[0]} & Chunk {max_idx[1]} "
        f"(similarity: {sims[max_idx]:.4f})"
    )

    print(
        f"Chunk {max_idx[0]} preview: "
        f"{chunks[max_idx[0]].page_content[:150]}"
    )

    print(
        f"Chunk {max_idx[1]} preview: "
        f"{chunks[max_idx[1]].page_content[:150]}"
    )

    # ---------------------------------------------------------
    # 7. LEAST SIMILAR CHUNK PAIR
    # ---------------------------------------------------------

    sims_for_min = sims.copy()

    # Ignore self-similarity
    np.fill_diagonal(sims_for_min, 2)

    min_idx = np.unravel_index(
        np.argmin(sims_for_min),
        sims_for_min.shape
    )

    print(
        f"\nLeast similar pair: "
        f"Chunk {min_idx[0]} & Chunk {min_idx[1]} "
        f"(similarity: {sims[min_idx]:.4f})"
    )

    print(
        f"Chunk {min_idx[0]} preview: "
        f"{chunks[min_idx[0]].page_content[:150]}"
    )

    print(
        f"Chunk {min_idx[1]} preview: "
        f"{chunks[min_idx[1]].page_content[:150]}"
    )

    # ---------------------------------------------------------
    # 8. OVERALL SIMILARITY STATISTICS
    # ---------------------------------------------------------

    upper_triangle = sims[
        np.triu_indices(n, k=1)
    ]

    print(
        f"\nAverage similarity across all chunk pairs: "
        f"{upper_triangle.mean():.4f}"
    )

    print(
        f"Similarity range: "
        f"{upper_triangle.min():.4f} "
        f"to {upper_triangle.max():.4f}"
    )

    # ---------------------------------------------------------
    # 9. QUERY EMBEDDING TEST
    # ---------------------------------------------------------

    sample_question = (
        "What topics are covered in Week 1 of Month 2?"
    )

    query_vector = embedder.embed_query(
        sample_question
    )

    print(
        f"\nQuery embedding dimension: "
        f"{len(query_vector)}"
    )

    # ---------------------------------------------------------
    # 10. QUERY → CHUNK SIMILARITY
    # ---------------------------------------------------------

    query_sims = [
        cosine_similarity(query_vector, vector)
        for vector in all_vectors
    ]

    # ---------------------------------------------------------
    # 11. TOP-K RETRIEVAL
    # ---------------------------------------------------------

    top_k = 16

    top_indices = np.argsort(query_sims)[::-1][:top_k]

    print(
        f"\nQuestion: '{sample_question}'"
    )

    print(
        f"\nTop {top_k} matching chunks:"
    )

    for rank, idx in enumerate(
        top_indices,
        start=1
    ):

        print("\n" + "=" * 60)

        print(f"Rank       : {rank}")
        print(f"Chunk ID   : {idx}")
        print(f"Similarity : {query_sims[idx]:.4f}")

        # IMPORTANT:
        # Show metadata so we can verify
        # Month → Week → Section.

        print("\nMetadata:")
        print(chunks[idx].metadata)

        print("\nPreview:")
        print(
            chunks[idx].page_content[:500]
        )