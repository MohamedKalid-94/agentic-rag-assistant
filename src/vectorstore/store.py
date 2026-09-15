import chromadb

from ingestion.loader import load_documents
from processing.chunking import clean_text, split_documents
from embeddings.embedder import get_embedding_model
from config import CHROMA_DB_PATH, COLLECTION_NAME, DOCUMENTS_FOLDER

def rebuild_vector_store(documents_folder: str = DOCUMENTS_FOLDER) -> int:
    """
    Runs the full ingestion pipeline: load -> clean -> chunk -> embed -> store.
    Always clears the old collection first, so stale chunks never linger
    when the document set changes.
    """
    documents = load_documents(documents_folder)

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    chunks = split_documents(documents)

    embedder = get_embedding_model()
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass  # doesn't exist yet on first run — fine

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    embeddings = embedder.embed_documents(texts)

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


if __name__ == "__main__":
    count = rebuild_vector_store()
    print(f"Stored {count} chunks in Chroma")