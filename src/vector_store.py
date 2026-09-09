import chromadb

from ingestion import load_documents
from chunking import clean_text, split_documents
from embeddings import get_embedding_model


# --------------------------------------------------
# 1. LOAD DOCUMENTS
# --------------------------------------------------

documents = load_documents(
    "data/documents"
)

print(
    f"Loaded {len(documents)} documents/pages"
)


# --------------------------------------------------
# 2. CLEAN TEXT
# --------------------------------------------------

for doc in documents:

    doc.page_content = clean_text(
        doc.page_content
    )


# --------------------------------------------------
# 3. CREATE CHUNKS
# --------------------------------------------------

chunks = split_documents(
    documents
)

print(
    f"Total chunks: {len(chunks)}"
)


# --------------------------------------------------
# 4. LOAD EMBEDDING MODEL
# --------------------------------------------------

embedder = get_embedding_model()


# --------------------------------------------------
# 5. CREATE CHROMA CLIENT
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="data/chroma"
)


# --------------------------------------------------
# 6. CREATE / GET COLLECTION
# --------------------------------------------------

collection = client.get_or_create_collection(
    name="roadmap_collection",
    metadata={"hnsw:space": "cosine"}
)

print("\nChroma collection metadata:")
print(collection.metadata)

# --------------------------------------------------
# 7. PREPARE DATA
# --------------------------------------------------

ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]

texts = [
    chunk.page_content
    for chunk in chunks
]

metadatas = [
    chunk.metadata
    for chunk in chunks
]


# --------------------------------------------------
# 8. GENERATE EMBEDDINGS
# --------------------------------------------------

embeddings = embedder.embed_documents(
    texts
)

print(
    f"Generated {len(embeddings)} embeddings"
)

print(
    f"Embedding dimension: "
    f"{len(embeddings[0])}"
)


# --------------------------------------------------
# 9. STORE EVERYTHING IN CHROMA
# --------------------------------------------------

collection.upsert(
    ids=ids,
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas
)

print(
    f"Stored {collection.count()} chunks in Chroma"
)


# --------------------------------------------------
# 10. RAW SIMILARITY SEARCH
# --------------------------------------------------

query = (
    "What topics are covered in "
    "Week 1 of Month 2?"
)

query_embedding = embedder.embed_query(
    query
)


results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)


# --------------------------------------------------
# 11. DISPLAY RESULTS
# --------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    f"QUERY: {query}"
)

print(
    "=" * 70
)


for rank, (
    result_id,
    document,
    metadata,
    distance
) in enumerate(
    zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ),
    start=1
):

    print("\n" + "-" * 70)
    print(f"Rank     : {rank}")
    print(f"ID       : {result_id}")
    print(f"Distance : {distance:.4f}")
    print(f"Metadata : {metadata}")
    print("\nPreview:")
    print(document[:500])