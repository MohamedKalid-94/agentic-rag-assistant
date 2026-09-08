import chromadb


# Connect to existing Chroma database
client = chromadb.PersistentClient(
    path="data/chroma"
)


# Get existing collection
collection = client.get_collection(
    name="agentic_rag"
)


# Check stored chunks
print(
    f"Chunks stored: {collection.count()}"
)