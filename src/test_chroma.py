import chromadb

# Connect to existing Chroma database
client = chromadb.PersistentClient(path="data/chroma_db")

# Get existing collection
collection = client.get_collection(name="roadmap_collection")

# Check stored chunks
print(f"Chunks stored: {collection.count()}")