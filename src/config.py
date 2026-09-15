"""Shared configuration constants for the Agentic RAG project."""
import os

# Project root = one level up from this file's location (src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LLM
GROQ_MODEL = "openai/gpt-oss-20b"

# Embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Reranking
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Vector store
CHROMA_DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
COLLECTION_NAME = "roadmap_collection"

# Chunking (fallback splitter, for non-structured documents)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Documents
DOCUMENTS_FOLDER = os.path.join(BASE_DIR, "data", "documents")

# Agent loop limits
MAX_RETRIEVAL_ATTEMPTS = 3
MAX_GENERATION_ATTEMPTS = 2