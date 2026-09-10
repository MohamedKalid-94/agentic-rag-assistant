from langchain_community.document_loaders import PyPDFLoader
import hashlib
import os


def load_documents(folder_path: str):
    """
    Load all PDF files from a folder.
    Skips files whose content is identical to one already loaded,
    even if the filename is different (content-hash-based deduplication).
    """
    documents = []
    seen_hashes = {}  # hash -> filename that was kept

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(folder_path, filename)

        with open(file_path, "rb") as f:
            file_bytes = f.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if file_hash in seen_hashes:
            print(f"[load_documents] Skipping '{filename}' — identical content to '{seen_hashes[file_hash]}' (already loaded)")
            continue

        seen_hashes[file_hash] = filename

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        documents.extend(docs)

    return documents