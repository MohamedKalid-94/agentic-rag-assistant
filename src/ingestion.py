from langchain_community.document_loaders import PyPDFLoader
import os

def load_documents(folder_path: str):
    """Load all PDF files from a folder."""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
    return documents