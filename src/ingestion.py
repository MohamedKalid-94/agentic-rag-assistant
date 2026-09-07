from langchain_community.document_loaders import PyPDFLoader
import os

def load_documents(folder_path: str):
    """Loads all PDFs in a folder and returns a list of LangChain Document objects."""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
    return documents

if __name__ == "__main__":
    docs = load_documents("data/documents")
    print(f"Loaded {len(docs)} pages total")
    print("--- First page preview ---")
    print(docs[0].page_content[:500])
    print("--- Metadata ---")
    print(docs[0].metadata)