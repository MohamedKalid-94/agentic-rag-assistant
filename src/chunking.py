import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion import load_documents


def clean_text(text: str) -> str:
    """Removes the repeated PDF footer block and stray bullet placeholders."""
    text = re.sub(
        r"120 Days · Agentic AI Engineering RoadMap 2026\s*"
        r"The Planner Sheet · Minimum 2 Hours / Day · Month-wise Topics · Week-wise Deliverables · Free Resources\s*"
        r"AI Coach John \| PROITBRIDGE Page \d+ of \d+",
        "",
        text
    )
    text = re.sub(r"^\s*•\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def split_documents(documents):
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


def validate_chunks(chunks):
    empty_chunks = [i for i, chunk in enumerate(chunks) if not chunk.page_content.strip()]
    if empty_chunks:
        print(f"Warning: {len(empty_chunks)} empty chunks found")
    else:
        print("No empty chunks found")

    lengths = [len(chunk.page_content) for chunk in chunks]
    print(f"Minimum chunk length: {min(lengths)}")
    print(f"Maximum chunk length: {max(lengths)}")
    print(f"Average chunk length: {sum(lengths) / len(lengths):.2f}")


if __name__ == "__main__":
    documents = load_documents("data/documents")
    print(f"Loaded {len(documents)} pages")

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    validate_chunks(chunks)

    print("\n--- Boundary Inspection ---")
    for i in range(min(10, len(chunks) - 1)):
        current = chunks[i].page_content.strip()
        next_chunk = chunks[i + 1].page_content.strip()
        print(f"\n===== Chunk {i + 1} → Chunk {i + 2} =====")
        print("\nEND OF CURRENT CHUNK:")
        print(current[-300:])
        print("\nSTART OF NEXT CHUNK:")
        print(next_chunk[:300])
        print("-" * 80)

        tiny = min(chunks, key=lambda c: len(c.page_content))
        print(repr(tiny.page_content))