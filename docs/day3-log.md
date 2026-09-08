# Day 3 Log — Chunking Strategies
### Agentic RAG Project (rag-project-1)

---

## Goal for the day
Split loaded documents into smaller chunks suitable for embedding, validate chunk quality, and inspect chunk boundaries to confirm overlap is working correctly.

---

## Steps completed

### 1. Split project code by responsibility
Kept `ingestion.py` focused only on loading PDFs, and put chunking, cleaning, and validation logic in a separate `chunking.py`, importing the loader from `ingestion.py`.

**Why separate files (not just documentation convention — actual software design principle):**
- **Single responsibility** — one file, one job (load vs. chunk), easier to test and debug independently
- **Reusability** — `load_documents()` will be called again later from `vector_store.py` and `app.py`; keeping it in its own module avoids duplicating logic
- **Matches the original planned folder structure** (`src/ingestion.py`, `src/chunking.py`, etc. as separate modules from Day 0 planning)
- **Easier debugging** — when something breaks, it's immediately clear whether it's a loading problem or a chunking problem

### 2. `src/ingestion.py` — loader only
```python
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
```

### 3. `src/chunking.py` — cleaning, splitting, validation
```python
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
```

---

## Testing approach — first pass (before cleaning)

Ran chunking directly on raw loaded text (no cleaning step yet):
- 24 pages → 36 chunks
- No empty chunks
- Min length 216, max 999, average 710.28
- Boundary inspection confirmed overlap was working (e.g. shared "01 Python Tutorial For Beginners..." text appearing at both the end of one chunk and start of the next)

**Issue spotted:** the PDF's repeated footer block (title, tagline, "AI Coach John | PROITBRIDGE Page X of 24") was showing up inside nearly every chunk boundary, along with stray bullet-only lines from a table/graphic that didn't extract cleanly. This is noise that would get embedded into every chunk's vector, diluting semantic signal and wasting space in the chunk size budget on repeated boilerplate instead of real content.

## Testing approach — second pass (after adding `clean_text`)

Added a cleaning step (regex-based footer removal + stray bullet removal + blank-line collapsing) applied to each document's `page_content` before chunking. Re-ran:
- 24 pages → 33 chunks (down from 36 — less filler text needed fewer chunks)
- No empty chunks
- Min length 23, max 996, average 610.06 (down from 710.28 — confirms footer bloat was removed)
- Boundary inspection re-checked: footer text no longer interrupting real content at any of the 10 inspected boundaries; overlap still working correctly

**Investigated the 23-character minimum chunk:** printed it directly and confirmed it was `"SEGMENT 02\nWHAT'S NEXT?"` — a harmless leftover section-header fragment, not corrupted content. Confirmed safe to leave as-is; small chunks like this are a normal side effect of splitting at document/section boundaries.

---

## Errors / issues encountered

| # | Issue | Cause | Resolution |
|---|---|---|---|
| 1 | Repeated PDF footer text embedded inside chunk content | PDF footer appears on every page and gets extracted as part of the page text | Added a `clean_text()` regex step to strip the footer block and stray bullet lines before chunking |
| 2 | Debug print of the smallest chunk appeared 10 times in output | The one-off inspection line was placed inside the boundary-inspection `for` loop instead of after it | Cosmetic only, not functionally broken — noted for cleanup, not urgent |
| 3 | Needed to re-apply PowerShell execution policy before activating venv again in a new session | Process-scoped policy fix from a previous session doesn't persist to a new terminal session the same way | Re-ran `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` as a quick precaution |

---

## Key lesson from Day 3
Chunking isn't just "split text every N characters" — the quality of what goes *into* the splitter matters just as much as the split logic itself. Repeated boilerplate (headers/footers) and extraction artifacts (stray bullets) will get embedded into every chunk's vector if not cleaned first, silently degrading retrieval quality later. Always inspect chunk boundaries manually before moving on — numbers alone (chunk count, avg length) don't reveal this kind of noise.

Also reinforced: separating code by responsibility (loading vs. chunking) isn't just style — it enables reuse (the same loader will be called again in later days) and makes debugging much faster.

---

## Confirmed state after Day 3
- `src/ingestion.py` — loads PDFs cleanly (unchanged responsibility)
- `src/chunking.py` — cleans footer/bullet noise, splits into chunks (`chunk_size=1000`, `chunk_overlap=200`), validates chunk quality
- Final chunk set: 33 chunks from 24 pages, no empty chunks, average length 610 characters, overlap confirmed working at chunk boundaries

---

## Next up: Day 4 — Embeddings
