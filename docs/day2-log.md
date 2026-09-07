# Day 2 Log — Document Loading & Text Extraction
### Agentic RAG Project (rag-project-1)

---

## Goal for the day
Load PDF documents into the project and inspect the raw extracted text and metadata, before moving into chunking (Day 3).

---

## Steps completed

### 1. Added sample PDFs to `data/documents/`
Started with a small/simple PDF first (a bus ticket) to test the loader quickly, then added a larger, more realistic PDF once the pipeline was confirmed working.

### 2. Installed PDF loading packages
Added to `requirements.txt`:
```
pypdf
langchain-community
```
```powershell
pip install pypdf
pip install langchain-community
```

### 3. Wrote the loader — `src/ingestion.py`
```python
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
```

### 4. First test run — bus ticket PDF (4 pages)
```powershell
python src/ingestion.py
```
Result: loaded successfully, 4 pages, metadata correct (source filename, page number).

**Observation:** the extracted text came out flattened and jumbled — labels and values merged together with no clear reading order (e.g. "FlixBus", city names, dates all run together). This happens because the PDF uses a visual/table-style layout, and text extractors read content in internal document order, not visual reading order. Useful early example of why raw PDF extraction is often messy.

### 5. Second test run — richer PDF (24 pages)
Added a real multi-page document ("120 Days · Agentic AI Engineering RoadMap Planner 2026") to `data/documents/` and re-ran the same script.

Result: loaded successfully, 24 pages, metadata correct. Extracted text was clean, readable paragraph text — a much better representative sample for the actual RAG project than the bus ticket.

---

## Testing approach
- Ran the loader script directly (`python src/ingestion.py`) rather than through the VS Code Run button, consistent with the fix from Day 1
- Verified correctness by checking three things each run: (1) total page count matches the PDF, (2) first-page text preview looks like real extracted content, (3) metadata includes the correct source filename and page number
- Deliberately tested with two very different PDF types (a short visually-formatted ticket vs. a long paragraph-based document) to see how extraction quality differs

---

## Errors / issues encountered

| # | Issue | Cause | Resolution |
|---|---|---|---|
| 1 | `DeprecationWarning: langchain-community is being sunset` | `langchain_community.document_loaders` is being split into standalone packages over time | Not a breaking error — ignored for now, functionality unaffected; only worth revisiting if it later breaks |
| 2 | Messy/jumbled text extraction from the bus ticket PDF | PDF used a visual/table layout rather than plain paragraph text; extractor reads in internal document order, not visual order | Not a bug — expected behavior. Noted as the reason chunking and text cleanup matter in later days. Solved practically by testing with a second, more paragraph-heavy PDF instead |
| 3 | Had to re-apply execution policy (`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`) before activating venv again | Process-scoped policy fix doesn't persist across all terminal/session types the same way the Day 1 CurrentUser-scoped fix does | Re-ran the process-scoped command as a quick precaution; no further issues |

---

## Key lesson from Day 2
Not all PDFs extract cleanly — layout-heavy documents (tickets, invoices, forms) produce fragmented text, while paragraph-based documents (reports, manuals, articles) extract much more reliably. Good to test with both types early, and to pick a paragraph-heavy document as the main working example for retrieval quality in later days.

---

## Confirmed state after Day 2
- Loader (`src/ingestion.py`) works correctly on multiple PDF types
- Metadata (source, page number) confirmed present and correct — needed later for citing sources
- Working test documents: bus ticket (4 pages, messy) + 120-Day Agentic AI Roadmap Planner (24 pages, clean) — the roadmap PDF will be the main document going forward

---

## Next up: Day 3 — Chunking strategies
