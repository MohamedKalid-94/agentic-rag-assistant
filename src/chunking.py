import re

from langchain_core.documents import Document

from ingestion import load_documents


def clean_text(text: str):
    """Remove repeated PDF footer blocks and stray bullet placeholders."""

    # Remove repeated PDF footer
    text = re.sub(
        r"120 Days · Agentic AI Engineering RoadMap 2026\s*"
        r"The Planner Sheet · Minimum 2 Hours / Day · Month-wise Topics · Week-wise Deliverables · Free Resources\s*"
        r"AI Coach John\s*\|\s*PROITBRIDGE Page \d+ of \d+",
        "",
        text,
    )

    # Remove standalone bullet placeholders
    text = re.sub(
        r"^\s*•\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Collapse excessive blank lines
    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text,
    )

    return text.strip()


def split_documents(documents):
    """
    Split documents by WEEK headings while carrying MONTH context
    across PDF pages.
    """

    chunks = []

    # Example:
    # WEEK 1 Deep Learning Fundamentals & Neural Networks
    #
    # Also matches:
    # WEEK 4 (Projects) ML Project Development & Deployment
    week_pattern = re.compile(
        r"(?m)^WEEK\s+(\d+)(?:\s+\(([^)]*)\))?\s*(.*)$"
    )

    # Example:
    # MONTH 2 · DAYS 31–60
    month_pattern = re.compile(
        r"(?m)^MONTH\s+(\d+)"
    )

    # Keep track of the current Month while moving
    # through PDF pages.
    current_month = None

    for doc in documents:

        text = doc.page_content

        # -----------------------------------------------------
        # 1. Check whether this page contains a MONTH heading
        # -----------------------------------------------------

        month_matches = list(
            month_pattern.finditer(text)
        )

        if month_matches:
            current_month = int(
                month_matches[-1].group(1)
            )

        # -----------------------------------------------------
        # 2. Find WEEK headings on this page
        # -----------------------------------------------------

        matches = list(
            week_pattern.finditer(text)
        )

        if not matches:
            continue

        # -----------------------------------------------------
        # 3. Create one chunk for each WEEK
        # -----------------------------------------------------

        for i, match in enumerate(matches):

            # -------------------------------------------------
            # Extract Week information FIRST
            # -------------------------------------------------

            week_number = int(
                match.group(1)
            )

            week_type = match.group(2)

            week_title = match.group(3).strip()

            # -------------------------------------------------
            # Determine start/end of this Week
            # -------------------------------------------------

            start = match.start()

            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)

            week_text = text[start:end].strip()

            # -------------------------------------------------
            # Build metadata
            # -------------------------------------------------

            metadata = dict(doc.metadata)

            metadata.update(
                {
                    "month": current_month,
                    "week": week_number,
                    "week_type": week_type,
                    "section": week_title,
                }
            )

            # -------------------------------------------------
            # Add explicit hierarchy to embedding text
            # -------------------------------------------------

            if current_month is not None:

                embedding_text = (
                    f"MONTH {current_month}\n"
                    f"WEEK {week_number}\n"
                    f"{week_title}\n\n"
                    f"{week_text}"
                )

            else:

                embedding_text = week_text

            # -------------------------------------------------
            # Create ONE LangChain Document
            # -------------------------------------------------

            chunks.append(
                Document(
                    page_content=embedding_text,
                    metadata=metadata,
                )
            )

    return chunks


def validate_chunks(chunks):
    """Validate generated chunks."""

    if not chunks:
        print("Warning: No chunks were created.")
        return

    empty_chunks = [
        i
        for i, chunk in enumerate(chunks)
        if not chunk.page_content.strip()
    ]

    if empty_chunks:
        print(
            f"Warning: {len(empty_chunks)} empty chunks found"
        )
    else:
        print("No empty chunks found")

    lengths = [
        len(chunk.page_content)
        for chunk in chunks
    ]

    print(
        f"Minimum chunk length: {min(lengths)}"
    )

    print(
        f"Maximum chunk length: {max(lengths)}"
    )

    print(
        f"Average chunk length: "
        f"{sum(lengths) / len(lengths):.2f}"
    )


if __name__ == "__main__":

    # ---------------------------------------------------------
    # 1. Load PDF pages
    # ---------------------------------------------------------

    documents = load_documents(
        "data/documents"
    )

    print(
        f"Loaded {len(documents)} pages"
    )

    # ---------------------------------------------------------
    # 2. Clean text
    # ---------------------------------------------------------

    for doc in documents:
        doc.page_content = clean_text(
            doc.page_content
        )

    # ---------------------------------------------------------
    # 3. Structure-aware chunking
    # ---------------------------------------------------------

    chunks = split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks"
    )

    # ---------------------------------------------------------
    # 4. Validate
    # ---------------------------------------------------------

    validate_chunks(chunks)

    # ---------------------------------------------------------
    # 5. Inspect chunks
    # ---------------------------------------------------------

    print(
        "\n--- Chunk Inspection ---"
    )

    for i, chunk in enumerate(chunks):

        print(
            "\n" + "=" * 80
        )

        print(
            f"CHUNK {i}"
        )

        print(
            "=" * 80
        )

        print(
            f"Length: "
            f"{len(chunk.page_content)} characters"
        )

        print(
            "\nMetadata:"
        )

        print(
            chunk.metadata
        )

        print(
            "\nSTART:"
        )

        print(
            chunk.page_content[:300]
        )

        print(
            "\nEND:"
        )

        print(
            chunk.page_content[-300:]
        )