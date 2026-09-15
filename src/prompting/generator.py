from dotenv import load_dotenv
from langchain_groq import ChatGroq

from config import GROQ_MODEL

load_dotenv()


def get_generator():
    return ChatGroq(model=GROQ_MODEL, temperature=0)

def generate_answer(question: str, chunks: list) -> str:
    """
    Generates an answer strictly grounded in the retrieved chunks,
    with source citations. Falls back to filename-based citation
    for chunks that don't have month/week metadata (e.g. non-roadmap PDFs).
    """
    if not chunks:
        return "I don't have enough information in the document to answer this question."

    # Build context with source labels the LLM can cite
    context_parts = []
    for chunk_id, doc, distance, metadata in chunks:
        month = metadata.get('month')
        week = metadata.get('week')

        if month is not None and week is not None:
            source_label = f"[Month {month}, Week {week} — {metadata.get('section')}]"
        else:
            filename = metadata.get('source', 'Unknown source').split('/')[-1].split('\\')[-1]
            source_label = f"[{filename} — {metadata.get('section', 'General')}]"

        context_parts.append(f"{source_label}\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    llm = get_generator()
    prompt = (
        f"Answer the question using ONLY the information in the context below. "
        f"Do not use any outside knowledge. If the context doesn't contain the answer, say so clearly "
        f"and do NOT include a source citation in that case.\n\n"
        f"If you DO find the answer in the context, cite which source(s) you used in this format: "
        f"(Source: Month X, Week Y) for roadmap content, or (Source: filename) for other documents.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    from retrieval.retriever import retrieve

    question = "What topics are covered in Week 1 of Month 2?"
    chunks, filters = retrieve(question)

    answer = generate_answer(question, chunks)
    print(f"Question: {question}")
    print(f"\nAnswer:\n{answer}")