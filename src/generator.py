from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def get_generator():
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0)

def generate_answer(question: str, chunks: list) -> str:
    """
    Generates an answer strictly grounded in the retrieved chunks,
    with source citations (month/week/section).
    """
    if not chunks:
        return "I don't have enough information in the document to answer this question."

    # Build context with source labels the LLM can cite
    context_parts = []
    for chunk_id, doc, distance, metadata in chunks:
        source_label = f"[Month {metadata.get('month')}, Week {metadata.get('week')} — {metadata.get('section')}]"
        context_parts.append(f"{source_label}\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    llm = get_generator()
    prompt = (
        f"Answer the question using ONLY the information in the context below. "
        f"Do not use any outside knowledge. If the context doesn't contain the answer, say so clearly.\n\n"
        f"After your answer, cite which source(s) you used in this format: (Source: Month X, Week Y).\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    
    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    from retriever import retrieve

    question = "What topics are covered in Week 1 of Month 2?"
    chunks, filters = retrieve(question)

    answer = generate_answer(question, chunks)
    print(f"Question: {question}")
    print(f"\nAnswer:\n{answer}")