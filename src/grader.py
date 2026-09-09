from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

load_dotenv()


class RelevanceGrade(BaseModel):
    """Judgment on whether retrieved chunks answer the question."""
    is_relevant: bool = Field(description="True if the chunks contain information that answers the question")
    reasoning: str = Field(description="Brief explanation of the judgment")


def get_grader():
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    return llm.with_structured_output(RelevanceGrade)


def grade_relevance(question: str, chunks: list) -> RelevanceGrade:
    """Uses the LLM to judge whether retrieved chunks actually answer the question."""
    if not chunks:
        return RelevanceGrade(is_relevant=False, reasoning="No chunks were retrieved.")

    context = "\n\n---\n\n".join(chunk[1] for chunk in chunks)  # chunk[1] = document text

    grader = get_grader()
    result = grader.invoke(
        f"Question: {question}\n\n"
        f"Retrieved content:\n{context}\n\n"
        f"Does the retrieved content contain enough information to answer the question? "
        f"Judge strictly — only mark relevant if the specific answer is actually present."
    )
    return result


def rewrite_query(original_question: str) -> str:
    """Uses the LLM to rewrite a question that failed to retrieve relevant chunks."""
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)
    response = llm.invoke(
        f"This question failed to retrieve relevant results from a document search: '{original_question}'\n"
        f"Rewrite it as a clearer, more specific search query that might retrieve better results. "
        f"Return ONLY the rewritten question, nothing else."
    )
    return response.content.strip()


if __name__ == "__main__":
    # Quick standalone test
    from retriever import retrieve

    question = "What topics are covered in Week 1 of Month 2?"
    chunks, filters = retrieve(question)

    grade = grade_relevance(question, chunks)
    print(f"Question: {question}")
    print(f"Is relevant: {grade.is_relevant}")
    print(f"Reasoning: {grade.reasoning}")

    # Test with a question that should NOT match well
    bad_question = "What is the capital of France?"
    bad_chunks, _ = retrieve(bad_question, use_filters=False)
    bad_grade = grade_relevance(bad_question, bad_chunks)
    print(f"\nQuestion: {bad_question}")
    print(f"Is relevant: {bad_grade.is_relevant}")
    print(f"Reasoning: {bad_grade.reasoning}")