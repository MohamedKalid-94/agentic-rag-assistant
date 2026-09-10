from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

load_dotenv()


class RelevanceGrade(BaseModel):
    """Judgment on whether retrieved chunks answer the question."""
    is_relevant: bool = Field(description="True if the chunks contain information that answers the question")
    reasoning: str = Field(description="Brief explanation of the judgment")


class GroundednessGrade(BaseModel):
    """Judgment on whether an answer is actually supported by the given context."""
    is_grounded: bool = Field(description="True if every claim in the answer is supported by the context")
    reasoning: str = Field(description="Brief explanation of the judgment")


def get_grader():
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    return llm.with_structured_output(RelevanceGrade)


def grade_relevance(question: str, chunks: list) -> RelevanceGrade:
    if not chunks:
        return RelevanceGrade(is_relevant=False, reasoning="No chunks were retrieved.")

    context = "\n\n---\n\n".join(chunk[1] for chunk in chunks)
    grader = get_grader()

    try:
        result = grader.invoke(
            f"Question: {question}\n\n"
            f"Retrieved content:\n{context}\n\n"
            f"Does the retrieved content contain enough information to answer the question? "
            f"Judge strictly — only mark relevant if the specific answer is actually present."
        )
        return result
    except Exception as e:
        print(f"[grade_relevance] Structured output parsing failed: {e}")
        return RelevanceGrade(is_relevant=False, reasoning="Grading failed due to a parsing error — treated as not relevant.")



def rewrite_query(original_question: str) -> str:
    """Uses the LLM to rewrite a question that failed to retrieve relevant chunks."""
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)
    response = llm.invoke(
        f"This question failed to retrieve relevant results from a document search: '{original_question}'\n"
        f"Rewrite it as a clearer, more specific search query that might retrieve better results. "
        f"Return ONLY the rewritten question, nothing else."
    )
    return response.content.strip()


def get_groundedness_grader():
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    return llm.with_structured_output(GroundednessGrade)


def check_groundedness(answer: str, chunks: list) -> GroundednessGrade:
    context = "\n\n---\n\n".join(chunk[1] for chunk in chunks)
    grader = get_groundedness_grader()

    try:
        result = grader.invoke(
            f"Context:\n{context}\n\n"
            f"Generated answer:\n{answer}\n\n"
            f"Does this answer contain ONLY information that is directly supported by the context? "
            f"Flag it as NOT grounded if it adds any fact, number, or claim not present in the context above. "
            f"An honest 'I don't know' / 'the context doesn't contain this' answer should always be marked grounded."
        )
        return result
    except Exception as e:
        print(f"[check_groundedness] Structured output parsing failed: {e}")
        return GroundednessGrade(is_grounded=False, reasoning="Groundedness check failed due to a parsing error — treated as not grounded, triggering a retry.")



if __name__ == "__main__":
    from retriever import retrieve
    from generator import generate_answer

    question = "What topics are covered in Week 1 of Month 2?"
    chunks, filters = retrieve(question)
    answer = generate_answer(question, chunks)

    grounded = check_groundedness(answer, chunks)
    print(f"Answer:\n{answer}")
    print(f"\nIs grounded: {grounded.is_grounded}")
    print(f"Reasoning: {grounded.reasoning}")

    fake_answer = (
        "The topics covered in Week 1 of Month 2 include Deep Learning fundamentals, "
        "and this week was taught by Dr. Andrew Ng in a 12-hour live workshop with a certification exam."
    )
    fake_grounded = check_groundedness(fake_answer, chunks)
    print(f"\n\nFake answer:\n{fake_answer}")
    print(f"\nIs grounded: {fake_grounded.is_grounded}")
    print(f"Reasoning: {fake_grounded.reasoning}")