from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
from langchain_groq import ChatGroq

load_dotenv()

class QueryFilter(BaseModel):
    """Structured filters extracted from a user's natural language question."""
    month: Optional[int] = Field(
        default=None,
        description="The month number (1-4) if the question mentions a specific month, otherwise null"
    )
    week: Optional[int] = Field(
        default=None,
        description="The week number (1-4) if the question mentions a specific week, otherwise null"
    )


def get_filter_extractor():
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    return llm.with_structured_output(QueryFilter)


def extract_filters(question: str) -> dict:
    """Uses the LLM to pull structured month/week filters out of a natural language question."""
    extractor = get_filter_extractor()
    result = extractor.invoke(
        f"Extract the month and week numbers mentioned in this question, if any: {question}"
    )

    # Build a Chroma-compatible where clause, only including fields that were actually found
    where_clause = {}
    if result.month is not None:
        where_clause["month"] = result.month
    if result.week is not None:
        where_clause["week"] = result.week

    return where_clause


if __name__ == "__main__":
    test_questions = [
        "What topics are covered in Week 1 of Month 2?",
        "Tell me about the deep learning projects",
        "What's in Week 3?",
        "What happens in Month 4?",
    ]

    for q in test_questions:
        filters = extract_filters(q)
        print(f"\nQuestion: '{q}'")
        print(f"Extracted filters: {filters}")