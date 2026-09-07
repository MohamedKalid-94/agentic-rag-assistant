from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # reads your .env file

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

response = llm.invoke("In one sentence, what is Retrieval-Augmented Generation?")
print(response.content)