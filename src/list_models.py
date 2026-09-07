from dotenv import load_dotenv
import os
import requests

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
response = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)
print(response.status_code)
for model in response.json().get("data", []):
    print(model["id"])