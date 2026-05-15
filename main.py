import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature=0.1, max_output_tokens=512
)

output = llm.invoke("What is the capital of France?")
print(output.content)
