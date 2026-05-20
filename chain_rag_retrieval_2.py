from dotenv import load_dotenv

load_dotenv()
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from langchain_pinecone import PineconeVectorStore

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vectorstore = PineconeVectorStore(
    embedding=embeddings, index_name=os.getenv("INDEX_NAME")
)

query = "Who is Professor McGonagall?"

docs = vectorstore.as_retriever(search_kwargs={"k": 3})
# retrieved_docs = docs.invoke(query)


def format_docs(docs):
    return "\n\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(docs)])


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

chain = (
    {"context": docs | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
)

query = "Who is Professor McGonagall?"

output = chain.invoke(query)

print("Answer:")
print(output)
