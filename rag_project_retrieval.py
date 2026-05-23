from dotenv import load_dotenv

load_dotenv()

import os
from typing import Any, Dict

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from langchain_pinecone import PineconeVectorStore

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings,
)

model = init_chat_model("google_genai:gemini-2.5-flash", temperature=0.3)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str) -> str:
    """
    Retrieves relevant context from the vector store based on the input query.
    Arguments:
    query: The input query for which context needs to be retrieved.
    Returns:
    A string representing the retrieved context.
    """
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)
    context = "\n".join([result.page_content for result in retrieved_docs])
    return context, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Runs the language model with the given query and retrieves relevant context.
    Arguments:
    query: The input query for which the language model needs to be run.
    Returns:
    A dictionary containing the response from the language model and the retrieved context.
    """
    system_prompt = (
        "You are an assistant that helps answer questions based on retrieved context. "
        "You have access to a tool that can retrieve relevant context from a vector store. "
        "Use the tool to get the necessary context and then answer the question as accurately as possible. "
        "Always cite the sources in your response. "
        "If the retrieved context is not sufficient to answer the question, indicate that in your response."
    )
    agent = create_agent(
        model=model, tools=[retrieve_context], system_prompt=system_prompt
    )
    messages = [{"role": "user", "content": query}]
    response = agent.invoke({"messages": messages})
    last_content = response["messages"][-1].content

    if isinstance(last_content, list):
        answer = last_content[0]["text"]
    else:
        answer = last_content

    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {"answer": answer, "context_docs": context_docs}


if __name__ == "__main__":
    result = run_llm("Who is Professor McGonagall and what is her role at Hogwarts?")
    print("Answer:")
    print(result["answer"])
    print("\nRetrieved Context:")
    for idx, doc in enumerate(result["context_docs"], 1):
        print(f"Context Document {idx}:")
        print(doc.page_content)
        print("-" * 50)
