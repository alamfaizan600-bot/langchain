import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = PineconeVectorStore(
    embedding=embeddings, index_name=os.getenv("INDEX_NAME")
)
query = "Who is Professor McGonagall?"
docs = vectorstore.similarity_search_with_score(query, k=3)
print(docs[0])

# prompt_template = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant that provides concise answers based on the retrieved documents.",
#         ),
#         ("human", "{query}\n\nHere are the relevant documents:\n{documents}"),
#     ]
# )

# chain = prompt_template | llm
# formatted_docs = "\n\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(docs)])
# output = chain.invoke({"query": query, "documents": formatted_docs})
# print("Answer:")
# print(output.content)
