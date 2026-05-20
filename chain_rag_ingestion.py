import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

loader = TextLoader(
    "C:\\Users\\Faizan\\Desktop\\langchain_udemy\\medium.txt", encoding="utf-8"
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

chunks = text_splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# VERY IMPORTANT
ids = [str(uuid4()) for _ in range(len(chunks))]

vectorstore = PineconeVectorStore.from_documents(
    documents=chunks, embedding=embeddings, index_name=os.getenv("INDEX_NAME"), ids=ids
)

print("Upload complete")
