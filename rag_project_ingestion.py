from dotenv import load_dotenv

load_dotenv()
import asyncio
import logging
import os
import ssl
from typing import Any, Dict, List

import certifi
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import (TavilyCrawl, TavilyExtract, TavilyMap,
                              TavilySearch)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure SSL context to use certifi's CA bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", progress_bar=True, chunk_size=50
)

vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings,
)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_document_async(documents: List[Document], batch_size: int = 10):
    """
    Processes a list of documents in batches and indexes them into a vector store asynchronously.
    """
    log_header_indexing = (
        f"{Colors.BOLD}{Colors.CYAN}"
        "==================== RAG Project Indexing ===================="
        f"{Colors.RESET}"
    )
    log_success_indexing = (
        f"{Colors.GREEN}" "Documents indexed successfully." f"{Colors.RESET}"
    )
    print(log_header_indexing)

    # Create batches of documents
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    logging.info(f"Indexing {len(documents)} documents in {len(batches)} batches...")

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            logging.info(f"Batch {batch_num} indexed successfully.")
        except Exception as e:
            logging.error(f"Error occurred while indexing batch {batch_num}: {e}")
            return False
        return True

    # process batches concurrently
    tasks = [add_batch(batch, idx + 1) for idx, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check results
    successful_batches = sum(1 for result in results if result is True)

    if successful_batches == len(batches):
        print(log_success_indexing)
    else:
        logging.warning(
            f"{len(batches) - successful_batches} out of {len(batches)} batches failed to index."
        )


class Colors:
    RESET = "\033[0m"
    PURPLE = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"


# Header
log_header_main = (
    f"{Colors.BOLD}{Colors.CYAN}"
    "==================== RAG Project Ingestion ===================="
    f"{Colors.RESET}"
)

# Info Log
log_info = (
    f"{Colors.PURPLE}"
    "TavilyCrawl: Start crawling documentation from "
    "https://docs.langchain.com/"
    f"{Colors.RESET}"
)

# Success Log
log_success = f"{Colors.GREEN}" "Documentation crawled successfully." f"{Colors.RESET}"

# Warning Log
log_warning = (
    f"{Colors.YELLOW}" "Some pages were skipped during crawling." f"{Colors.RESET}"
)

# Error Log
log_error = f"{Colors.RED}" "Failed to crawl documentation." f"{Colors.RESET}"

# PRINT LOGS
print(log_header_main)
print(log_info)


# Crawl the documentions site
res = tavily_crawl.invoke(
    {"url": "https://docs.langchain.com/", "max_depth": 1, "extract_depth": "advanced"}
)

results = res["results"]
print(f"Crawled {len(results)} pages.")

all_docs = [
    Document(page_content=doc["raw_content"], metadata={"source": doc["url"]})
    for doc in results
]
print(log_success)


log_header_chunking = (
    f"{Colors.BOLD}{Colors.CYAN}"
    "==================== RAG Project Chunking ===================="
    f"{Colors.RESET}"
)
log_success_chunking = (
    f"{Colors.GREEN}" "Documents chunked successfully." f"{Colors.RESET}"
)

print(log_header_chunking)

# Chunk the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
splitted_docs = text_splitter.split_documents(all_docs)
print(f"Chunked into {len(splitted_docs)} documents.")
print(log_success_chunking)

# Index the documents
asyncio.run(index_document_async(splitted_docs, batch_size=10))

print(
    f"{Colors.BOLD}{Colors.CYAN}"
    "==================== RAG Project Ingestion Completed ===================="
    f"{Colors.RESET}"
)
