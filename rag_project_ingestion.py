from dotenv import load_dotenv

load_dotenv()
import os
import ssl
from typing import Any, Dict, List

import certifi
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyExtract, TavilyMap, TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
