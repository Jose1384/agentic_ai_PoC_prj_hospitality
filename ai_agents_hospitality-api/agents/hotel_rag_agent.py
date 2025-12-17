"""
Hotel RAG Agent

This module implements an agentic assistant that answers hotel-related questions
using a Retrieval-Augmented Generation (RAG) approach backed by ChromaDB.

Differences vs simple agent:
- Hotel files are embedded and stored in a vector database (Chroma)
- Context is retrieved dynamically per query instead of injected wholesale
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import Chroma
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    # Fallback to community package if google_genai not available
    from langchain_community.chat_models import ChatGoogleGenerativeAI

# Embeddings + LLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from util.configuration import PROJECT_ROOT
from util.logger_config import logger
from config.agent_config import _get_env_value, _load_config_file, get_agent_config

# Paths
HOTELS_DATA_PATH_LOCAL = PROJECT_ROOT / "data" / "hotels"
HOTELS_DATA_PATH_EXTERNAL = PROJECT_ROOT.parent / "bookings-db" / "output_files" / "hotels"
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma" / "hotels"

# Cached objects
_vectorstore: Optional[Chroma] = None
_rag_chain: Optional[RetrievalQA] = None


async def handle_hotel_query_rag(user_query: str) -> str:
    """Async wrapper for RAG agent (WebSocket compatible)."""
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, answer_hotel_question_rag, user_query
    )
    return response


def answer_hotel_question_rag(user_query: str) -> str:
    """Answer a hotel question using RAG + ChromaDB."""
    try:
        chain = _create_agent_chain()
        logger.info(f"Processing RAG question: {user_query[:100]}...")

        result = chain.invoke({"query": user_query})
        return result["result"]

    except Exception as e:
        logger.error("Error processing RAG question", exc_info=True)
        return f"❌ **Error**: {str(e)}"



def _create_agent_chain() -> RetrievalQA:
    """Create the RetrievalQA chain."""
    global _rag_chain

    if _rag_chain is not None:
        return _rag_chain

    config = get_agent_config()
    
    llm = ChatGoogleGenerativeAI(
            model=config.model,
            temperature=config.temperature,
            google_api_key=config.api_key
        )

    vectorstore = _get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 


    prompt_template = ChatPromptTemplate.from_messages([("system",
        """You are a helpful hotel assistant. Use ONLY the following retrieved hotel context to answer the question.

        When answering questions:
        - Be accurate and specific
        - Reference hotel names, locations, and details from the data
        - If information is not available, say so clearly
        - Format responses in a clear, readable way using markdown
        - Use bullet points and tables when appropriate
        - Include specific prices, addresses, and details when available

        Hotel Data:
        {hotel_context}"""),
        ("human","{question}")
    ])


    _rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template,
                           "document_variable_name": "hotel_context"},
        return_source_documents=True,
    )
# 

    return _rag_chain


def _get_vectorstore() -> Chroma:
    """Create or load the Chroma vector store."""
    global _vectorstore

    # Load existing vector store
    embeddings  = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key= _get_env_value("AI_AGENTIC_API_KEY"))
    _vectorstore = Chroma(
        persist_directory="./vector_store/chroma_db",
        embedding_function=embeddings
    )

    return _vectorstore
