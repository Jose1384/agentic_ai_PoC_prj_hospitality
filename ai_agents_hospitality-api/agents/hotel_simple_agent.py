"""
Exercise 0: Simple Agentic Assistant with File Context

This module implements a simple AI agent that answers questions about hotels
by passing hotel files directly to the LLM context. This is the basic scaffolding
for an agentic application without the complexity of RAG.

Uses a small sample of 3 hotels for learning purposes.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional, Tuple

try:
    # Try new LangChain structure (v0.2+)
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    # Fallback to old structure (v0.1)
    from langchain_classic.prompts import ChatPromptTemplate

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    # Fallback to community package if google_genai not available
    from langchain_community.chat_models import ChatGoogleGenerativeAI

# Import ChatOpenAI for proxy/custom endpoint support
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from util.configuration import PROJECT_ROOT
from util.logger_config import logger
from config.agent_config import _load_config_file, get_agent_config

# Path to hotel data files (relative to project root)
# First try local data directory (for Docker), then fallback to bookings-db
HOTELS_DATA_PATH_LOCAL = PROJECT_ROOT / "data" / "hotels"
HOTELS_DATA_PATH_EXTERNAL = PROJECT_ROOT.parent / "bookings-db" / "output_files" / "hotels"


def _get_hotels_data_path():
    """
    Determine the correct path to hotel data files.
    Tries local path first (for Docker), then external path (for local development).
    
    Returns:
        Path: Path to the hotels data directory
    """
    config = _load_config_file()
    local = config.get("hotels", {}).get("local", True)
    if local and HOTELS_DATA_PATH_LOCAL.exists() and (HOTELS_DATA_PATH_LOCAL / "hotels.json").exists():
        logger.info(f"Using local hotel data path: {HOTELS_DATA_PATH_LOCAL}")
        return HOTELS_DATA_PATH_LOCAL
    elif HOTELS_DATA_PATH_EXTERNAL.exists() and (HOTELS_DATA_PATH_EXTERNAL / "hotels.json").exists():
        logger.info(f"Using external hotel data path: {HOTELS_DATA_PATH_EXTERNAL}")
        return HOTELS_DATA_PATH_EXTERNAL
    else:
        logger.error(f"No specific hotel data path configured or files not found, defaulting to local path: {HOTELS_DATA_PATH_LOCAL}")
        raise FileNotFoundError(
            f"Hotel data files not found in either path.\n"
            f"Tried paths:\n"
            f"  - Local: {HOTELS_DATA_PATH_LOCAL / 'hotels.json'}\n"
            f"  - External: {HOTELS_DATA_PATH_EXTERNAL / 'hotels.json'}\n"
            f"Please generate hotel data first:\n"
            f"cd bookings-db && python src/gen_synthetic_hotels.py --num_hotels 3\n"
            f"Or copy files to: {HOTELS_DATA_PATH_LOCAL}"
        )
# Global variables to cache loaded data and agent
_hotels_data: Optional[dict] = None
_hotel_details_text: Optional[str] = None
_agent_chain = None


def load_hotel_data() -> Tuple[dict, str]:
    """
    Load hotel data from JSON and markdown files.
    
    Returns:
        tuple: (hotels_data dict, hotel_details_text str)
        
    Raises:
        FileNotFoundError: If hotel data files don't exist
        json.JSONDecodeError: If hotels.json is invalid
    """
    global _hotels_data, _hotel_details_text
    
    # Return cached data if already loaded
    if _hotels_data is not None and _hotel_details_text is not None:
        return _hotels_data, _hotel_details_text
    
    # Determine the correct path to hotel data
    hotels_data_path = _get_hotels_data_path()
    hotels_json_file = hotels_data_path / "hotels.json"
    hotel_details_file = hotels_data_path / "hotel_details.md"
    
    # Check if files exist
    if not hotels_json_file.exists():
        raise FileNotFoundError(
            f"Hotel data file not found: {hotels_json_file}\n"
            f"Tried paths:\n"
            f"  - Local: {HOTELS_DATA_PATH_LOCAL / 'hotels.json'}\n"
            f"  - External: {HOTELS_DATA_PATH_EXTERNAL / 'hotels.json'}\n"
            f"Please generate hotel data first:\n"
            f"cd bookings-db && python src/gen_synthetic_hotels.py --num_hotels 3\n"
            f"Or copy files to: {HOTELS_DATA_PATH_LOCAL}"
        )
    
    if not hotel_details_file.exists():
        raise FileNotFoundError(
            f"Hotel details file not found: {hotel_details_file}\n"
            f"Tried paths:\n"
            f"  - Local: {HOTELS_DATA_PATH_LOCAL / 'hotel_details.md'}\n"
            f"  - External: {HOTELS_DATA_PATH_EXTERNAL / 'hotel_details.md'}\n"
            f"Please generate hotel data first:\n"
            f"cd bookings-db && python src/gen_synthetic_hotels.py --num_hotels 3\n"
            f"Or copy files to: {HOTELS_DATA_PATH_LOCAL}"
        )
    
    # Load JSON data
    logger.info(f"Loading hotel data from {hotels_json_file}")
    with open(hotels_json_file, 'r', encoding='utf-8') as f:
        _hotels_data = json.load(f)
    
    # Load markdown details
    logger.info(f"Loading hotel details from {hotel_details_file}")
    with open(hotel_details_file, 'r', encoding='utf-8') as f:
        _hotel_details_text = f.read()
    
    logger.info(f"Successfully loaded hotel data ({len(_hotels_data.get('hotels', []))} hotels)")
    
    return _hotels_data, _hotel_details_text


def _create_agent_chain():
    """
    Create and return the LangChain agent chain.
    
    Returns:
        LangChain chain: Prompt template + LLM chain
    """
    global _agent_chain
    
    if _agent_chain is not None:
        return _agent_chain
    
    # Load configuration from centralized config system
    config = get_agent_config()
    
    # Create LLM instance based on provider and configuration
    if config.provider == "openai":
        # Standard OpenAI API
        if not ChatOpenAI:
            raise ImportError("langchain_openai is required for OpenAI provider. Install with: pip install langchain-openai")
        llm = ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            api_key=config.api_key
        )
        logger.info(f"Using OpenAI API with model: {config.model}")
    else:
        # Standard Gemini API usage
        llm = ChatGoogleGenerativeAI(
            model=config.model,
            temperature=config.temperature,
            google_api_key=config.api_key
        )
        logger.info(f"Using Gemini API with model: {config.model}")
    
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful hotel assistant. Use the following hotel information to answer questions.

Hotel Data:
{hotel_context}

When answering questions:
- Be accurate and specific
- Reference hotel names, locations, and details from the data
- If information is not available, say so clearly
- Format responses in a clear, readable way using markdown
- Use bullet points and tables when appropriate
- Include specific prices, addresses, and details when available"""),
        ("human", "{question}")
    ])
    
    # Create the chain
    _agent_chain = prompt_template | llm
    
    return _agent_chain


def answer_hotel_question(question: str) -> str:
    """
    Simple agent that answers questions using hotel files as context.
    
    This function loads hotel data and uses it as context for the LLM
    to answer questions about hotels, rooms, and configurations.
    
    Args:
        question: User's question about hotels
        
    Returns:
        str: Agent's response
        
    Raises:
        FileNotFoundError: If hotel data files don't exist
        ValueError: If configuration is invalid or missing required values
    """
    try:
        # Load hotel data
        hotels_data, hotel_details_text = load_hotel_data()
        
        # Prepare context from loaded files
        hotel_context = f"""
{hotel_details_text}

Hotels JSON Summary:
{json.dumps(hotels_data, indent=2, ensure_ascii=False)}
"""
        
        # Create agent chain
        chain = _create_agent_chain()
        
        # Invoke the chain
        logger.info(f"Processing question: {question[:100]}...")
        response = chain.invoke({
            "hotel_context": hotel_context,
            "question": question
        })
        
        return response.content
        
    except FileNotFoundError as e:
        logger.error(f"Hotel data files not found: {e}")
        return f"""❌ **Error**: Hotel data files not found.

Please generate the hotel data first:
```bash
cd bookings-db
python src/gen_synthetic_hotels.py --num_hotels 3
```

Then restart the API server."""
    
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return f"""❌ **Error**: {str(e)}"""
    
    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        return f"""❌ **Error**: An unexpected error occurred while processing your question.

Error details: {str(e)}

Please try again or contact support if the problem persists."""


async def handle_hotel_query_simple(user_query: str) -> str:
    """
    Handle hotel queries using simple file context approach.
    
    This is the async wrapper for the WebSocket API integration.
    Executes the synchronous agent function in a thread pool to avoid
    blocking the event loop.
    
    Args:
        user_query: User's query string
        
    Returns:
        str: Formatted response from the agent
    """
    # Run the synchronous function in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, answer_hotel_question, user_query)
    return response

