from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
from util.logger_config import logger
from config.agent_config import get_agent_config
import asyncio

try:
    # Try new LangChain structure (v0.2+)
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    # Fallback to old structure (v0.1)
    from langchain_classic.prompts import ChatPromptTemplate


async def handle_orchestrator(user_query: str) -> str:
    """oRchestrator agent async wrapper for WebSocket / API usage."""
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, orchestrator_agent, user_query
    )
    return response


def orchestrator_agent(user_query: str) -> str:
    """orchestrator agent to route user queries to appropriate agents."""
    try:
        chain = _create_orchestrator_chain()
        logger.info(f"Processing RAG question: {user_query[:100]}...")

        result = chain.invoke({"user_query": user_query})
        
        return result.content

    except Exception as e:
        logger.error("Error processing RAG question", exc_info=True)
        return f"❌ **Error**: {str(e)}"


def _create_orchestrator_chain():
    config = get_agent_config()

    llm = ChatGoogleGenerativeAI(
        model=config.model,
        temperature=0,
        google_api_key=config.api_key
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
You are a smart Orchestrator Agent.
Available agents:

1. Bookings SQL Agent
2. RAG Agent

Task:
- Given a user query, decide which agent is the best fit.
- Only choose one agent.
- Return a JSON with this exact structure (use double curly braces to escape):
  {{
      "agent": "Bookings SQL Agent" | "RAG Agent"
  }}
"""),
        ("human", "{user_query}")
    ])

    # Devuelve un LLMChain combinando prompt + LLM
    chain = prompt_template | llm
    return chain
