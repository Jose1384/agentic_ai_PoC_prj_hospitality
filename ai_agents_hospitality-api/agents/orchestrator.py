from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_sql_agent
from util.logger_config import logger
from config.agent_config import get_agent_config

# Meta-agente: Orchestrator
def create_orchestrator_agent():
    """
    Crea un agente que decide qué agente interno usar según la query.
    """
    global _orchestrator_agent
    if "_orchestrator_agent" in globals() and _orchestrator_agent is not None:
        return _orchestrator_agent

    config = get_agent_config()

    llm = ChatGoogleGenerativeAI(
        model=config.model,
        temperature=0,
        google_api_key=config.api_key
    )

    system_prompt = """
You are a smart Orchestrator Agent.

You have the following agents available:

1. Bookings SQL Agent
   - Specialized in hospitality analytics
   - Can generate SQL queries for bookings table
   - Good for questions like "number of bookings", "occupancy rate", "total revenue"

2. RAG Agent
   - Uses hotel data context for general hotel information
   - Good for questions about hotel facilities, amenities, locations

Your task:
- Given a user query, decide which agent is the best fit.
- Only choose one agent.
- Respond with a JSON containing:
  {
      "agent": "Bookings SQL Agent" | "RAG Agent" ,
      "query": "cleaned query to pass to the chosen agent"
  }
- Be concise and deterministic.
"""

    _orchestrator_agent = create_sql_agent(
        llm=llm,
        toolkit=None,  # No DB needed, it only decides
        verbose=True,
        system_prompt=system_prompt
    )

    return _orchestrator_agent
