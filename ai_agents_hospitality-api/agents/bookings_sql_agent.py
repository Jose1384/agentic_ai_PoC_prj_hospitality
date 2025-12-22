"""
Bookings SQL Agent

Agent especializado en analítica hotelera sobre PostgreSQL.
Convierte lenguaje natural en consultas SQL y devuelve métricas
de negocio (bookings, revenue, occupancy, RevPAR, etc.).

Este agente complementa al Hotel Agent (RAG / Simple),
añadiendo capacidades analíticas sobre datos estructurados.
"""

import asyncio
from typing import Any, Dict

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_classic.agents import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    from langchain_community.chat_models import ChatGoogleGenerativeAI

from util.logger_config import logger
from config.agent_config import get_agent_config
from db.db_session import get_database

_bookings_sql_agent = None # Lazy singleton instance - do not modify directly, do not use this on production environments

async def handle_bookings_sql_query(user_query: str) -> str:
    """
    Async wrapper for WebSocket / API usage.
    """
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, run_bookings_analytics, user_query
    )
    return response

def run_bookings_analytics(question: str) -> str:
    """
    Two-step analytics process:
    1. Agent generates SQL from natural language
    2. SQL is executed and results are formatted
    """
    try:
        agent = create_bookings_sql_agent()
        logger.info(f"Processing analytics question: {question}")

        # Step 1 + 2 happen internally in the SQL Agent
        response = agent.invoke({"input": question})

        return response["output"]

    except Exception as e:
        logger.error("Error running bookings analytics", exc_info=True)
        return f"❌ **Error**: {str(e)}"


def create_bookings_sql_agent():
    """
    Create a SQL Agent specialized in hospitality analytics.
    """
    global _bookings_sql_agent
    if _bookings_sql_agent is not None:
        return _bookings_sql_agent

    logger.info("Initializing Bookings SQL Agent...")


    config = get_agent_config()

    llm = ChatGoogleGenerativeAI(
        model=config.model,
        temperature=0,  # Importante para SQL determinista
        google_api_key=config.api_key
    )

    # Define tools
    db = get_database()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm) # TODO(0): add more tools for analytics

    system_prompt = """
You are a senior hospitality data analyst.

You work with a PostgreSQL database containing hotel bookings data.
Your task is to:
- Translate user questions into correct SQL queries
- Use ONLY the available database schema
- Never hallucinate tables or columns
- Prefer aggregate queries for analytics
- Use clear aliases for metrics

Important business definitions:
- Total Revenue = SUM(total_price)
- Total Occupied Nights = SUM(total_nights)
- Occupancy Rate = (Total Occupied Nights / Total Available Room-Nights) * 100
- RevPAR = Total Revenue / Total Available Room-Nights

Assumptions:
- Each row represents one booking
- total_nights already represents nights stayed
- total_price is in EUR
- Dates are stored as DATE

Always return SQL compatible with PostgreSQL.
"""

    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        system_prompt=system_prompt
    )

    return agent



# --- DIFINE TOOLS FOR ANALYTICS METRICS CALCULATION ---
 #TODO(0): Add analysis-specific tools to avoid depending just on toolkit-sql
@tool
def calculate_revpar(total_revenue: float, total_available_room_nights: int) -> float:
        """"Calculate RevPAR given total revenue and total available room-nights."""
        if total_available_room_nights == 0:
              return 0.0
        return total_revenue / total_available_room_nights