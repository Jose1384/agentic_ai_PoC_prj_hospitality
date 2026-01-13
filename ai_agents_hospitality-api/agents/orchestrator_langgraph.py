"""
Orchestrator Agent with LangGraph

This module implements a multi-agent orchestrator using LangGraph.
It routes user queries to the appropriate specialized agent:
- Bookings SQL Agent: For analytics and database queries
- Hotel RAG Agent: For hotel information retrieval
- Hotel Simple Agent: Fallback for simple questions

The graph flow:
    User Query → Router → [SQL Agent | RAG Agent | Simple Agent] → Response

LangGraph provides:
- Explicit state management
- Conditional routing between agents
- Potential for retries and fallbacks
- Visual debugging of agent flows
"""

import asyncio
import json
from typing import TypedDict, Literal, Annotated
from operator import add

from langgraph.graph import StateGraph, END, START
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from util.logger_config import logger
from config.agent_config import get_agent_config

# Import the specialized agents
from agents.bookings_sql_agent import run_bookings_analytics
from agents.hotel_rag_agent import answer_hotel_question_rag
from agents.hotel_simple_agent import answer_hotel_question


# --- STATE DEFINITION ---
class OrchestratorState(TypedDict):
    """
    State object that flows through the LangGraph.
    
    Attributes:
        query: Original user question
        agent_choice: Which agent to route to ("sql", "rag", "simple")
        response: Final response from the chosen agent
        reasoning: Router's explanation for the choice
        error: Error message if something fails
    """
    query: str
    agent_choice: str
    response: str
    reasoning: str
    error: str


# --- NODE FUNCTIONS ---

def router_node(state: OrchestratorState) -> OrchestratorState:
    """
    Classify the user query and decide which agent should handle it.
    
    Decision criteria:
    - SQL Agent: Questions about bookings, revenue, occupancy, analytics, metrics
    - RAG Agent: Questions about hotel details, amenities, locations, descriptions
    - Simple Agent: General questions, fallback when uncertain
    """
    logger.info(f"[Router] Processing query: {state['query'][:100]}...")
    
    config = get_agent_config()
    
    llm = ChatGoogleGenerativeAI(
        model=config.model,
        temperature=0,  # Deterministic routing
        google_api_key=config.api_key
    )
    
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query router for a hotel management system.
Analyze the user's question and decide which specialized agent should handle it.

Available agents:

1. **sql** - Bookings SQL Agent
   - Use for: Analytics, metrics, statistics, revenue, occupancy rates, bookings data
   - Keywords: "how many", "total", "average", "revenue", "bookings", "occupancy", "RevPAR", "statistics", "compare", "trend"
   - Examples: "What's the total revenue?", "How many bookings last month?", "Show occupancy rates"

2. **rag** - Hotel RAG Agent  
   - Use for: Hotel information, descriptions, amenities, room types, locations, policies
   - Keywords: "tell me about", "describe", "what amenities", "where is", "hotel details", "room types", "facilities"
   - Examples: "Describe Hotel Marina", "What amenities does the Grand Hotel have?", "Tell me about rooms"

3. **simple** - Simple Agent (Fallback)
   - Use for: General questions, greetings, unclear queries, or when both SQL and RAG could work
   - Examples: "Hello", "What can you help me with?", "Thanks"

Respond with a JSON object:
{{
    "agent": "sql" | "rag" | "simple",
    "reasoning": "Brief explanation of why this agent was chosen"
}}

Important: Only respond with the JSON, no additional text."""),
        ("human", "{query}")
    ])
    
    chain = router_prompt | llm
    
    try:
        result = chain.invoke({"query": state["query"]})
        
        # Parse the JSON response
        response_text = result.content.strip()
        # Handle potential markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        parsed = json.loads(response_text)
        agent_choice = parsed.get("agent", "simple")
        reasoning = parsed.get("reasoning", "No reasoning provided")
        
        # Validate agent choice
        if agent_choice not in ["sql", "rag", "simple"]:
            agent_choice = "simple"
            reasoning = f"Invalid agent choice '{agent_choice}', falling back to simple"
        
        logger.info(f"[Router] Decision: {agent_choice} - {reasoning}")
        
        return {
            **state,
            "agent_choice": agent_choice,
            "reasoning": reasoning
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"[Router] Failed to parse LLM response: {e}")
        return {
            **state,
            "agent_choice": "simple",
            "reasoning": "Failed to parse router response, using fallback"
        }
    except Exception as e:
        logger.error(f"[Router] Error: {e}")
        return {
            **state,
            "agent_choice": "simple",
            "reasoning": f"Router error: {str(e)}, using fallback",
            "error": str(e)
        }


def sql_agent_node(state: OrchestratorState) -> OrchestratorState:
    """Execute the Bookings SQL Agent for analytics queries."""
    logger.info(f"[SQL Agent] Processing: {state['query'][:50]}...")
    
    try:
        response = run_bookings_analytics(state["query"])
        logger.info("[SQL Agent] Completed successfully")
        return {
            **state,
            "response": response
        }
    except Exception as e:
        logger.error(f"[SQL Agent] Error: {e}")
        return {
            **state,
            "response": f"❌ Error en SQL Agent: {str(e)}",
            "error": str(e)
        }


def rag_agent_node(state: OrchestratorState) -> OrchestratorState:
    """Execute the Hotel RAG Agent for information retrieval."""
    logger.info(f"[RAG Agent] Processing: {state['query'][:50]}...")
    
    try:
        response = answer_hotel_question_rag(state["query"])
        logger.info("[RAG Agent] Completed successfully")
        return {
            **state,
            "response": response
        }
    except Exception as e:
        logger.error(f"[RAG Agent] Error: {e}")
        return {
            **state,
            "response": f"❌ Error en RAG Agent: {str(e)}",
            "error": str(e)
        }


def simple_agent_node(state: OrchestratorState) -> OrchestratorState:
    """Execute the Simple Agent as fallback."""
    logger.info(f"[Simple Agent] Processing: {state['query'][:50]}...")
    
    try:
        response = answer_hotel_question(state["query"])
        logger.info("[Simple Agent] Completed successfully")
        return {
            **state,
            "response": response
        }
    except Exception as e:
        logger.error(f"[Simple Agent] Error: {e}")
        return {
            **state,
            "response": f"❌ Error en Simple Agent: {str(e)}",
            "error": str(e)
        }


def format_response_node(state: OrchestratorState) -> OrchestratorState:
    """
    Optional: Format the final response with metadata.
    Could add agent attribution, confidence scores, etc.
    """
    # Add routing info as a subtle footer (optional)
    agent_names = {
        "sql": "📊 Bookings Analytics",
        "rag": "🔍 Hotel Knowledge Base",
        "simple": "💬 General Assistant"
    }
    
    agent_name = agent_names.get(state["agent_choice"], "Assistant")
    
    # You can uncomment this to show which agent responded:
    # formatted_response = f"{state['response']}\n\n---\n*Answered by: {agent_name}*"
    
    return state


# --- ROUTING FUNCTION ---

def route_to_agent(state: OrchestratorState) -> Literal["sql_agent", "rag_agent", "simple_agent"]:
    """
    Conditional edge function that routes to the appropriate agent node.
    """
    choice = state.get("agent_choice", "simple")
    
    if choice == "sql":
        return "sql_agent"
    elif choice == "rag":
        return "rag_agent"
    else:
        return "simple_agent"


# --- GRAPH CONSTRUCTION ---

def create_orchestrator_graph() -> StateGraph:
    """
    Build the LangGraph orchestrator.
    
    Graph structure:
    
        START
          │
          ▼
       ┌──────────┐
       │  Router  │
       └────┬─────┘
            │
      ┌─────┼─────┐
      │     │     │
      ▼     ▼     ▼
    ┌───┐ ┌───┐ ┌──────┐
    │SQL│ │RAG│ │Simple│
    └─┬─┘ └─┬─┘ └──┬───┘
      │     │      │
      └─────┼──────┘
            │
            ▼
          END
    """
    # Initialize the graph with our state type
    workflow = StateGraph(OrchestratorState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("sql_agent", sql_agent_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("simple_agent", simple_agent_node)
    
    # Define the entry point
    workflow.add_edge(START, "router")
    
    # Add conditional edges from router to agents
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "sql_agent": "sql_agent",
            "rag_agent": "rag_agent",
            "simple_agent": "simple_agent"
        }
    )
    
    # All agents lead to END
    workflow.add_edge("sql_agent", END)
    workflow.add_edge("rag_agent", END)
    workflow.add_edge("simple_agent", END)
    
    return workflow.compile()


# --- SINGLETON GRAPH INSTANCE ---
_orchestrator_graph = None

def get_orchestrator_graph():
    """Get or create the compiled orchestrator graph (singleton)."""
    global _orchestrator_graph
    if _orchestrator_graph is None:
        logger.info("Compiling LangGraph orchestrator...")
        _orchestrator_graph = create_orchestrator_graph()
        logger.info("LangGraph orchestrator ready!")
    return _orchestrator_graph


# --- PUBLIC API ---

def orchestrator_agent_langgraph(user_query: str) -> str:
    """
    Main entry point for the LangGraph orchestrator.
    
    Args:
        user_query: The user's question
        
    Returns:
        str: The response from the appropriate agent
    """
    graph = get_orchestrator_graph()
    
    # Initialize state
    initial_state: OrchestratorState = {
        "query": user_query,
        "agent_choice": "",
        "response": "",
        "reasoning": "",
        "error": ""
    }
    
    # Run the graph
    logger.info(f"[Orchestrator] Starting graph execution for: {user_query[:50]}...")
    
    try:
        final_state = graph.invoke(initial_state)
        logger.info(f"[Orchestrator] Completed. Agent used: {final_state['agent_choice']}")
        return final_state["response"]
    except Exception as e:
        logger.error(f"[Orchestrator] Graph execution failed: {e}", exc_info=True)
        return f"❌ Error en el orquestador: {str(e)}"


async def handle_orchestrator_langgraph(user_query: str) -> str:
    """
    Async wrapper for WebSocket / API usage.
    """
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, orchestrator_agent_langgraph, user_query
    )
    return response


# --- UTILITIES FOR DEBUGGING ---

def get_graph_visualization():
    """
    Return a Mermaid diagram of the graph for visualization.
    Useful for documentation and debugging.
    """
    return """
```mermaid
graph TD
    START((Start)) --> Router[🧠 Router Node]
    Router -->|sql| SQL[📊 SQL Agent]
    Router -->|rag| RAG[🔍 RAG Agent]
    Router -->|simple| Simple[💬 Simple Agent]
    SQL --> END((End))
    RAG --> END
    Simple --> END
```
"""


if __name__ == "__main__":
    # Quick test
    test_queries = [
        "¿Cuál es el revenue total del último mes?",
        "Describe el Hotel Marina Bay",
        "Hola, ¿qué puedes hacer?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        result = orchestrator_agent_langgraph(query)
        print(f"Response: {result[:200]}...")
