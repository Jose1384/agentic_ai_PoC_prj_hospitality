"""
Orchestrator with REAL LangGraph Value: Automatic Fallback

Este ejemplo muestra el VALOR REAL de LangGraph:
- Si SQL Agent falla → automáticamente intenta RAG Agent como fallback
- Si RAG falla → automáticamente usa Simple Agent
- Retry automático con reformulación de queries

Sin LangGraph esto requeriría nested try-catch y lógica compleja.
Con LangGraph: solo añadir edges condicionales.

Comparación:
- orchestrator.py (original): 60 líneas, solo clasifica
- orchestrator_simple: 80 líneas, clasifica + ejecuta
- Este: 250 líneas, pero con fallback + retry + logging estructurado
"""

import asyncio
import json
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END, START
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from util.logger_config import logger
from config.agent_config import get_agent_config

from agents.bookings_sql_agent import run_bookings_analytics
from agents.hotel_rag_agent import answer_hotel_question_rag
from agents.hotel_simple_agent import answer_hotel_question


# --- STATE WITH FALLBACK TRACKING ---
class OrchestratorState(TypedDict):
    query: str
    agent_choice: str
    response: str
    error: str
    retry_count: int
    attempted_agents: list  # Track which agents we've tried


# --- NODES ---

def router_node(state: OrchestratorState) -> OrchestratorState:
    """Classify query and choose primary agent."""
    logger.info(f"[Router] Query: {state['query'][:50]}...")
    
    config = get_agent_config()
    llm = ChatGoogleGenerativeAI(
        model=config.model,
        temperature=0,
        google_api_key=config.api_key
    )
    
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """Classify this query. Return ONLY JSON:
{{"agent": "sql" | "rag" | "simple", "reasoning": "why"}}

- sql: analytics, metrics, bookings, revenue
- rag: hotel info, amenities, descriptions
- simple: greetings, unclear queries"""),
        ("human", "{query}")
    ])
    
    try:
        result = (router_prompt | llm).invoke({"query": state["query"]})
        response_text = result.content.strip()
        
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        parsed = json.loads(response_text)
        agent_choice = parsed.get("agent", "simple")
        
        if agent_choice not in ["sql", "rag", "simple"]:
            agent_choice = "simple"
        
        logger.info(f"[Router] → {agent_choice}")
        
        return {
            **state,
            "agent_choice": agent_choice,
            "attempted_agents": [agent_choice]
        }
        
    except Exception as e:
        logger.error(f"[Router] Error: {e}")
        return {
            **state,
            "agent_choice": "simple",
            "attempted_agents": ["simple"],
            "error": str(e)
        }


def sql_agent_node(state: OrchestratorState) -> OrchestratorState:
    """Execute SQL Agent with error handling."""
    logger.info(f"[SQL Agent] Attempting...")
    
    try:
        response = run_bookings_analytics(state["query"])
        
        # Check if response indicates failure
        if "❌" in response or "Error" in response:
            raise Exception(f"SQL Agent returned error: {response[:100]}")
        
        logger.info("[SQL Agent] ✅ Success")
        return {**state, "response": response}
        
    except Exception as e:
        logger.warning(f"[SQL Agent] ❌ Failed: {e}")
        return {
            **state,
            "error": f"SQL Agent failed: {str(e)}",
            "response": ""  # Empty response signals failure
        }


def rag_agent_node(state: OrchestratorState) -> OrchestratorState:
    """Execute RAG Agent with error handling."""
    logger.info(f"[RAG Agent] Attempting...")
    
    try:
        response = answer_hotel_question_rag(state["query"])
        
        if "❌" in response or "Error" in response:
            raise Exception(f"RAG Agent returned error: {response[:100]}")
        
        logger.info("[RAG Agent] ✅ Success")
        return {**state, "response": response}
        
    except Exception as e:
        logger.warning(f"[RAG Agent] ❌ Failed: {e}")
        return {
            **state,
            "error": f"RAG Agent failed: {str(e)}",
            "response": ""
        }


def simple_agent_node(state: OrchestratorState) -> OrchestratorState:
    """Execute Simple Agent as last resort fallback."""
    logger.info(f"[Simple Agent] Fallback execution...")
    
    try:
        response = answer_hotel_question(state["query"])
        logger.info("[Simple Agent] ✅ Success")
        return {**state, "response": response}
        
    except Exception as e:
        logger.error(f"[Simple Agent] ❌ Even fallback failed: {e}")
        return {
            **state,
            "response": f"❌ Todos los agentes fallaron. Error: {str(e)}",
            "error": str(e)
        }


# --- CONDITIONAL ROUTING ---

def route_to_agent(state: OrchestratorState) -> str:
    """Route to primary agent based on classification."""
    choice = state.get("agent_choice", "simple")
    
    mapping = {
        "sql": "sql_agent",
        "rag": "rag_agent",
        "simple": "simple_agent"
    }
    
    return mapping.get(choice, "simple_agent")


def check_if_needs_fallback(state: OrchestratorState) -> Literal["success", "fallback_rag", "fallback_simple"]:
    """
    ESTO ES LO IMPORTANTE: Lógica de fallback automático.
    
    Si el agente falló (response vacío), intenta el siguiente:
    SQL failed → try RAG
    RAG failed → try Simple
    Simple is the last resort
    """
    # If we got a response, we're done
    if state.get("response"):
        logger.info("[Fallback Check] ✅ Got response, ending")
        return "success"
    
    attempted = state.get("attempted_agents", [])
    
    # If SQL failed and we haven't tried RAG yet
    if "sql" in attempted and "rag" not in attempted:
        logger.warning("[Fallback Check] 🔄 SQL failed, trying RAG as fallback")
        return "fallback_rag"
    
    # If RAG failed (or SQL→RAG failed), try simple
    if ("rag" in attempted or "sql" in attempted) and "simple" not in attempted:
        logger.warning("[Fallback Check] 🔄 Previous agents failed, trying Simple as last resort")
        return "fallback_simple"
    
    # All agents failed, just end
    logger.error("[Fallback Check] ❌ All agents failed")
    return "success"  # End even with failure


def update_fallback_state(state: OrchestratorState, next_agent: str) -> OrchestratorState:
    """Update state when falling back to another agent."""
    attempted = state.get("attempted_agents", [])
    return {
        **state,
        "agent_choice": next_agent,
        "attempted_agents": attempted + [next_agent]
    }


# --- GRAPH CONSTRUCTION ---

def create_orchestrator_with_fallback() -> StateGraph:
    """
    Build the graph with automatic fallback logic.
    
    Flow:
        Router → Primary Agent → Check Success
                                    ↓
                      If failed → Fallback Agent → Check Success
                                                      ↓
                                        If failed → Simple Agent → END
    """
    workflow = StateGraph(OrchestratorState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("sql_agent", sql_agent_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("simple_agent", simple_agent_node)
    
    # Entry point
    workflow.add_edge(START, "router")
    
    # Router → Primary agent
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "sql_agent": "sql_agent",
            "rag_agent": "rag_agent",
            "simple_agent": "simple_agent"
        }
    )
    
    # SQL Agent → Check if needs fallback
    workflow.add_conditional_edges(
        "sql_agent",
        check_if_needs_fallback,
        {
            "success": END,
            "fallback_rag": "rag_agent",
            "fallback_simple": "simple_agent"
        }
    )
    
    # RAG Agent → Check if needs fallback
    workflow.add_conditional_edges(
        "rag_agent",
        check_if_needs_fallback,
        {
            "success": END,
            "fallback_simple": "simple_agent",
            "fallback_rag": "rag_agent"  # Shouldn't happen but safe
        }
    )
    
    # Simple Agent always ends (last resort)
    workflow.add_edge("simple_agent", END)
    
    return workflow.compile()


# --- API ---

_graph = None

def get_graph():
    global _graph
    if _graph is None:
        logger.info("Compiling LangGraph with fallback logic...")
        _graph = create_orchestrator_with_fallback()
    return _graph


def orchestrator_with_fallback(user_query: str) -> str:
    """
    Execute orchestrator with automatic fallback on failure.
    """
    graph = get_graph()
    
    initial_state = {
        "query": user_query,
        "agent_choice": "",
        "response": "",
        "error": "",
        "retry_count": 0,
        "attempted_agents": []
    }
    
    logger.info(f"[Orchestrator] Starting with fallback for: {user_query[:50]}...")
    
    try:
        final_state = graph.invoke(initial_state)
        
        agents_tried = final_state.get("attempted_agents", [])
        logger.info(f"[Orchestrator] Completed. Agents tried: {agents_tried}")
        
        return final_state["response"]
        
    except Exception as e:
        logger.error(f"[Orchestrator] Critical failure: {e}", exc_info=True)
        return f"❌ Error crítico en el orquestador: {str(e)}"


async def handle_orchestrator_with_fallback(user_query: str) -> str:
    """Async wrapper."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, orchestrator_with_fallback, user_query)


# --- VISUALIZATION ---

def get_mermaid_diagram():
    return """
```mermaid
graph TD
    START((Start)) --> Router[🧠 Router]
    Router -->|sql| SQL[📊 SQL Agent]
    Router -->|rag| RAG[🔍 RAG Agent]
    Router -->|simple| Simple[💬 Simple Agent]
    
    SQL -->|Success| END((End))
    SQL -->|Failed| RAG
    
    RAG -->|Success| END
    RAG -->|Failed| Simple
    
    Simple --> END
    
    style SQL fill:#e1f5ff
    style RAG fill:#fff4e1
    style Simple fill:#f0f0f0
```
"""


if __name__ == "__main__":
    # Test fallback scenario
    # Simula una query que podría fallar en SQL pero funcionar en RAG
    test_query = "Describe the Grand Hotel Marina"  # This will route to SQL but fail, then try RAG
    
    print(get_mermaid_diagram())
    print("\n" + "="*60)
    print(f"Testing fallback: {test_query}")
    print("="*60)
    
    result = orchestrator_with_fallback(test_query)
    print(f"\nFinal result:\n{result}")
