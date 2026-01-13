"""
SQL Agent with Validation Loop - REAL LangGraph Power

Este ejemplo muestra donde LangGraph REALMENTE brilla:
- Genera SQL → Valida → Si falla, REINTENTA con feedback
- Ciclo iterativo con límite de reintentos
- Estado compartido entre iteraciones

SIN LangGraph necesitarías:
- While loop manual
- Contador de reintentos
- State management manual
- Try-catch anidados

CON LangGraph:
- Nodos claros (generate, validate, execute)
- Edges condicionales (retry si falla)
- Estado automático

Este es el tipo de patrón donde LangGraph justifica su complejidad.
"""

import asyncio
import re
from typing import TypedDict, Literal, Annotated
from operator import add

from langgraph.graph import StateGraph, END, START
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase

from util.logger_config import logger
from config.agent_config import get_agent_config
from db.session_langchain import get_database


# --- STATE WITH VALIDATION TRACKING ---
class SQLAgentState(TypedDict):
    """
    Estado que fluye por el grafo.
    
    Esto es MÁS COMPLEJO que un simple dict porque:
    - Necesitamos acumular errores de cada intento
    - Rastrear cuántos intentos llevamos
    - Mantener el SQL generado para validación
    - Guardar resultados intermedios
    """
    query: str                          # Pregunta original del usuario
    generated_sql: str                  # SQL generado por el LLM
    validation_errors: Annotated[list, add]  # Lista acumulativa de errores
    execution_result: str               # Resultado final
    retry_count: int                    # Intentos realizados
    max_retries: int                    # Límite de reintentos
    is_valid: bool                      # ¿El SQL pasó validación?
    feedback: str                       # Feedback del validador al generador


# --- NODES ---

def generate_sql_node(state: SQLAgentState) -> SQLAgentState:
    """
    Genera SQL a partir de lenguaje natural.
    
    IMPORTANTE: En reintentos, usa el feedback del validador
    para mejorar la generación.
    """
    retry_count = state.get("retry_count", 0)
    feedback = state.get("feedback", "")
    
    logger.info(f"[SQL Generator] Attempt {retry_count + 1}/{state['max_retries']}")
    
    config = get_agent_config()
    llm = ChatGoogleGenerativeAI(
        model=config.model,
        temperature=0,
        google_api_key=config.api_key
    )
    
    # Get database schema
    db = get_database()
    schema_info = db.get_table_info()
    
    # Construir prompt diferente en reintentos
    if retry_count == 0:
        system_msg = """You are a PostgreSQL expert. Generate a SQL query for the user's question.

Database Schema:
{schema}

Rules:
- Only use tables and columns from the schema
- Return ONLY the SQL query, no explanations
- Use PostgreSQL syntax
- Prefer aggregate functions (SUM, COUNT, AVG)"""
    else:
        system_msg = """You are a PostgreSQL expert. Your previous SQL had errors.

Database Schema:
{schema}

Previous SQL:
{previous_sql}

Validation Error:
{feedback}

Generate a CORRECTED SQL query. Learn from the error above."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "{query}")
    ])
    
    chain = prompt | llm
    
    try:
        result = chain.invoke({
            "query": state["query"],
            "schema": schema_info,
            "previous_sql": state.get("generated_sql", ""),
            "feedback": feedback
        })
        
        # Extract SQL from response
        sql = result.content.strip()
        
        # Clean markdown code blocks if present
        if sql.startswith("```"):
            sql = re.sub(r"```sql\n?", "", sql)
            sql = re.sub(r"```\n?", "", sql)
        
        sql = sql.strip()
        
        logger.info(f"[SQL Generator] Generated:\n{sql[:100]}...")
        
        return {
            **state,
            "generated_sql": sql,
            "retry_count": retry_count + 1
        }
        
    except Exception as e:
        logger.error(f"[SQL Generator] Error: {e}")
        return {
            **state,
            "validation_errors": state.get("validation_errors", []) + [f"Generation error: {str(e)}"],
            "retry_count": retry_count + 1,
            "is_valid": False
        }


def validate_sql_node(state: SQLAgentState) -> SQLAgentState:
    """
    Valida el SQL generado SIN ejecutarlo.
    
    Validaciones:
    1. Sintaxis básica (no puede estar vacío, debe tener SELECT, etc.)
    2. Usa solo tablas del schema
    3. No tiene comandos peligrosos (DROP, DELETE, etc.)
    
    Si falla, genera feedback específico para el generador.
    """
    logger.info("[SQL Validator] Checking SQL...")
    
    sql = state.get("generated_sql", "").strip()
    
    # Validación 1: No vacío
    if not sql:
        error = "Generated SQL is empty"
        logger.warning(f"[SQL Validator] ❌ {error}")
        return {
            **state,
            "is_valid": False,
            "feedback": error,
            "validation_errors": state.get("validation_errors", []) + [error]
        }
    
    # Validación 2: Debe ser SELECT (no modificaciones)
    sql_upper = sql.upper()
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"]
    
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            error = f"SQL contains forbidden keyword: {keyword}"
            logger.warning(f"[SQL Validator] ❌ {error}")
            return {
                **state,
                "is_valid": False,
                "feedback": f"Remove {keyword} - only SELECT queries allowed",
                "validation_errors": state.get("validation_errors", []) + [error]
            }
    
    # Validación 3: Debe tener SELECT
    if "SELECT" not in sql_upper:
        error = "SQL must contain SELECT"
        logger.warning(f"[SQL Validator] ❌ {error}")
        return {
            **state,
            "is_valid": False,
            "feedback": error,
            "validation_errors": state.get("validation_errors", []) + [error]
        }
    
    # Validación 4: Verificar sintaxis básica con el DB (sin ejecutar)
    db = get_database()
    try:
        # PostgreSQL permite EXPLAIN sin ejecutar
        db.run(f"EXPLAIN {sql}", fetch="cursor")
        
        logger.info("[SQL Validator] ✅ SQL is valid")
        return {
            **state,
            "is_valid": True,
            "feedback": ""
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"[SQL Validator] ❌ Syntax error: {error_msg[:100]}")
        
        # Generar feedback útil
        feedback = f"Syntax error: {error_msg}. Check column names and table references."
        
        return {
            **state,
            "is_valid": False,
            "feedback": feedback,
            "validation_errors": state.get("validation_errors", []) + [error_msg]
        }


def execute_sql_node(state: SQLAgentState) -> SQLAgentState:
    """
    Ejecuta el SQL validado y formatea los resultados.
    
    Solo se llama si el SQL pasó validación.
    """
    logger.info("[SQL Executor] Running validated SQL...")
    
    db = get_database()
    sql = state["generated_sql"]
    
    try:
        result = db.run(sql)
        
        logger.info(f"[SQL Executor] ✅ Success - {len(str(result))} chars")
        
        # Format result nicely
        formatted = f"""**SQL Query:**
```sql
{sql}
```

**Results:**
{result}
"""
        
        return {
            **state,
            "execution_result": formatted
        }
        
    except Exception as e:
        logger.error(f"[SQL Executor] ❌ Execution error: {e}")
        return {
            **state,
            "execution_result": f"❌ Execution error: {str(e)}"
        }


def max_retries_node(state: SQLAgentState) -> SQLAgentState:
    """
    Nodo que se ejecuta cuando agotamos los reintentos.
    
    Devuelve un mensaje de error con todos los intentos.
    """
    logger.error("[SQL Agent] ❌ Max retries exceeded")
    
    errors = state.get("validation_errors", [])
    error_summary = "\n".join(f"{i+1}. {e}" for i, e in enumerate(errors))
    
    result = f"""❌ **No se pudo generar SQL válido después de {state['retry_count']} intentos.**

**Errores encontrados:**
{error_summary}

**Última SQL generada:**
```sql
{state.get('generated_sql', 'N/A')}
```

Por favor reformula tu pregunta o proporciona más detalles."""
    
    return {
        **state,
        "execution_result": result
    }


# --- CONDITIONAL ROUTING ---

def should_retry(state: SQLAgentState) -> Literal["retry", "execute", "max_retries"]:
    """
    AQUÍ ESTÁ EL PODER DE LANGGRAPH: Ciclos condicionales.
    
    Decide si:
    - retry: Generar SQL de nuevo (volver al nodo generador)
    - execute: SQL válido, ejecutar
    - max_retries: Agotamos intentos, terminar con error
    """
    is_valid = state.get("is_valid", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    
    if is_valid:
        logger.info("[Router] ✅ SQL valid → execute")
        return "execute"
    
    if retry_count >= max_retries:
        logger.warning(f"[Router] ❌ Max retries ({max_retries}) reached")
        return "max_retries"
    
    logger.info(f"[Router] 🔄 Retry {retry_count}/{max_retries}")
    return "retry"


# --- GRAPH CONSTRUCTION ---

def create_sql_validation_graph():
    """
    Build the iterative validation graph.
    
    Flow:
        Generate SQL → Validate
                         ↓
                    Valid? 
                    ├─ Yes → Execute → END
                    ├─ No (retries left) → Generate SQL (loop back)
                    └─ No (max retries) → Error Message → END
    
    Este patrón (loop con condición) es DIFÍCIL sin LangGraph.
    """
    workflow = StateGraph(SQLAgentState)
    
    # Add nodes
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("validate_sql", validate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("max_retries_error", max_retries_node)
    
    # Entry point
    workflow.add_edge(START, "generate_sql")
    
    # Generate → Validate (always)
    workflow.add_edge("generate_sql", "validate_sql")
    
    # Validate → Router (conditional)
    workflow.add_conditional_edges(
        "validate_sql",
        should_retry,
        {
            "retry": "generate_sql",        # LOOP BACK! 🔄
            "execute": "execute_sql",       # Success path
            "max_retries": "max_retries_error"  # Failure path
        }
    )
    
    # End points
    workflow.add_edge("execute_sql", END)
    workflow.add_edge("max_retries_error", END)
    
    return workflow.compile()


# --- API ---

_graph = None

def get_sql_validation_graph():
    global _graph
    if _graph is None:
        logger.info("Compiling SQL validation graph with retry loop...")
        _graph = create_sql_validation_graph()
    return _graph


def sql_agent_with_validation(user_query: str, max_retries: int = 3) -> str:
    """
    Execute SQL agent with automatic retry on validation errors.
    
    Args:
        user_query: Natural language question
        max_retries: Maximum retry attempts (default 3)
        
    Returns:
        SQL execution results or error message
    """
    graph = get_sql_validation_graph()
    
    initial_state = {
        "query": user_query,
        "generated_sql": "",
        "validation_errors": [],
        "execution_result": "",
        "retry_count": 0,
        "max_retries": max_retries,
        "is_valid": False,
        "feedback": ""
    }
    
    logger.info(f"[SQL Agent] Starting with validation loop: {user_query[:50]}...")
    
    try:
        final_state = graph.invoke(initial_state)
        
        logger.info(f"[SQL Agent] Completed in {final_state['retry_count']} attempts")
        
        return final_state["execution_result"]
        
    except Exception as e:
        logger.error(f"[SQL Agent] Critical error: {e}", exc_info=True)
        return f"❌ Error crítico: {str(e)}"


async def handle_sql_with_validation(user_query: str) -> str:
    """Async wrapper."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, sql_agent_with_validation, user_query)


# --- VISUALIZATION ---

def get_graph_diagram():
    return """
```mermaid
graph TD
    START((Start)) --> Gen[🔧 Generate SQL]
    Gen --> Val[✓ Validate SQL]
    
    Val -->|Valid| Exec[▶️ Execute SQL]
    Val -->|Invalid + Retries Left| Gen
    Val -->|Invalid + Max Retries| Error[❌ Error Message]
    
    Exec --> END((End))
    Error --> END
    
    style Gen fill:#e1f5ff
    style Val fill:#fff4e1
    style Exec fill:#d4edda
    style Error fill:#f8d7da
```

**The Loop is the Key**: Sin LangGraph necesitarías un while loop manual 
con state tracking complejo. LangGraph hace el loop automáticamente.
"""


if __name__ == "__main__":
    print(get_graph_diagram())
    
    # Test con una query que probablemente genere SQL inválido primero
    test_query = "Show me the top 5 hotels by revenue with their ocupancy rates"
    
    print("\n" + "="*60)
    print(f"Testing validation loop: {test_query}")
    print("="*60)
    
    result = sql_agent_with_validation(test_query, max_retries=3)
    print(f"\n{result}")
