# Comparación: AgentExecutor (LangChain) vs LangGraph

## ¿Qué hace `create_sql_agent` internamente?

Cuando usas:
```python
agent = create_sql_agent(llm, toolkit, verbose=True)
agent.invoke({"input": "¿Cuál es el revenue total?"})
```

Internamente ejecuta un **AgentExecutor** con este loop:

```
1. [Razonamiento] → "Necesito consultar la base de datos"
2. [Acción] → Ejecuta tool: sql_db_list_tables
3. [Observación] → "Tablas: bookings, hotels"
4. [Razonamiento] → "Necesito ver el schema de bookings"
5. [Acción] → Ejecuta tool: sql_db_schema
6. [Observación] → "Columnas: id, total_price, ..."
7. [Razonamiento] → "Ahora puedo generar SQL"
8. [Acción] → Ejecuta tool: sql_db_query con "SELECT SUM(total_price) FROM bookings"
9. [Observación] → "Result: 150000"
10. [Razonamiento] → "Ya tengo la respuesta"
11. [Respuesta Final] → "El revenue total es €150,000"
```

Este loop es **automático** y **no puedes modificarlo**.

---

## ¿Qué puede hacer LangGraph que AgentExecutor NO puede?

### 1. **Validación Custom antes de ejecutar SQL**

**Con AgentExecutor:** No puedes interceptar la query antes de ejecutarla.

**Con LangGraph:**
```python
# Puedes validar ANTES de ejecutar
def validate_performance_node(state):
    sql = state["generated_sql"]
    
    # Custom logic: Estima el costo
    estimated_rows = estimate_query_cost(sql)
    
    if estimated_rows > 1_000_000:
        # Pausa y pide confirmación al usuario
        return {**state, "needs_confirmation": True}
    
    return {**state, "needs_confirmation": False}

# En el grafo:
workflow.add_conditional_edges(
    "validate",
    lambda s: "ask_user" if s["needs_confirmation"] else "execute",
    {"ask_user": "human_approval", "execute": "execute_sql"}
)
```

### 2. **Human-in-the-Loop**

**Con AgentExecutor:** No puedes pausar y esperar input del usuario.

**Con LangGraph:**
```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Guarda estado en cada paso
memory = SqliteSaver.from_conn_string(":memory:")
graph = workflow.compile(checkpointer=memory)

# Primera ejecución
state = graph.invoke({"query": "DELETE old bookings"})

# Si detecta query peligrosa, se pausa
# Usuario puede revisar y aprobar
if state["needs_approval"]:
    # Muestra al usuario: "¿Ejecutar DELETE? [Sí/No]"
    user_confirms = ask_user()
    
    # Resume desde donde se quedó
    final_state = graph.invoke(
        {"user_approved": user_confirms},
        config={"configurable": {"thread_id": "123"}}
    )
```

### 3. **Paralelización de Queries**

**Con AgentExecutor:** Ejecuta queries secuencialmente.

**Con LangGraph:**
```python
# Ejecuta múltiples queries en paralelo
def parallel_queries_node(state):
    queries = [
        "SELECT COUNT(*) FROM bookings",
        "SELECT SUM(total_price) FROM bookings",
        "SELECT AVG(total_nights) FROM bookings"
    ]
    
    # Ejecuta en paralelo
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(execute_sql, queries))
    
    return {**state, "results": results}
```

### 4. **Custom Error Handling**

**Con AgentExecutor:** Si falla, reintenta con el mismo approach.

**Con LangGraph:**
```python
def error_recovery_node(state):
    error = state["error"]
    
    # Custom logic basado en el tipo de error
    if "permission denied" in error:
        return {**state, "next": "request_elevated_access"}
    elif "timeout" in error:
        return {**state, "next": "simplify_query"}
    else:
        return {**state, "next": "fallback_to_rag"}

workflow.add_conditional_edges(
    "error_handler",
    lambda s: s["next"],
    {
        "request_elevated_access": "auth_node",
        "simplify_query": "query_simplifier",
        "fallback_to_rag": "rag_agent"
    }
)
```

### 5. **Observabilidad Granular**

**Con AgentExecutor:** Solo ves logs de cada step.

**Con LangGraph:**
```python
# Puedes guardar métricas de cada nodo
def track_metrics_node(state):
    metrics = {
        "sql_generation_time": state["gen_time"],
        "validation_time": state["val_time"],
        "execution_time": state["exec_time"],
        "total_retries": state["retry_count"]
    }
    
    # Envía a tu sistema de monitoreo
    send_to_datadog(metrics)
    
    return state
```

---

## Decisión Final

| Tu Caso de Uso | Usa esto | Razón |
|----------------|----------|-------|
| **SQL Agent básico** | ✅ `create_sql_agent` | Ya funciona, es simple |
| **SQL con aprobación manual** | ✅ LangGraph | Human-in-the-loop |
| **SQL con validación de performance** | ✅ LangGraph | Custom validation |
| **Orquestador multi-agente** | ✅ LangGraph | Integración compleja |
| **SQL + RAG en paralelo** | ✅ LangGraph | Paralelización |
| **Debugging profundo** | ✅ LangGraph | Observabilidad |

---

## Recomendación para tu Workshop

**Nivel Básico (Ejercicio 1-2):**
- Usa `create_sql_agent` tal como está
- Explica que internamente hace un loop ReAct
- Muestra cómo funciona con verbose=True

**Nivel Intermedio (Ejercicio 3):**
- Muestra el orquestador simple (sin LangGraph)
- Explica las limitaciones del if/else simple

**Nivel Avanzado (Ejercicio 4 - OPCIONAL):**
- Implementa UN caso de LangGraph que demuestre valor real:
  * Human-in-the-loop para queries costosas
  * O validación custom de SQL
  * O fallback automático entre agentes

**NO implementes todo con LangGraph "porque sí".** Usa la herramienta adecuada para cada problema.

---

## Conclusión

- `create_sql_agent` = **"Opinionated framework"** → Fácil pero menos control
- LangGraph = **"Unopinionated primitives"** → Complejo pero control total

Para el 80% de casos, `create_sql_agent` es suficiente.
LangGraph es para el 20% donde necesitas hacer cosas que AgentExecutor no puede.
