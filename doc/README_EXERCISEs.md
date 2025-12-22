
## Exercise 0

* Only the functionality for selecting external hotel data has been modified.

---

## Exercise 1
> Not all desired tests could be performed due to `rate-limiting` issues, especially when trying to optimize and compare results.

1. **Generate Hotel Data**

   Generate 50 hotels using the following command from the project root:
   ```bash
   python src/gen_synthetic_hotels.py --num_hotels 50
   ```

2. **Document Ingestion**

   The script `ingest_documents.py` is located in the `ai_agents_hospitality-api/scripts` directory. It loads hotel information files into a vector database. Two ingestion implementations are provided:
   - Ingestion into ChromaDB
   - Ingestion into pgvector

   > **Note:**
   > Run the `ingest_documents.py` script only once before testing the RAG functionality. This will create a local vector database with the loaded documents. See `docs/IDE` for VSCode configuration if needed.

3. **Configuration**

   Update `config/agent_config.yaml` to enable RAG search and configure document ingestion:

   ```yaml
   hotels:
     local: False

   rag:
     active: True
     chunk_size: 1000
     chunk_overlap: 200
     batch_size: 10000
   ```

4. **RAG Functionality**

   Retrieval-Augmented Generation (RAG) is implemented and exposed via WebSocket.

---

### Embedding Models and Rate Limiting

To work with embeddings and avoid rate limiting, you can use different models:

```python
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=_get_env_value("AI_AGENTIC_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=_get_env_value("AI_AGENTIC_API_KEY"))
```

## Exerise 2

### Resumen de cambios realizados en el Ejercicio 2

En este ejercicio se han implementado dos mejoras principales:

1. **Agente SQL para analítica sobre PostgreSQL:**
   - Se ha creado un agente especializado que permite convertir preguntas en lenguaje natural a consultas SQL sobre la base de datos de reservas hoteleras (PostgreSQL).
   - El agente es capaz de devolver métricas de negocio como bookings, revenue, occupancy, RevPAR, etc., utilizando únicamente el esquema real de la base de datos.

2. **Orquestador de agentes:**
   - Se ha modificado el orquestador principal (WebSocket/API) para decidir dinámicamente qué agente debe responder a cada consulta.
   - Según el tipo de pregunta, el orquestador selecciona entre el agente RAG (para preguntas sobre información textual/documental) o el agente SQL (para analítica estructurada sobre la base de datos).
   - Esto permite una experiencia unificada y transparente para el usuario final.

**Configuración necesaria:**

- Es imprescindible definir la variable de entorno `DB_URI` con la cadena de conexión a la base de datos PostgreSQL para que el agente SQL funcione correctamente.
  - Ejemplo: `export DB_URI=postgresql://usuario:password@host:puerto/nombre_db`
