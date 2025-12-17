# 📋 TODO - Agentic AI Hospitality PoC

> Last updated: 2025-12-06

---

## 🔥 In Progress (Current)

| Task | Priority | Started | Notes |
|------|----------|---------|-------|
| _No tasks in progress_ | - | - | - |

---

## 📌 Pending (Backlog)

### High Priority
| # | Task | Created | Context |
|---|------|--------|---------|
| 1 | xxxx | 2025-12-06 | xxxx |
| 2 | xxxy | 2025-12-06 | xxxy |
| 3 | xxxz | 2025-12-06 | xxxz |

### Medium Priority
| # | Task | Created | Context |
|---|------|--------|---------|
| - | _No tasks_ | - | - |

### Low Priority
| # | Task | Created | Context |
|---|------|--------|---------|
| - | _No tasks_ | - | - |

---

## ✅ Completed (Done)

| Task | Completed | Commit | Notes |
|------|-----------|--------|-------|
| _No tasks completed yet_ | - | - | - |

---

## 🐛 Technical Debt

| Description | Impact | Detected | Status |
|-------------|--------|----------|--------|
| _No technical debt registered_ | - | - | - |

---

## 📝 Usage Notes

### How to manage this file

1. **New task** → Add to **Backlog** with date and priority
2. **Start task** → Move to **In Progress** with start date
3. **Complete task** → Move to **Completed** with date and commit hash
4. **Technical debt** → Register in specific section to not forget it

### Commit format
When you complete a task, reference the commit like this:
- Short hash: `abc1234`
- With link (if using GitHub): `[abc1234](url-to-commit)`

### Priorities
- 🔴 **High**: Blocks other tasks or is critical
- 🟡 **Medium**: Important but not urgent
- 🟢 **Low**: Nice-to-have, minor improvements

---

## 🎓 Workshop Exercise Plans

### Exercise 0: Simple Agentic Assistant with File Context

#### Phase 1: Setup & Data Preparation
- [X] Install LangChain dependencies (`langchain`, `langchain-google-genai`)
- [X] Configure Google Gemini API key as environment variable (`AI_AGENTIC_API_KEY`)
- [X] Generate synthetic hotel data (3 hotels) using `gen_synthetic_hotels.py`
- [X] Verify hotel files are created in `bookings-db/output_files/hotels/`

#### Phase 2: Core Implementation
- [X] Create function to load hotel JSON file (`hotels.json`)
- [X] Create function to load hotel details markdown (`hotel_details.md`)
- [X] Implement `answer_hotel_question()` function with file context
- [X] Create ChatPromptTemplate with system prompt for hotel assistant
- [X] Build LangChain chain (prompt template + LLM)

#### Phase 3: Integration & Testing
- [X] Create `handle_hotel_query_simple()` async function for WebSocket API
- [X] Test with basic queries (hotel names, addresses, locations)
- [X] Test with meal plan queries
- [X] Test with room information queries
- [X] Verify error handling works correctly

#### Phase 4: Documentation & Cleanup
- [X] Add code comments and docstrings
- [X] Test integration with WebSocket API endpoint
- [X] Verify responses are properly formatted

---

### Exercise 1: Hotel Details with RAG

#### Phase 1: Setup & Data Preparation
- [X] Install RAG dependencies (`langchain-community`, `chromadb`)
- [X] Generate full hotel dataset (50 hotels) using `gen_synthetic_hotels.py`
- [X] Verify all hotel files are created (JSON, markdown files)

#### Phase 2: Vector Store Creation
- [X] Implement document loader for `hotels.json` (JSONLoader)
- [X] Implement document loader for `hotel_details.md` (TextLoader)
- [X] Implement document loader for `hotel_rooms.md` (TextLoader)
- [X] Configure RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
- [X] Create GoogleGenerativeAIEmbeddings instance
- [X] Build ChromaDB vector store from all documents
- [X] Persist vector store to disk for reuse

#### Phase 3: RAG Chain Implementation
- [X] Create ChatGoogleGenerativeAI LLM instance (gemini-2.5-flash-lite, temperature=0)
- [X] Implement RetrievalQA chain with vector store
- [X] Design system prompt for hotel assistant context
- [X] Configure retrieval parameters (k=5 documents)
- [X] Test retrieval quality with sample queries

#### Phase 4: Agent Implementation
- [ ] Create hotel details agent function
- [ ] Implement query preprocessing (normalization, validation)
- [ ] Add response formatting (markdown structure)
- [ ] Handle edge cases (no results, ambiguous queries)

#### Phase 5: Integration & Testing
- [ ] Integrate RAG agent with WebSocket API
- [ ] Test with hotel location queries
- [ ] Test with meal plan and pricing queries
- [ ] Test with room comparison queries
- [ ] Verify performance (response time < 10s)
- [ ] Compare results with Exercise 0 (should be more accurate)

#### Phase 6: Optimization
- [ ] Tune chunk size and overlap if needed
- [ ] Optimize retrieval k parameter
- [ ] Add caching for frequent queries (optional)
- [ ] Document vector store persistence strategy

---

### Exercise 2: Booking Analytics with SQL Agent

#### Phase 1: Setup & Database Connection
- [ ] Start PostgreSQL database using `./start-app.sh --no_ai_agent`
- [ ] Install SQL dependencies (`langchain-community`, `psycopg2-binary`)
- [ ] Verify database connection (test connection string)
- [ ] Inspect database schema and understand table structure
- [ ] Load sample booking data to test queries

#### Phase 2: SQL Database Integration
- [ ] Create SQLDatabase instance from connection URI
- [ ] Test basic SQL queries manually (SELECT, COUNT, SUM)
- [ ] Verify database schema introspection works
- [ ] Test date filtering and aggregation queries

#### Phase 3: SQL Agent Implementation
- [ ] Create SQLDatabaseToolkit with database and LLM
- [ ] Implement create_sql_agent with proper system prompt
- [ ] Configure agent for hospitality context (hotel names, dates, metrics)
- [ ] Add custom system prompt explaining booking schema
- [ ] Test agent with simple queries (booking counts)

#### Phase 4: Analytics Calculations
- [ ] Implement bookings count query logic
- [ ] Implement occupancy rate calculation (two-step: query + formula)
- [ ] Implement total revenue aggregation
- [ ] Implement RevPAR calculation (revenue / available room-nights)
- [ ] Handle edge cases (no bookings, division by zero)

#### Phase 5: Two-Step Query Process
- [ ] Implement Step 1: Generate SQL from natural language
- [ ] Implement Step 2: Execute query and format results
- [ ] Add query validation before execution
- [ ] Implement result formatting (tables, markdown)
- [ ] Add error handling for SQL syntax errors

#### Phase 6: Advanced Queries & Testing
- [ ] Test with date range queries (months, quarters, years)
- [ ] Test with hotel-specific filters
- [ ] Test with guest country/city filters
- [ ] Test with meal plan comparisons
- [ ] Verify occupancy and RevPAR calculations are accurate
- [ ] Test with edge cases (empty results, invalid dates)

#### Phase 7: Integration & Error Handling
- [ ] Integrate SQL agent with WebSocket API
- [ ] Add comprehensive error handling (connection errors, query errors)
- [ ] Implement query timeout protection
- [ ] Add logging for debugging SQL generation
- [ ] Test end-to-end with WebSocket interface

#### Phase 8: Optimization & Documentation
- [ ] Optimize system prompt for better SQL generation
- [ ] Add query result caching for common queries (optional)
- [ ] Document SQL agent limitations and best practices
- [ ] Add code comments and docstrings

---

## 📊 Quick Summary

```
📌 Pending:  4
🔥 In progress: 0
✅ Completed: 0
🐛 Technical debt: 0
🎓 Workshop Exercises: 3 (Exercise 0, 1, 2)
```

> ⚠️ **Remember**: Update this file after each work session
