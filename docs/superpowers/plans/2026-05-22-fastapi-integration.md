# FastAPI Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a FastAPI web service that provides a RAG query endpoint using existing vector store and LangGraph agents.

**Architecture:** A single-file FastAPI application (`src/api.py`) that initializes the `Ingestor` for context retrieval and invokes the `run_financial_rag` workflow.

**Tech Stack:** FastAPI, Uvicorn, Pydantic, pytest, httpx.

---

### Task 1: Environment Setup

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependencies to requirements.txt**

```text
fastapi
uvicorn
httpx
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: Success

- [ ] **Step 3: Commit changes**

```bash
git add requirements.txt
git commit -m "chore: add fastapi and uvicorn dependencies"
```

---

### Task 2: API Skeleton and Health Check

**Files:**
- Create: `src/api.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing health check test**

```python
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_api.py::test_health_check`
Expected: FAIL (ModuleNotFoundError or 404)

- [ ] **Step 3: Implement health check endpoint**

```python
from fastapi import FastAPI

app = FastAPI(title="Financial RAG API")

@app.get("/health")
async def health_check():
    return {"status": "ok"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_api.py::test_health_check`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api.py tests/test_api.py
git commit -m "feat: add api skeleton and health check"
```

---

### Task 3: Query Endpoint Implementation

**Files:**
- Modify: `src/api.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing query test (with mocking)**

```python
from unittest.mock import patch

def test_query_endpoint():
    with patch("src.api.ingestor.search") as mock_search:
        with patch("src.api.run_financial_rag") as mock_run:
            mock_search.return_value = [
                type('obj', (object,), {'page_content': 'Revenue was $100', 'metadata': {'source': 'doc1'}})
            ]
            mock_run.return_value = "The revenue was $100."
            
            response = client.post("/query", json={"query": "What is the revenue?"})
            
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "What is the revenue?"
            assert data["answer"] == "The revenue was $100."
            assert len(data["sources"]) == 1
            assert data["sources"][0]["content"] == "Revenue was $100"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_api.py::test_query_endpoint`
Expected: FAIL (404 Not Found)

- [ ] **Step 3: Implement Query Models and Endpoint**

```python
from pydantic import BaseModel
from typing import List, Dict
from src.utils.vector_store import Ingestor
from src.agents.graph import run_financial_rag

# Initialize dependencies
ingestor = Ingestor()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict]

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    # 1. Search for context
    docs = ingestor.search(request.query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. Run Agent Workflow
    answer = run_financial_rag(request.query, context)
    
    # 3. Format sources
    sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    
    return {
        "query": request.query,
        "answer": answer,
        "sources": sources
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_api.py::test_query_endpoint`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api.py tests/test_api.py
git commit -m "feat: implement /query endpoint"
```

---

### Task 4: Final Verification and Documentation

- [ ] **Step 1: Run all tests to ensure no regressions**

Run: `PYTHONPATH=. pytest`
Expected: ALL PASS

- [ ] **Step 2: Update README.md with API usage**

Add section to `README.md`:
```markdown
## API Usage

Start the server:
```bash
PYTHONPATH=. uvicorn src.api:app --reload
```

Query the API:
```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the revenue?"}'
```
```

- [ ] **Step 3: Final Commit**

```bash
git add README.md
git commit -m "docs: add api usage instructions to readme"
```
