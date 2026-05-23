# AI Financial RAG

A Financial RAG service using LangGraph, OpenAI, and ChromaDB.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure environment:
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   ```

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

## Running Tests

To run the full test suite:
```bash
PYTHONPATH=. pytest
```
