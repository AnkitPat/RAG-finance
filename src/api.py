from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from src.utils.vector_store import Ingestor
from src.agents.graph import run_financial_rag

app = FastAPI(title="Financial RAG API")

# Initialize dependencies
ingestor = Ingestor()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict]

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    # 1. Search for context
    docs = ingestor.search(request.query, k=3)
    
    # 2. Extract page content for context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 3. Run Agent Workflow
    answer = run_financial_rag(request.query, context)
    
    # 4. Format sources
    sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    
    return {
        "query": request.query,
        "answer": answer,
        "sources": sources
    }
