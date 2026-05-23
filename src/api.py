from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv
import os

from src.utils.vector_store import Ingestor
from src.agents.graph import run_financial_rag
from src.utils.ingestion_manager import IngestionManager

load_dotenv()

# Initialize dependencies at module level
ingestor = Ingestor()
ingestion_manager = IngestionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run ingestion on startup
    print("Starting automated ingestion...")
    try:
        newly_ingested = ingestion_manager.scan_and_ingest()
        print(f"Ingested {len(newly_ingested)} new files: {newly_ingested}")
    except Exception as e:
        print(f"Ingestion failed: {e}")
    yield

# Update app initialization
app = FastAPI(title="Financial RAG API", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict]

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/ingest")
async def trigger_ingestion():
    try:
        newly_ingested = ingestion_manager.scan_and_ingest()
        return {
            "status": "success",
            "newly_ingested": newly_ingested,
            "count": len(newly_ingested)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.get("/ingest/status")
async def get_ingestion_status():
    try:
        ingested = list(ingestion_manager.get_ingested_files())
        return {
            "ingested_files": ingested,
            "count": len(ingested)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch ingestion status: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
