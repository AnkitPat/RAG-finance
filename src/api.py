from fastapi import FastAPI

app = FastAPI(title="Financial RAG API")

@app.get("/health")
async def health_check():
    return {"status": "ok"}
