# Code Implementation Steps: From Prototype to Production

Building this system involved four distinct architectural phases, prioritizing security and modularity from day one.

## Phase 1: Core Agentic Engine (LangGraph)
We established an agentic workflow using **LangGraph**, decoupling data extraction from verification.

### The Verification Loop
The core logic resides in a state-based graph where the Auditor agent decides whether to terminate or retry the extraction:

```python
# The workflow forces a retry if the Auditor finds an error
def should_continue(state: GraphState):
    if state.get('is_valid', False) or state.get('attempts', 0) >= 2:
        return "end"
    return "extractor"

def auditor_node(state: GraphState):
    agent = AuditorAgent()
    is_valid, feedback = agent.verify(state['draft'], state['context'])
    return {"is_valid": is_valid, "feedback": feedback, "final_answer": state['draft'] if is_valid else ""}
```

## Phase 2: Intelligent Document Ingestion
To handle complex financial PDFs, we moved beyond basic text splitting.

### Automated Incremental Ingestion
The `IngestionManager` scans the `documents/` folder and compares filenames against existing documents in ChromaDB.

```python
def scan_and_ingest(self):
    ingested = self.get_ingested_files() # Queries ChromaDB metadata
    all_files = [f for f in os.listdir(self.docs_dir) if f.endswith(".pdf")]
    
    new_files = [f for f in all_files if f not in ingested]
    
    for filename in new_files:
        path = os.path.join(self.docs_dir, filename)
        text = parse_pdf_to_markdown(path) # Uses marker-pdf
        self.ingestor.ingest_text(text, metadata={"source": filename})
```

## Phase 3: Web Layer & Secure Configuration
We bridged the agentic core with a RESTful interface, utilizing FastAPI's `lifespan` handler to automate startup routines.

### FastAPI Lifespan & Query Endpoint
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run ingestion on startup automatically
    manager = IngestionManager()
    manager.scan_and_ingest()
    yield

app = FastAPI(title="Financial RAG API", lifespan=lifespan)

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    docs = ingestor.search(request.query, k=10)
    context = "

".join([doc.page_content for doc in docs])
    answer = run_financial_rag(request.query, context)
    return {"query": request.query, "answer": answer, "sources": [...]}
```

## Phase 4: Verification & Auditing
The `AuditorAgent` prompt is designed to strictly enforce structural output, ensuring the graph can parse the result.

### The Auditor Prompt
```python
self.prompt = ChatPromptTemplate.from_template("""
    You are a financial auditor. Verify the Draft Answer against the Source Context.
    Respond in JSON format:
    {{
        "is_valid": true/false,
        "discrepancies": ["list of issues found"]
    }}
""")
```

This structure enables the system to treat accuracy as a workflow, turning LLM outputs from probabilistic guesses into verifiable financial data.
