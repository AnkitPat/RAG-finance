# Financial RAG Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a RAG service for financial PDFs that uses an "Agentic Auditor" loop to ensure numeric accuracy.

**Architecture:** A two-stage pipeline: (1) PDF-to-Markdown ingestion with hybrid retrieval, and (2) An agentic generation loop where an Extractor drafts an answer and an Auditor verifies every numeric amount against the source.

**Tech Stack:** Python, LangGraph, ChromaDB, OpenAI (Embeddings & GPT-4o), Marker (PDF Parsing).

---

### Task 1: Project Initialization & Environment Setup

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `README.md`

- [ ] **Step 1: Create requirements.txt with core dependencies**

```text
langchain
langchain-openai
langgraph
chromadb
marker-pdf
python-dotenv
pytest
```

- [ ] **Step 2: Create .env.example**

```text
OPENAI_API_KEY=your_key_here
```

- [ ] **Step 3: Initialize project structure**

Run: `mkdir -p data/pdfs data/processed src/agents src/utils tests`
Expected: Directories created.

- [ ] **Step 4: Commit initialization**

```bash
git add requirements.txt .env.example README.md
git commit -m "chore: initialize project structure and dependencies"
```

---

### Task 2: PDF Parsing and Markdown Extraction

**Files:**
- Create: `src/utils/parser.py`
- Test: `tests/test_parser.py`

- [ ] **Step 1: Write test for PDF to Markdown conversion**

```python
import os
from src.utils.parser import parse_pdf_to_markdown

def test_parse_pdf_to_markdown():
    # Assume a dummy PDF exists for testing or mock the marker output
    # For now, we will verify the interface
    path = "tests/data/sample.pdf"
    # Ensure dir exists
    os.makedirs("tests/data", exist_ok=True)
    # Create empty pdf if doesn't exist (mocking)
    if not os.path.exists(path):
        with open(path, "wb") as f: f.write(b"%PDF-1.1")
    
    result = parse_pdf_to_markdown(path)
    assert isinstance(result, str)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_parser.py`
Expected: FAIL (ImportError or NameError)

- [ ] **Step 3: Implement parse_pdf_to_markdown using Marker**

```python
from marker.convert import convert_single_pdf
from marker.models import load_all_models

def parse_pdf_to_markdown(pdf_path: str) -> str:
    model_lst = load_all_models()
    full_text, images, out_meta = convert_single_pdf(pdf_path, model_lst)
    return full_text
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_parser.py`
Expected: PASS (Note: Marker might need models downloaded, which takes time)

- [ ] **Step 5: Commit parser**

```bash
git add src/utils/parser.py tests/test_parser.py
git commit -m "feat: implement PDF to Markdown parser using Marker"
```

---

### Task 3: Ingestion and Vector Storage

**Files:**
- Create: `src/utils/vector_store.py`
- Test: `tests/test_vector_store.py`

- [ ] **Step 1: Write test for document ingestion**

```python
from src.utils.vector_store import Ingestor

def test_ingest_text():
    ingestor = Ingestor(collection_name="test_collection")
    text = "# Financial Results\nRevenue was $10.5 million."
    ingestor.ingest_text(text, metadata={"source": "test.pdf"})
    
    results = ingestor.search("How much was the revenue?")
    assert len(results) > 0
    assert "$10.5 million" in results[0].page_content
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_vector_store.py`
Expected: FAIL

- [ ] **Step 3: Implement Ingestor with ChromaDB**

```python
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Ingestor:
    def __init__(self, collection_name="financial_docs"):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", " "]
        )

    def ingest_text(self, text, metadata):
        chunks = self.splitter.create_documents([text], metadatas=[metadata])
        self.vectorstore.add_documents(chunks)

    def search(self, query, k=3):
        return self.vectorstore.similarity_search(query, k=k)
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_vector_store.py`
Expected: PASS

- [ ] **Step 5: Commit ingestion**

```bash
git add src/utils/vector_store.py tests/test_vector_store.py
git commit -m "feat: implement vector store ingestor with ChromaDB"
```

---

### Task 4: The Extractor Agent

**Files:**
- Create: `src/agents/extractor.py`
- Test: `tests/test_extractor.py`

- [ ] **Step 1: Write test for Extractor**

```python
from src.agents.extractor import ExtractorAgent

def test_extractor_draft():
    agent = ExtractorAgent()
    context = "Revenue in Q1 was $500M."
    query = "What was the Q1 revenue?"
    draft = agent.generate_draft(query, context)
    assert "$500M" in draft
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_extractor.py`
Expected: FAIL

- [ ] **Step 3: Implement ExtractorAgent**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class ExtractorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.prompt = ChatPromptTemplate.from_template("""
        You are a financial data extractor. Use the context below to answer the query.
        Be precise with amounts.
        
        Context: {context}
        Query: {query}
        Answer:""")

    def generate_draft(self, query, context):
        chain = self.prompt | self.llm
        response = chain.invoke({"query": query, "context": context})
        return response.content
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_extractor.py`
Expected: PASS

- [ ] **Step 5: Commit extractor**

```bash
git add src/agents/extractor.py tests/test_extractor.py
git commit -m "feat: implement Extractor agent"
```

---

### Task 5: The Auditor Agent (Verification Loop)

**Files:**
- Create: `src/agents/auditor.py`
- Test: `tests/test_auditor.py`

- [ ] **Step 1: Write test for Auditor (Successful Case)**

```python
from src.agents.auditor import AuditorAgent

def test_auditor_success():
    agent = AuditorAgent()
    draft = "The revenue was $500M."
    context = "In Q1, revenue was $500M."
    is_valid, feedback = agent.verify(draft, context)
    assert is_valid is True

def test_auditor_failure():
    agent = AuditorAgent()
    draft = "The revenue was $500B." # Hallucination (B instead of M)
    context = "In Q1, revenue was $500M."
    is_valid, feedback = agent.verify(draft, context)
    assert is_valid is False
    assert "500B" in feedback
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_auditor.py`
Expected: FAIL

- [ ] **Step 3: Implement AuditorAgent**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

class AuditorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.prompt = ChatPromptTemplate.from_template("""
        You are a financial auditor. Verify the Draft Answer against the Source Context.
        Focus ONLY on numeric amounts and their scales (M vs B vs K).
        
        Draft Answer: {draft}
        Source Context: {context}
        
        Respond in JSON format:
        {{
            "is_valid": true/false,
            "discrepancies": ["list of issues found"]
        }}
        """)

    def verify(self, draft, context):
        chain = self.prompt | self.llm
        response = chain.invoke({"draft": draft, "context": context})
        data = json.loads(response.content)
        return data["is_valid"], ", ".join(data["discrepancies"])
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_auditor.py`
Expected: PASS

- [ ] **Step 5: Commit auditor**

```bash
git add src/agents/auditor.py tests/test_auditor.py
git commit -m "feat: implement Auditor agent with JSON verification loop"
```

---

### Task 6: Orchestrating the Workflow with LangGraph

**Files:**
- Create: `src/agents/graph.py`
- Test: `tests/test_graph.py`

- [ ] **Step 1: Write test for orchestrated graph**

```python
from src.agents.graph import run_financial_rag

def test_graph_end_to_end():
    # Mock search result for testing
    query = "What is the revenue?"
    context = "Annual revenue was $1.2 billion."
    # This should return the verified answer
    result = run_financial_rag(query, context)
    assert "$1.2 billion" in result
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_graph.py`
Expected: FAIL

- [ ] **Step 3: Implement LangGraph Workflow**

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from src.agents.extractor import ExtractorAgent
from src.agents.auditor import AuditorAgent

class GraphState(TypedDict):
    query: str
    context: str
    draft: str
    is_valid: bool
    feedback: str
    final_answer: str
    attempts: int

def extractor_node(state: GraphState):
    agent = ExtractorAgent()
    prompt_query = state['query']
    if state.get('feedback'):
        prompt_query += f"\n\nPrevious draft was incorrect. Feedback: {state['feedback']}. Fix the amounts."
    
    draft = agent.generate_draft(prompt_query, state['context'])
    return {"draft": draft, "attempts": state.get('attempts', 0) + 1}

def auditor_node(state: GraphState):
    agent = AuditorAgent()
    is_valid, feedback = agent.verify(state['draft'], state['context'])
    return {"is_valid": is_valid, "feedback": feedback, "final_answer": state['draft'] if is_valid else ""}

def should_continue(state: GraphState):
    if state['is_valid'] or state['attempts'] >= 2:
        return "end"
    return "extractor"

def create_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("auditor", auditor_node)
    
    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "auditor")
    workflow.add_conditional_edges("auditor", should_continue, {"extractor": "extractor", "end": END})
    
    return workflow.compile()

def run_financial_rag(query, context):
    graph = create_graph()
    initial_state = {"query": query, "context": context, "attempts": 0}
    result = graph.invoke(initial_state)
    return result.get('final_answer') or result.get('draft')
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_graph.py`
Expected: PASS

- [ ] **Step 5: Commit orchestration**

```bash
git add src/agents/graph.py tests/test_graph.py
git commit -m "feat: orchestrate agents using LangGraph"
```
