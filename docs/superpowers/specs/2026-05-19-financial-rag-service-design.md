# Design Specification: Financial RAG Service (Agentic Auditor)

**Date:** 2026-05-19
**Status:** Draft
**Topic:** RAG service for financial PDFs with high numeric accuracy.

## 1. Overview
The goal is to build a RAG (Retrieval-Augmented Generation) service that processes financial PDF documents (specifically unstructured ones like annual reports) and provides highly accurate answers, with a strict focus on the correctness of financial amounts and figures.

## 2. Requirements
- **High Fidelity Extraction:** Must accurately extract text and structure from unstructured PDFs.
- **Numeric Precision:** Financial amounts in answers must be verified against source data.
- **Scalability:** Must be capable of being exposed as an API.
- **Explainability:** Answers should ideally include citations or references to source chunks.

## 3. Architecture (Agentic Auditor)

### 3.1 Data Ingestion Pipeline
1.  **PDF Parsing:** Use `Marker` or `LlamaParse` to convert PDF files into Markdown. Markdown is preferred over raw text to preserve semantic structure (headers, tables, emphasis).
2.  **Chunking Strategy:**
    *   Tool: `RecursiveCharacterTextSplitter`.
    *   Markers: Split based on Markdown headers (`#`, `##`, etc.) to keep related sections together.
    *   Chunk Size: ~1000 tokens with 10% overlap (to be tuned).
3.  **Embedding:** `text-embedding-3-small` (OpenAI) or equivalent high-density model.
4.  **Vector Storage:** `ChromaDB` (local-first for development) or `Qdrant` (if scalability is prioritized).

### 3.2 Retrieval Strategy
- **Hybrid Search:** Combine Vector Search (semantic) with BM25 (keyword). This is critical for finding specific numbers or financial terms that might not have a strong semantic signature but are unique.
- **Context Window:** Retrieve top-K chunks (e.g., K=5) and potentially their adjacent chunks (parent/child) to provide full context to the agents.

### 3.3 Generation & Verification Loop
A two-agent "Auditor" workflow:

1.  **Extractor Agent:**
    *   **Input:** User query + Retrieved chunks.
    *   **Task:** Synthesize a clear, direct answer. Include citations if possible.
    *   **Output:** Draft Answer.
2.  **Auditor Agent:**
    *   **Input:** Draft Answer + Raw Retrieved chunks.
    *   **Task:** Identify every numeric figure in the Draft Answer. Verify each figure against the provided source chunks.
    *   **Rules:**
        *   If the figure matches the source, approve.
        *   If the figure is missing from the source or incorrect (e.g., mismatched scale like Millions vs Billions), flag it.
3.  **Verification Result:**
    *   If all figures verified: Return Answer to user.
    *   If discrepancies found: Send corrective feedback back to the Extractor Agent for one regeneration attempt.

## 4. Technology Stack (Recommended)
- **Language:** Python
- **Framework:** LangGraph (for agentic workflow) or LangChain.
- **Parser:** `Marker` (open-source) or `LlamaParse`.
- **Vector DB:** `ChromaDB`.
- **LLM:** OpenAI GPT-4o or Gemini 1.5 Pro (preferred for large context and reasoning).

## 5. Success Criteria
- [ ] Successful extraction of text from test 10-K PDFs.
- [ ] Verification loop correctly identifies a "hallucinated" number in a test case.
- [ ] End-to-end RAG query returns correct numeric values with >99% accuracy in controlled tests.
