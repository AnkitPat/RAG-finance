import os
from src.utils.vector_store import Ingestor
from src.utils.parser import parse_pdf_to_markdown

class IngestionManager:
    def __init__(self, vectorstore=None, docs_dir="documents"):
        self.ingestor = Ingestor()
        self.vectorstore = vectorstore or self.ingestor.vectorstore
        self.docs_dir = docs_dir

    def get_ingested_files(self):
        data = self.vectorstore.get(include=["metadatas"])
        if not data or not data["metadatas"]:
            return set()
        return {m["source"] for m in data["metadatas"] if "source" in m}

    def scan_and_ingest(self):
        ingested = self.get_ingested_files()
        all_files = [f for f in os.listdir(self.docs_dir) if f.endswith(".pdf")]
        
        new_files = [f for f in all_files if f not in ingested]
        
        results = []
        for filename in new_files:
            path = os.path.join(self.docs_dir, filename)
            print(f"Ingesting: {filename}")
            text = parse_pdf_to_markdown(path)
            self.ingestor.ingest_text(text, metadata={"source": filename})
            results.append(filename)
            
        return results
