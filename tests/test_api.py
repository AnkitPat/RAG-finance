from fastapi.testclient import TestClient
from src.api import app
from unittest.mock import patch

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

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

def test_ingestion_endpoints():
    with patch("src.api.IngestionManager.scan_and_ingest") as mock_scan:
        with patch("src.api.IngestionManager.get_ingested_files") as mock_get:
            mock_scan.return_value = ["doc1.pdf"]
            mock_get.return_value = ["doc1.pdf", "doc2.pdf"]
            
            # Test trigger ingestion
            response_ingest = client.post("/ingest")
            assert response_ingest.status_code == 200
            assert response_ingest.json()["count"] == 1
            
            # Test status
            response_status = client.get("/ingest/status")
            assert response_status.status_code == 200
            assert response_status.json()["count"] == 2
            assert "doc1.pdf" in response_status.json()["ingested_files"]

