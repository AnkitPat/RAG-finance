import os
from unittest.mock import MagicMock, patch
from src.utils.ingestion_manager import IngestionManager

def test_get_ingested_files():
    mock_vectorstore = MagicMock()
    mock_vectorstore.get.return_value = {
        "metadatas": [{"source": "doc1.pdf"}, {"source": "doc2.pdf"}]
    }
    
    manager = IngestionManager(vectorstore=mock_vectorstore)
    ingested = manager.get_ingested_files()
    
    assert "doc1.pdf" in ingested
    assert "doc2.pdf" in ingested

def test_scan_and_ingest():
    mock_vectorstore = MagicMock()
    mock_vectorstore.get.return_value = {
        "metadatas": [{"source": "doc1.pdf"}]
    }
    
    with patch("os.listdir") as mock_listdir, \
         patch("src.utils.ingestion_manager.parse_pdf_to_markdown") as mock_parse, \
         patch("src.utils.ingestion_manager.Ingestor") as mock_ingestor_class:
        
        mock_listdir.return_value = ["doc1.pdf", "doc2.pdf", "other.txt"]
        mock_parse.return_value = "mocked markdown content"
        
        mock_ingestor_instance = mock_ingestor_class.return_value
        mock_ingestor_instance.vectorstore = mock_vectorstore
        
        manager = IngestionManager(vectorstore=mock_vectorstore)
        results = manager.scan_and_ingest()
        
        assert results == ["doc2.pdf"]
        mock_parse.assert_called_once_with("documents/doc2.pdf")
        mock_ingestor_instance.ingest_text.assert_called_once_with(
            "mocked markdown content", metadata={"source": "doc2.pdf"}
        )
