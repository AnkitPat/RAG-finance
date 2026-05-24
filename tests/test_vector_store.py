from unittest.mock import MagicMock, patch
from src.utils.vector_store import Ingestor

def test_ingest_text():
    # Setup - patch embeddings to avoid API key requirements and return a valid embedding vector
    mock_embeddings = MagicMock()
    # Return a 768-dimensional vector to match the model (simulated)
    mock_embeddings.embed_query.return_value = [0.1] * 768
    mock_embeddings.embed_documents.return_value = [[0.1] * 768]
    
    with patch('src.utils.vector_store.GoogleGenerativeAIEmbeddings', return_value=mock_embeddings):
        ingestor = Ingestor(collection_name="test_collection")
        text = "# Financial Results\nRevenue was $10.5 million."
        ingestor.ingest_text(text, metadata={"source": "test.pdf"})
        
        results = ingestor.search("How much was the revenue?")
        assert len(results) > 0
        assert "$10.5 million" in results[0].page_content
