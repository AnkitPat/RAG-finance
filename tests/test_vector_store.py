from src.utils.vector_store import Ingestor

def test_ingest_text():
    # Setup - use an in-memory collection if possible, or a dedicated test one
    ingestor = Ingestor(collection_name="test_collection")
    text = "# Financial Results\nRevenue was $10.5 million."
    ingestor.ingest_text(text, metadata={"source": "test.pdf"})
    
    results = ingestor.search("How much was the revenue?")
    assert len(results) > 0
    assert "$10.5 million" in results[0].page_content
