from src.agents.graph import run_financial_rag
from unittest.mock import MagicMock

def test_graph_end_to_end():
    # Mock search result for testing
    query = "What is the revenue?"
    context = "Annual revenue was $1.2 billion."
    
    # Mocking agents to avoid API calls
    with MagicMock() as mock_extractor, MagicMock() as mock_auditor:
        # Mock ExtractorAgent.generate_draft
        mock_extractor.generate_draft.return_value = "The annual revenue is $1.2 billion."
        
        # Mock AuditorAgent.verify
        mock_auditor.verify.return_value = (True, "Valid")
        
        # Patching agents in the graph
        import src.agents.graph
        src.agents.graph.ExtractorAgent = MagicMock(return_value=mock_extractor)
        src.agents.graph.AuditorAgent = MagicMock(return_value=mock_auditor)
        
        # This should return the verified answer
        result = run_financial_rag(query, context)
        assert "$1.2 billion" in result
