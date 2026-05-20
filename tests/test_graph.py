from unittest.mock import MagicMock, patch
from src.agents.graph import run_financial_rag

def test_graph_end_to_end():
    query = "What is the revenue?"
    context = "Annual revenue was $1.2 billion."
    
    # Mock agents to avoid API calls
    with patch('src.agents.graph.ExtractorAgent') as MockExtractor:
        with patch('src.agents.graph.AuditorAgent') as MockAuditor:
            mock_extractor = MockExtractor.return_value
            mock_extractor.generate_draft.return_value = "The annual revenue was $1.2 billion."
            
            mock_auditor = MockAuditor.return_value
            mock_auditor.verify.return_value = (True, "")
            
            result = run_financial_rag(query, context)
            assert "$1.2 billion" in result
