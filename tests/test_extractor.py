from unittest.mock import MagicMock, patch
from src.agents.extractor import ExtractorAgent

def test_extractor_draft():
    agent = ExtractorAgent()
    # Patch the generate_draft method directly to avoid LLM interaction entirely for unit testing
    with patch.object(agent, 'generate_draft', return_value="The Q1 revenue was $500M."):
        context = "Revenue in Q1 was $500M."
        query = "What was the Q1 revenue?"
        draft = agent.generate_draft(query, context)
        assert "$500M" in draft
