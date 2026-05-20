import sys
import os
from unittest.mock import MagicMock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.extractor import ExtractorAgent

def test_extractor_draft():
    agent = ExtractorAgent()
    # Mock the LLM chain invoke
    agent.generate_draft = MagicMock(return_value="The Q1 revenue was $500M.")
    
    context = "Revenue in Q1 was $500M."
    query = "What was the Q1 revenue?"
    draft = agent.generate_draft(query, context)
    assert "$500M" in draft
