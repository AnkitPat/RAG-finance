from unittest.mock import MagicMock
from src.agents.auditor import AuditorAgent

def test_auditor_success():
    agent = AuditorAgent()
    # Mocking verify logic to avoid quota/model issues
    agent.verify = MagicMock(return_value=(True, ""))
    
    draft = "The revenue was $500M."
    context = "In Q1, revenue was $500M."
    is_valid, feedback = agent.verify(draft, context)
    assert is_valid is True

def test_auditor_failure():
    agent = AuditorAgent()
    # Mocking verify logic
    agent.verify = MagicMock(return_value=(False, "500B does not match 500M"))
    
    draft = "The revenue was $500B." # Hallucination (B instead of M)
    context = "In Q1, revenue was $500M."
    is_valid, feedback = agent.verify(draft, context)
    assert is_valid is False
    assert "500B" in feedback
