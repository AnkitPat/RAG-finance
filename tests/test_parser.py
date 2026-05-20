import os
import pytest
from unittest.mock import MagicMock, patch
from src.utils.parser import parse_pdf_to_markdown

def test_parse_pdf_to_markdown_mocked():
    # Setup test data
    path = "tests/data/sample.pdf"
    os.makedirs("tests/data", exist_ok=True)
    
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(b"%PDF-1.1\n%%EOF")
    
    # Mocking marker components
    with patch("src.utils.parser.create_model_dict") as mock_create_model_dict, \
         patch("src.utils.parser.PdfConverter") as mock_pdf_converter:
        
        # Mock model dict
        mock_create_model_dict.return_value = {"mock_model": "data"}
        
        # Mock converter instance
        mock_instance = MagicMock()
        mock_pdf_converter.return_value = mock_instance
        
        # Mock conversion result
        mock_rendered = MagicMock()
        mock_rendered.markdown = "# Mocked Markdown Content"
        mock_instance.return_value = mock_rendered
        
        # Execute
        result = parse_pdf_to_markdown(path)
        
        # Assertions
        assert result == "# Mocked Markdown Content"
        mock_create_model_dict.assert_called_once()
        mock_pdf_converter.assert_called_once_with(artifact_dict={"mock_model": "data"})
        mock_instance.assert_called_once_with(path)

@pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") != "1", reason="Slow test that loads ML models")
def test_parse_pdf_to_markdown_original():
    # Setup test data
    path = "tests/data/sample.pdf"
    os.makedirs("tests/data", exist_ok=True)
    
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(b"%PDF-1.1\n%%EOF")
            
    # This will load real models and might fail/timeout in CI/restricted environments
    result = parse_pdf_to_markdown(path)
    assert isinstance(result, str)
