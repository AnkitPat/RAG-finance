from marker.models import create_model_dict
from marker.converters.pdf import PdfConverter

def parse_pdf_to_markdown(pdf_path: str) -> str:
    """
    Converts a PDF file to Markdown using the marker-pdf library.
    
    Args:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        str: Extracted text in Markdown format.
    """
    # Note: In a production app, model_dict should be loaded once and reused.
    model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict)
    rendered = converter(pdf_path)
    return rendered.markdown
