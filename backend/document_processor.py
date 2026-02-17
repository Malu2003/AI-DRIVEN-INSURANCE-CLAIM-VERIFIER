"""
Document Text Extraction
========================

Extracts text from clinical documents (.pdf, .docx, .txt).
Used by the backend to process uploaded clinical documents.
"""

import os
from pathlib import Path


def extract_text_from_document(file_path: str) -> str:
    """
    Extract text from a document file.
    
    Supports: .txt, .docx, .pdf
    
    Args:
        file_path: Path to document file
        
    Returns:
        Extracted text string
        
    Raises:
        ValueError: If file type not supported or file doesn't exist
        Exception: If extraction fails
    """
    
    if not os.path.exists(file_path):
        raise ValueError(f"Document file not found: {file_path}")
    
    file_ext = Path(file_path).suffix.lower()
    
    # TXT files - simple text extraction
    if file_ext == '.txt':
        return _extract_text_from_txt(file_path)
    
    # DOCX files - Word documents
    elif file_ext == '.docx':
        return _extract_text_from_docx(file_path)
    
    # PDF files
    elif file_ext == '.pdf':
        return _extract_text_from_pdf(file_path)
    
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Supported: .txt, .docx, .pdf")


def _extract_text_from_txt(file_path: str) -> str:
    """Extract text from plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text.strip()
    except Exception as e:
        raise Exception(f"Failed to read text file: {str(e)}")


def _extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from DOCX file.
    
    Requires: python-docx package
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required for DOCX support. Install with: pip install python-docx")
    
    try:
        doc = Document(file_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        raise Exception(f"Failed to extract text from DOCX: {str(e)}")


def _extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF file.
    
    Requires: PyPDF2 package (fallback to simple extraction)
    """
    try:
        import PyPDF2
        
        text = ""
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except:
            # Fallback if PyPDF2 fails
            text = _extract_pdf_fallback(file_path)
        
        return text.strip()
    
    except ImportError:
        # If PyPDF2 not available, try fallback
        return _extract_pdf_fallback(file_path)


def _extract_pdf_fallback(file_path: str) -> str:
    """
    Fallback PDF extraction using pdfplumber (if available).
    Falls back to error message if neither package available.
    """
    try:
        import pdfplumber
        
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    
    except ImportError:
        # Neither PyPDF2 nor pdfplumber available
        raise ImportError(
            "PDF extraction requires PyPDF2 or pdfplumber. "
            "Install with: pip install PyPDF2 or pip install pdfplumber"
        )
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")
