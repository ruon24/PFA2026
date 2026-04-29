"""
PDF Parser - Extract text from PDF files
"""

import os
import pdfplumber
from typing import List, Optional


class PDFParser:
    """Extract text from PDF files using pdfplumber"""
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract all text from a PDF file"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        return text
    
    def extract_text_by_page(self, pdf_path: str) -> List[str]:
        """Extract text from each page separately"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
        
        return pages
    
    def get_page_count(self, pdf_path: str) -> int:
        """Get the number of pages in a PDF"""
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    
    def is_valid_pdf(self, file_path: str) -> bool:
        """Check if file is a valid PDF"""
        if not os.path.exists(file_path):
            return False
        _, ext = os.path.splitext(file_path)
        return ext.lower() in self.supported_extensions