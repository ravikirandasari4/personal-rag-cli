import os
import PyPDF2
from docx import Document
from typing import List, Dict

"""
DocumentParser: A class to handle parsing of various document formats (TXT, PDF, DOCX).
This class is responsible for reading documents from disk, extracting their content, and returning structured data
that includes the text content along with metadata such as filename, file path, and file type.
"""
class DocumentParser:
    """DocumentParser: A class to handle parsing of various document formats (TXT, PDF, DOCX)."""
    def __init__(self):
        self.supported_formats = {'.txt', '.pdf', '.docx'}
    
    """Parse plain text files."""
    def parse_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    """Parse PDF files."""
    def parse_pdf(self, file_path: str) -> str:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    """Parse Word documents."""
    def parse_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    """Parse a single document and return metadata."""
    def parse_document(self, file_path: str) -> Dict[str, str]:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        if file_ext == '.txt':
            content = self.parse_txt(file_path)
        elif file_ext == '.pdf':
            content = self.parse_pdf(file_path)
        elif file_ext == '.docx':
            content = self.parse_docx(file_path)
        
        return {
            'content': content,
            'filename': os.path.basename(file_path),
            'file_path': file_path,
            'file_type': file_ext
        }
    
    """Parse all supported documents in a directory."""
    def parse_directory(self, directory_path: str) -> List[Dict[str, str]]:
        documents = []
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                try:
                    doc = self.parse_document(file_path)
                    documents.append(doc)
                    print(f"Parsed: {filename}")
                except ValueError as e:
                    print(f"Skipping {filename}: {e}")
        
        return documents