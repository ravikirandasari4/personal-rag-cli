import os
import pypdf  # Modern PDF library
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
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            with open(file_path, 'r', encoding='iso-8859-1') as file:
                return file.read()
    
    """Parse PDF files with better error handling."""
    def parse_pdf(self, file_path: str) -> str:
        text = ""
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    print(f"Warning: {os.path.basename(file_path)} is encrypted/password-protected")
                    # Try with empty password (some PDFs have empty password)
                    try:
                        pdf_reader.decrypt("")
                    except:
                        print(f"Cannot decrypt {os.path.basename(file_path)} - skipping")
                        return f"[Encrypted PDF - cannot read: {os.path.basename(file_path)}]"
                
                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        print(f"Error reading page {page_num + 1} of {os.path.basename(file_path)}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error parsing PDF {file_path}: {e}")
            return f"[PDF parsing failed for {os.path.basename(file_path)}]"
        
        return text
    
    """Parse Word documents."""
    def parse_docx(self, file_path: str) -> str:
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error parsing DOCX {file_path}: {e}")
            return f"[DOCX parsing failed for {os.path.basename(file_path)}]"
    
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
                    if doc['content'].strip():  # Only add if content is not empty
                        documents.append(doc)
                        print(f"✓ Successfully parsed: {filename}")
                    else:
                        print(f"⚠ Warning: {filename} appears to be empty or unreadable")
                except ValueError as e:
                    print(f"⚠ Skipping {filename}: {e}")
                except Exception as e:
                    print(f"✗ Error parsing {filename}: {e}")
        
        return documents