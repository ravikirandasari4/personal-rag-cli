# Personal RAG System - Complete Step-by-Step Tutorial

## Overview
We'll build a complete RAG (Retrieval-Augmented Generation) system that can process your documents, create embeddings, store them in a vector database, and answer queries using a local LLM.

## System Architecture
```
Documents → Parser → Chunker → Embedder → Vector DB → Retriever → LLM → Answer
```

## Prerequisites Setup

### Step 1: Install Required Software

#### 1.1 Install Ollama (Local LLM)
```bash
# Download and install Ollama for macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from: https://ollama.ai/download
```

#### 1.2 Pull a Language Model
```bash
# Install a lightweight but capable model
ollama pull llama3.2:3b

# Alternative options:
# ollama pull mistral:7b
# ollama pull codellama:7b
```

#### 1.3 Create Python Environment
```bash
# Create virtual environment
python3 -m venv rag_env
source rag_env/bin/activate

# Install required packages
pip install sentence-transformers faiss-cpu PyPDF2 python-docx numpy pandas ollama
```

### Step 2: Project Structure
Create the following directory structure:
```
personal_rag/
├── documents/          # Your input documents
├── data/              # Processed data and embeddings
├── src/
│   ├── document_parser.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── rag_system.py
└── main.py
```

## Implementation

### Step 3: Document Parser (`src/document_parser.py`)

```python
import os
import PyPDF2
from docx import Document
from typing import List, Dict

class DocumentParser:
    def __init__(self):
        self.supported_formats = {'.txt', '.pdf', '.docx'}
    
    def parse_txt(self, file_path: str) -> str:
        """Parse plain text files."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def parse_pdf(self, file_path: str) -> str:
        """Parse PDF files."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def parse_docx(self, file_path: str) -> str:
        """Parse Word documents."""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def parse_document(self, file_path: str) -> Dict[str, str]:
        """Parse a single document and return metadata."""
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
    
    def parse_directory(self, directory_path: str) -> List[Dict[str, str]]:
        """Parse all supported documents in a directory."""
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
```

### Step 4: Text Chunking and Embedding (`src/embedder.py`)

```python
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

class TextEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with a lightweight sentence transformer model."""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
        
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts."""
        print(f"Creating embeddings for {len(texts)} text chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def process_documents(self, documents: List[Dict[str, str]]) -> Dict:
        """Process documents into chunks with embeddings and metadata."""
        all_chunks = []
        all_metadata = []
        
        for doc in documents:
            chunks = self.chunk_text(doc['content'])
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'filename': doc['filename'],
                    'file_path': doc['file_path'],
                    'file_type': doc['file_type'],
                    'chunk_id': i,
                    'content': chunk
                })
        
        embeddings = self.create_embeddings(all_chunks)
        
        return {
            'chunks': all_chunks,
            'embeddings': embeddings,
            'metadata': all_metadata
        }
```

### Step 5: Vector Database (`src/vector_store.py`)

```python
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple

class VectorStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine similarity)
        self.metadata = []
        self.chunks = []
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[str], metadata: List[Dict]):
        """Add embeddings to the vector store."""
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings.astype('float32'))
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)
        
        print(f"Added {len(embeddings)} embeddings to vector store")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        # Normalize query embedding
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # Valid result
                results.append({
                    'content': self.chunks[idx],
                    'metadata': self.metadata[idx],
                    'similarity_score': float(distance),
                    'rank': i + 1
                })
        
        return results
    
    def save(self, filepath: str):
        """Save the vector store to disk."""
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save metadata and chunks
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"Vector store saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the vector store from disk."""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.index")
        
        # Load metadata and chunks
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']
            self.embedding_dim = data['embedding_dim']
        
        print(f"Vector store loaded from {filepath}")
```

### Step 6: LLM Integration (`src/retriever.py`)

```python
import ollama
from typing import List, Dict

class LLMRetriever:
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name
        self.client = ollama.Client()
    
    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate response using retrieved context."""
        # Prepare context from retrieved chunks
        context = "\n\n".join([
            f"Document: {chunk['metadata']['filename']}\n{chunk['content']}"
            for chunk in context_chunks
        ])
        
        prompt = f"""Based on the following context documents, please answer the question.

Context:
{context}

Question: {query}

Answer: Provide a comprehensive answer based on the context above. If the information is not available in the context, please state that clearly."""

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def summarize_documents(self, context_chunks: List[Dict]) -> str:
        """Generate a summary of the provided documents."""
        context = "\n\n".join([
            f"Document: {chunk['metadata']['filename']}\n{chunk['content']}"
            for chunk in context_chunks
        ])
        
        prompt = f"""Please provide a comprehensive summary of the following documents:

{context}

Summary: Create a detailed summary that captures the key points, main themes, and important information from these documents."""

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating summary: {str(e)}"
```

### Step 7: Main RAG System (`src/rag_system.py`)

```python
from src.document_parser import DocumentParser
from src.embedder import TextEmbedder
from src.vector_store import VectorStore
from src.retriever import LLMRetriever
import os

class PersonalRAG:
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.parser = DocumentParser()
        self.embedder = TextEmbedder()
        self.vector_store = VectorStore(self.embedder.embedding_dim)
        self.llm = LLMRetriever(model_name)
        self.is_indexed = False
    
    def build_index(self, documents_dir: str, save_path: str = "data/vector_store"):
        """Build the vector index from documents."""
        print("Starting document indexing...")
        
        # Parse documents
        documents = self.parser.parse_directory(documents_dir)
        print(f"Parsed {len(documents)} documents")
        
        # Create embeddings
        processed_data = self.embedder.process_documents(documents)
        
        # Add to vector store
        self.vector_store.add_embeddings(
            processed_data['embeddings'],
            processed_data['chunks'],
            processed_data['metadata']
        )
        
        # Save vector store
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.vector_store.save(save_path)
        
        self.is_indexed = True
        print("Indexing complete!")
    
    def load_index(self, save_path: str = "data/vector_store"):
        """Load existing vector index."""
        if os.path.exists(f"{save_path}.index"):
            self.vector_store.load(save_path)
            self.is_indexed = True
            print("Index loaded successfully!")
        else:
            print("No existing index found. Please build index first.")
    
    def query(self, question: str, top_k: int = 3) -> str:
        """Query the RAG system."""
        if not self.is_indexed:
            return "Please build or load an index first."
        
        # Create query embedding
        query_embedding = self.embedder.create_embeddings([question])
        
        # Search for relevant chunks
        results = self.vector_store.search(query_embedding, k=top_k)
        
        # Generate response
        response = self.llm.generate_response(question, results)
        
        return response
    
    def summarize(self, top_k: int = 5) -> str:
        """Generate a summary of all documents."""
        if not self.is_indexed:
            return "Please build or load an index first."
        
        # Get a representative sample of chunks for summary
        # For simplicity, we'll get top chunks from a generic query
        sample_query = "What are the main topics discussed in these documents?"
        query_embedding = self.embedder.create_embeddings([sample_query])
        results = self.vector_store.search(query_embedding, k=top_k)
        
        summary = self.llm.summarize_documents(results)
        return summary
```

### Step 8: Main Script (`main.py`)

```python
from src.rag_system import PersonalRAG
import os

def main():
    # Initialize RAG system
    rag = PersonalRAG()
    
    # Create documents directory if it doesn't exist
    os.makedirs("documents", exist_ok=True)
    
    print("Personal RAG System")
    print("==================")
    print("Commands:")
    print("1. 'build' - Build index from documents/")
    print("2. 'load' - Load existing index")
    print("3. 'query <question>' - Ask a question")
    print("4. 'summarize' - Get document summary")
    print("5. 'quit' - Exit")
    print()
    
    while True:
        command = input("Enter command: ").strip()
        
        if command == 'quit':
            break
        elif command == 'build':
            if not os.listdir("documents"):
                print("Please add documents to the 'documents/' directory first.")
                continue
            rag.build_index("documents")
        elif command == 'load':
            rag.load_index()
        elif command.startswith('query '):
            question = command[6:]  # Remove 'query ' prefix
            if question:
                print("\nSearching and generating response...")
                response = rag.query(question)
                print(f"\nAnswer: {response}\n")
            else:
                print("Please provide a question after 'query'")
        elif command == 'summarize':
            print("\nGenerating summary...")
            summary = rag.summarize()
            print(f"\nSummary: {summary}\n")
        else:
            print("Unknown command. Please try again.")

if __name__ == "__main__":
    main()
```

## Step 9: Running Your RAG System

### 9.1 Setup
```bash
# Activate your environment
source rag_env/bin/activate

# Create project structure
mkdir -p personal_rag/{documents,data,src}
cd personal_rag

# Copy all the code files to their respective locations
```

### 9.2 Add Your Documents
Place your 4 documents (PDFs, Word docs, or text files) in the `documents/` directory.

### 9.3 Run the System
```bash
python main.py
```

### 9.4 Usage Examples
```
# Build the index
> build

# Query your documents
> query What are the main findings in the research?
> query How does this relate to machine learning?

# Get a summary
> summarize
```

## Understanding Each Component

### Document Parser
- Handles multiple file formats (PDF, DOCX, TXT)
- Extracts text content and maintains metadata
- Extensible for other formats

### Text Embedder
- Uses sentence-transformers for semantic embeddings
- Chunks documents with overlap for better context
- Lightweight model that runs efficiently on M3

### Vector Store
- Uses FAISS for fast similarity search
- Stores embeddings with associated metadata
- Persistent storage for reuse

### LLM Integration
- Uses Ollama for local inference
- No API keys or internet required
- Customizable prompts for different tasks

## Next Steps and Enhancements

Once you have the basic system working, you can enhance it with:

1. **Better Chunking**: Implement semantic chunking based on sentences/paragraphs
2. **Metadata Filtering**: Add ability to filter by document type or date
3. **Query Expansion**: Enhance queries with synonyms or related terms
4. **Re-ranking**: Add a re-ranking step for better result ordering
5. **Web Interface**: Build a simple web UI using Flask or Streamlit
6. **Evaluation**: Add metrics to measure retrieval and generation quality

## Troubleshooting

### Common Issues:
1. **Ollama not found**: Ensure Ollama is installed and in PATH
2. **Memory issues**: Reduce chunk size or use smaller embedding models
3. **PDF parsing errors**: Some PDFs are image-based; consider OCR solutions
4. **Slow performance**: Use GPU acceleration if available

This system gives you complete control and understanding of each RAG component while staying within your zero-budget constraint.