from src.document_parser import DocumentParser
from src.embedder import TextEmbedder
from src.vector_store import VectorStore
from src.retriever import LLMRetriever
import os

"""
PersonalRAG: A Retrieval-Augmented Generation system for personal document management.
This class integrates document parsing, embedding, vector storage, and LLM retrieval to provide a seamless
experience for indexing and querying personal documents.
"""
class PersonalRAG:
    """Initialize the RAG system components."""
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.parser = DocumentParser()
        self.embedder = TextEmbedder()
        self.vector_store = VectorStore(self.embedder.embedding_dim)
        self.llm = LLMRetriever(model_name)
        self.is_indexed = False
    
    """Build the vector index from documents in the specified directory."""
    def build_index(self, documents_dir: str, save_path: str = "data/vector_store"):
        
        print("Starting document indexing...")
        
        # Parse documents
        documents = self.parser.parse_directory(documents_dir)
        print(f"Parsed {len(documents)} documents")
        
        if not documents:
            print("No supported documents found. Please add .txt, .pdf, or .docx files to the documents directory.")
            return
        
        # Create embeddings
        processed_data = self.embedder.process_documents(documents)
        
        if processed_data['embeddings'].size == 0:
            print("No text chunks were generated from the documents. Please verify document contents.")
            return
        
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
    
    """Load an existing vector index from disk."""
    def load_index(self, save_path: str = "data/vector_store"):
        """Load existing vector index."""
        if os.path.exists(f"{save_path}.index") and os.path.exists(f"{save_path}.pkl"):
            self.vector_store.load(save_path)
            self.is_indexed = True
            print("Index loaded successfully!")
        else:
            print("No existing index found. Please build index first.")
    
    """Query the RAG system with a question and retrieve an answer based on indexed documents."""
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
    
    """Generate a summary of all documents."""
    def summarize(self, top_k: int = 5) -> str:
        if not self.is_indexed:
            return "Please build or load an index first."
        
        # Get a representative sample of chunks for summary
        # For simplicity, we'll get top chunks from a generic query
        sample_query = "What are the main topics discussed in these documents?"
        query_embedding = self.embedder.create_embeddings([sample_query])
        results = self.vector_store.search(query_embedding, k=top_k)
        
        summary = self.llm.summarize_documents(results)
        return summary