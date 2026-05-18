from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

"""
TextEmbedder: A class to handle text embedding using a lightweight sentence transformer model.
This class is responsible for splitting documents into manageable chunks, creating embeddings for those chunks,
and preparing the data for storage in the vector index.
"""
class TextEmbedder:
    """TextEmbedder: A class to handle text embedding using a lightweight sentence transformer model."""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    """Split text into overlapping chunks."""
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
        
        return chunks
    
    """Create embeddings for a list of texts."""
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        print(f"Creating embeddings for {len(texts)} text chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    """Process documents into chunks with embeddings and metadata."""
    def process_documents(self, documents: List[Dict[str, str]]) -> Dict:
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