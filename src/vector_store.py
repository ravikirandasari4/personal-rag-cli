import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple

"""
VectorStore: A class to manage the vector index for document embeddings.
This class handles adding embeddings to the index, searching for similar documents based on query embeddings,
and saving/loading the index to/from disk. It uses FAISS for efficient similarity search and supports cosine similarity.
"""
class VectorStore:
    """VectorStore: A class to manage the vector index for document embeddings."""
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine similarity)
        self.metadata = []
        self.chunks = []
    
    """Add embeddings to the vector store."""
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[str], metadata: List[Dict]):
        """Add embeddings to the vector store."""
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings.astype('float32'))
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)
        
        print(f"Added {len(embeddings)} embeddings to vector store")
    
    """Search for similar documents."""
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
    
    """Save the vector store to disk."""
    def save(self, filepath: str):
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
    
    """Load the vector store from disk."""
    def load(self, filepath: str):
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.index")
        
        # Load metadata and chunks
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']
            self.embedding_dim = data['embedding_dim']
        
        print(f"Vector store loaded from {filepath}")