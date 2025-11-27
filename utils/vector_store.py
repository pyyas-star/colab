"""
Vector Store Module

Manages FAISS vector database for efficient similarity search.
Handles persistence, indexing, and retrieval operations.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu (or faiss-gpu)")


class VectorStore:
    """
    FAISS-based vector store for medical document chunks.
    
    Features:
    - Efficient similarity search
    - Persistence to disk
    - Metadata storage
    - Index management
    """
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat"
    ):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ('flat' or 'l2')
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is required. Install with: pip install faiss-cpu"
            )
        
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        # Create FAISS index
        if index_type == "flat":
            # Flat index - exact search, slower but accurate
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            # Use L2 distance
            self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Store metadata for each vector
        self.metadata_store: List[Dict] = []
        
        logger.info(f"Initialized vector store with {embedding_dim}D embeddings")
    
    def add_vectors(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict]
    ):
        """
        Add vectors and metadata to the store.
        
        Args:
            embeddings: Numpy array of embeddings (n_vectors, embedding_dim)
            metadata: List of metadata dicts (one per vector)
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        # Ensure embeddings are float32 for FAISS
        embeddings = embeddings.astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata_store.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} vectors to store. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query embedding vector (embedding_dim,)
            k: Number of results to return
        
        Returns:
            Tuple of (distances, indices, metadata_list)
        """
        if self.index.ntotal == 0:
            return np.array([]), np.array([]), []
        
        # Ensure query is float32 and reshape
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Get metadata for results
        result_metadata = [self.metadata_store[idx] for idx in indices[0]]
        
        return distances[0], indices[0], result_metadata
    
    def save(self, filepath: Path):
        """Save vector store to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(filepath.with_suffix('.index')))
        
        # Save metadata
        with open(filepath.with_suffix('.metadata.pkl'), 'wb') as f:
            pickle.dump({
                'metadata': self.metadata_store,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type
            }, f)
        
        logger.info(f"Saved vector store to {filepath}")
    
    def load(self, filepath: Path):
        """Load vector store from disk."""
        filepath = Path(filepath)
        
        # Load FAISS index
        self.index = faiss.read_index(str(filepath.with_suffix('.index')))
        
        # Load metadata
        with open(filepath.with_suffix('.metadata.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.metadata_store = data['metadata']
            self.embedding_dim = data['embedding_dim']
            self.index_type = data.get('index_type', 'flat')
        
        logger.info(f"Loaded vector store from {filepath}. {self.index.ntotal} vectors")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type
        }


def build_vector_store(
    embeddings: np.ndarray,
    chunks: List[Dict],
    save_path: Optional[Path] = None
) -> VectorStore:
    """
    Build a vector store from embeddings and chunks.
    
    Args:
        embeddings: Numpy array of embeddings
        chunks: List of chunk dictionaries with metadata
        save_path: Optional path to save the store
    
    Returns:
        VectorStore instance
    """
    if len(embeddings) != len(chunks):
        raise ValueError("Number of embeddings must match number of chunks")
    
    embedding_dim = embeddings.shape[1]
    store = VectorStore(embedding_dim=embedding_dim)
    
    # Prepare metadata
    metadata = [
        {
            "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
            "document_id": chunk.get("document_id", "unknown"),
            "text": chunk.get("text", ""),
            **chunk.get("metadata", {})
        }
        for i, chunk in enumerate(chunks)
    ]
    
    store.add_vectors(embeddings, metadata)
    
    if save_path:
        store.save(save_path)
    
    return store

