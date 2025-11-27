"""
Embedding Generation Module

Handles creation of embeddings for medical text chunks using
sentence transformers, with caching and batch processing support.
"""

import numpy as np
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")


class EmbeddingGenerator:
    """
    Generates embeddings for medical text using sentence transformers.
    
    Features:
    - Batch processing for efficiency
    - Embedding caching
    - Multiple model support
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Hugging Face model name
            cache_dir: Directory to cache embeddings
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Auto-detect device if not specified
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        
        self.device = device
        
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        cache: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            show_progress: Show progress bar
            cache: Use caching if available
        
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Check cache
        if cache and self.cache_dir:
            cached_embeddings = self._load_from_cache(texts)
            if cached_embeddings is not None:
                logger.info("Loaded embeddings from cache")
                return cached_embeddings
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        # Save to cache
        if cache and self.cache_dir:
            self._save_to_cache(texts, embeddings)
        
        return embeddings
    
    def _load_from_cache(self, texts: List[str]) -> Optional[np.ndarray]:
        """Load embeddings from cache if available."""
        if not self.cache_dir or not self.cache_dir.exists():
            return None
        
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if texts match
            if cache_data.get("texts") == texts:
                return cache_data.get("embeddings")
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
        
        return None
    
    def _save_to_cache(self, texts: List[str], embeddings: np.ndarray):
        """Save embeddings to cache."""
        if not self.cache_dir:
            return
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    "texts": texts,
                    "embeddings": embeddings,
                    "model_name": self.model_name
                }, f)
            logger.info(f"Cached embeddings to {cache_file}")
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")


def create_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    device: Optional[str] = None
) -> np.ndarray:
    """
    Convenience function to create embeddings.
    
    Args:
        texts: List of texts to embed
        model_name: Model name for sentence transformers
        batch_size: Batch size for processing
        device: Device to use
    
    Returns:
        Numpy array of embeddings
    """
    generator = EmbeddingGenerator(model_name=model_name, device=device)
    return generator.generate_embeddings(texts, batch_size=batch_size)

