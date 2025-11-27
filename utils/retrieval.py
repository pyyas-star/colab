"""
Retrieval Pipeline Module

Handles document retrieval with similarity search, reranking,
and medical domain filtering.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

from .vector_store import VectorStore
from .embeddings import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Retrieval engine for medical Q&A system.
    
    Features:
    - Similarity-based retrieval
    - Reranking (optional)
    - Medical topic filtering
    - Score thresholding
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        top_k: int = 5,
        score_threshold: float = 0.0
    ):
        """
        Initialize retrieval engine.
        
        Args:
            vector_store: VectorStore instance
            embedding_generator: EmbeddingGenerator instance
            top_k: Number of results to retrieve
            score_threshold: Minimum similarity score threshold
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.top_k = top_k
        self.score_threshold = score_threshold
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_by_disease: Optional[str] = None,
        filter_by_source: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of results (overrides default)
            filter_by_disease: Optional disease filter
            filter_by_source: Optional source filter
        
        Returns:
            List of retrieved documents with scores
        """
        if top_k is None:
            top_k = self.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embeddings(
            [query],
            show_progress=False,
            cache=False
        )[0]
        
        # Search vector store
        distances, indices, metadata_list = self.vector_store.search(
            query_embedding,
            k=top_k * 2  # Get more results for filtering
        )
        
        # Convert distances to similarity scores (cosine similarity)
        # For L2 distance, convert to similarity: 1 / (1 + distance)
        similarities = 1 / (1 + distances)
        
        # Combine results
        results = []
        for i, (dist, sim, meta) in enumerate(zip(distances, similarities, metadata_list)):
            # Apply filters
            if filter_by_disease:
                doc_diseases = meta.get("metadata", {}).get("diseases", [])
                if filter_by_disease.lower() not in [d.lower() for d in doc_diseases]:
                    continue
            
            if filter_by_source:
                if meta.get("metadata", {}).get("source", "").lower() != filter_by_source.lower():
                    continue
            
            # Apply score threshold
            if sim < self.score_threshold:
                continue
            
            result = {
                "text": meta.get("text", ""),
                "chunk_id": meta.get("chunk_id", ""),
                "document_id": meta.get("document_id", ""),
                "similarity_score": float(sim),
                "distance": float(dist),
                "rank": len(results) + 1,
                "metadata": meta.get("metadata", {})
            }
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results
    
    def rerank(
        self,
        query: str,
        results: List[Dict],
        method: str = "similarity"
    ) -> List[Dict]:
        """
        Rerank retrieval results (simple implementation).
        
        Args:
            query: Original query
            results: Retrieved results
            method: Reranking method ('similarity' or 'keyword')
        
        Returns:
            Reranked results
        """
        if method == "keyword":
            # Simple keyword-based reranking
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            for result in results:
                text_lower = result["text"].lower()
                text_words = set(text_lower.split())
                
                # Count overlapping words
                overlap = len(query_words & text_words)
                result["keyword_overlap"] = overlap
                result["rerank_score"] = result["similarity_score"] + (overlap * 0.1)
            
            # Sort by rerank score
            results.sort(key=lambda x: x.get("rerank_score", x["similarity_score"]), reverse=True)
        
        return results


def retrieve_relevant_docs(
    query: str,
    vector_store: VectorStore,
    embedding_generator: EmbeddingGenerator,
    top_k: int = 5
) -> List[Dict]:
    """
    Convenience function to retrieve documents.
    
    Args:
        query: Query string
        vector_store: VectorStore instance
        embedding_generator: EmbeddingGenerator instance
        top_k: Number of results
    
    Returns:
        List of retrieved documents
    """
    engine = RetrievalEngine(vector_store, embedding_generator, top_k=top_k)
    return engine.retrieve(query)

