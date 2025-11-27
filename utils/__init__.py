"""
Medical RAG Q&A Agent - Utility Modules

This package contains modular components for the Medical RAG system:
- Data loading and preprocessing
- Text chunking strategies
- Embedding generation
- Vector store management
- Retrieval pipeline
- RAG pipeline orchestration
- Evaluation metrics
- Medical domain safety checks
"""

__version__ = "1.0.0"

from .data_loader import load_data, clean_text, structure_documents
from .chunking import chunk_text, extract_metadata, MedicalChunker
from .embeddings import create_embeddings, EmbeddingGenerator
from .vector_store import build_vector_store, VectorStore
from .retrieval import retrieve_relevant_docs, RetrievalEngine
from .rag_pipeline import RAGPipeline, generate_answer
from .evaluation import evaluate_rag_system, RAGEvaluator
from .safety import validate_medical_query, MedicalSafetyChecker

__all__ = [
    "load_data",
    "clean_text",
    "structure_documents",
    "chunk_text",
    "extract_metadata",
    "MedicalChunker",
    "create_embeddings",
    "EmbeddingGenerator",
    "build_vector_store",
    "VectorStore",
    "retrieve_relevant_docs",
    "RetrievalEngine",
    "RAGPipeline",
    "generate_answer",
    "evaluate_rag_system",
    "RAGEvaluator",
    "validate_medical_query",
    "MedicalSafetyChecker",
]

