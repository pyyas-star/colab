"""
Data Loading and Preprocessing Module

Handles ingestion, cleaning, and structuring of medical documents
from various sources (WHO, CDC, local files, URLs).
"""

import re
import requests
from typing import List, Dict, Optional, Union
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(
    source: Union[str, Path, List[str]],
    source_type: str = "file"
) -> List[str]:
    """
    Load medical documents from various sources.
    
    Args:
        source: File path(s), URL(s), or list of text strings
        source_type: Type of source - 'file', 'url', or 'text'
    
    Returns:
        List of raw document texts
    """
    documents = []
    
    if source_type == "file":
        if isinstance(source, (str, Path)):
            source = [source]
        
        for file_path in source:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
                    logger.info(f"Loaded {file_path}: {len(content)} characters")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
    
    elif source_type == "url":
        if isinstance(source, str):
            source = [source]
        
        for url in source:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                documents.append(response.text)
                logger.info(f"Loaded URL: {url}")
            except Exception as e:
                logger.error(f"Error loading URL {url}: {e}")
    
    elif source_type == "text":
        if isinstance(source, str):
            documents = [source]
        else:
            documents = source
    
    return documents


def clean_text(text: str) -> str:
    """
    Clean and normalize medical text.
    
    Removes:
    - Extra whitespace
    - Non-printable characters
    - HTML tags (if present)
    - Normalizes line breaks
    
    Args:
        text: Raw text to clean
    
    Returns:
        Cleaned text
    """
    # Remove HTML tags if present
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-printable characters except newlines and tabs
    text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
    
    # Normalize line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def structure_documents(
    documents: List[str],
    metadata: Optional[List[Dict]] = None
) -> List[Dict]:
    """
    Structure documents with metadata for RAG pipeline.
    
    Args:
        documents: List of cleaned document texts
        metadata: Optional list of metadata dicts (one per document)
    
    Returns:
        List of structured documents with metadata
    """
    structured = []
    
    for idx, doc in enumerate(documents):
        doc_metadata = metadata[idx] if metadata and idx < len(metadata) else {}
        
        structured_doc = {
            "id": doc_metadata.get("id", f"doc_{idx}"),
            "text": doc,
            "title": doc_metadata.get("title", ""),
            "source": doc_metadata.get("source", "unknown"),
            "disease": doc_metadata.get("disease", ""),
            "section": doc_metadata.get("section", ""),
            "date": doc_metadata.get("date", ""),
            "metadata": doc_metadata
        }
        
        structured.append(structured_doc)
    
    return structured


def extract_disease_keywords(text: str) -> List[str]:
    """
    Extract potential disease/condition keywords from text.
    
    Simple keyword extraction - can be enhanced with NER models.
    
    Args:
        text: Medical text
    
    Returns:
        List of potential disease keywords
    """
    # Common medical condition patterns
    patterns = [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:disease|syndrome|disorder|condition|infection)\b',
        r'\b(malaria|tuberculosis|TB|HIV|AIDS|diabetes|hypertension|pneumonia|asthma|epilepsy)\b',
    ]
    
    keywords = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        keywords.update([m.lower() if isinstance(m, str) else m[0].lower() for m in matches])
    
    return list(keywords)

