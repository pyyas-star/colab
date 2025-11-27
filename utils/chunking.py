"""
Text Chunking Module

Implements advanced chunking strategies optimized for medical documents.
Handles metadata extraction and maintains context across chunks.
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: str
    document_id: str
    start_idx: int
    end_idx: int
    metadata: Dict
    overlap_with_prev: int = 0


class MedicalChunker:
    """
    Advanced chunker optimized for medical documents.
    
    Features:
    - Recursive text splitting with overlap
    - Sentence-aware chunking
    - Metadata preservation
    - Medical section detection
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size for chunks (characters)
            chunk_overlap: Overlap between chunks (characters)
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(
        self,
        text: str,
        document_id: str = "doc_0",
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Chunk text into overlapping segments.
        
        Args:
            text: Text to chunk
            document_id: ID of the source document
            metadata: Document metadata to attach to chunks
        
        Returns:
            List of Chunk objects
        """
        if metadata is None:
            metadata = {}
        
        chunks = []
        
        # Try to split by paragraphs first
        paragraphs = self._split_by_paragraphs(text)
        
        current_chunk = ""
        current_start = 0
        chunk_idx = 0
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size, finalize current chunk
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunk = self._create_chunk(
                    current_chunk,
                    document_id,
                    chunk_idx,
                    current_start,
                    current_start + len(current_chunk),
                    metadata
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + para
                current_start = chunk.end_idx - len(overlap_text)
                chunk_idx += 1
            else:
                current_chunk += para
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk,
                document_id,
                chunk_idx,
                current_start,
                current_start + len(current_chunk),
                metadata
            )
            chunks.append(chunk)
        
        # Filter out chunks that are too small
        chunks = [c for c in chunks if len(c.text.strip()) >= self.min_chunk_size]
        
        logger.info(f"Created {len(chunks)} chunks from document {document_id}")
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() + '\n\n' for p in paragraphs if p.strip()]
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get the last N characters for overlap."""
        if len(text) <= overlap_size:
            return text
        return text[-overlap_size:]
    
    def _create_chunk(
        self,
        text: str,
        document_id: str,
        chunk_idx: int,
        start_idx: int,
        end_idx: int,
        metadata: Dict
    ) -> Chunk:
        """Create a Chunk object with metadata."""
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            "chunk_index": chunk_idx,
            "char_count": len(text),
            "word_count": len(text.split())
        })
        
        return Chunk(
            text=text.strip(),
            chunk_id=f"{document_id}_chunk_{chunk_idx}",
            document_id=document_id,
            start_idx=start_idx,
            end_idx=end_idx,
            metadata=chunk_metadata
        )


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    document_id: str = "doc_0",
    metadata: Optional[Dict] = None
) -> List[Dict]:
    """
    Convenience function to chunk text.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        document_id: Document identifier
        metadata: Document metadata
    
    Returns:
        List of chunk dictionaries
    """
    chunker = MedicalChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_text(text, document_id, metadata or {})
    
    return [
        {
            "text": chunk.text,
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]


def extract_metadata(text: str, document_metadata: Optional[Dict] = None) -> Dict:
    """
    Extract metadata from medical document text.
    
    Args:
        text: Document text
        document_metadata: Existing metadata to enhance
    
    Returns:
        Enhanced metadata dictionary
    """
    metadata = document_metadata.copy() if document_metadata else {}
    
    # Extract title (usually first line or after "Title:")
    title_match = re.search(r'(?:Title|TITLE):\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if not title_match:
        # Try first line if it looks like a title
        first_line = text.split('\n')[0].strip()
        if len(first_line) < 100 and first_line.isupper():
            title_match = type('obj', (object,), {'group': lambda x: first_line})()
    
    if title_match:
        metadata["title"] = title_match.group(1).strip()
    
    # Extract disease/condition mentions
    disease_keywords = _extract_disease_mentions(text)
    if disease_keywords:
        metadata["diseases"] = disease_keywords
    
    # Detect sections
    sections = _detect_sections(text)
    if sections:
        metadata["sections"] = sections
    
    # Extract dates
    date_patterns = [
        r'\b(20\d{2}|19\d{2})\b',  # Years
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
    ]
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))
    if dates:
        metadata["dates"] = list(set(dates))
    
    return metadata


def _extract_disease_mentions(text: str) -> List[str]:
    """Extract disease/condition mentions from text."""
    # Common medical conditions
    common_diseases = [
        'malaria', 'tuberculosis', 'TB', 'HIV', 'AIDS', 'diabetes',
        'hypertension', 'pneumonia', 'asthma', 'epilepsy', 'covid-19',
        'influenza', 'hepatitis', 'dengue', 'cholera', 'typhoid'
    ]
    
    found = []
    text_lower = text.lower()
    for disease in common_diseases:
        if disease in text_lower:
            found.append(disease)
    
    # Pattern matching for "X disease", "X syndrome", etc.
    pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:disease|syndrome|disorder|condition|infection)\b'
    matches = re.findall(pattern, text)
    found.extend([m.lower() for m in matches])
    
    return list(set(found))


def _detect_sections(text: str) -> List[str]:
    """Detect document sections (Symptoms, Treatment, Prevention, etc.)."""
    section_keywords = [
        'symptoms', 'treatment', 'prevention', 'diagnosis', 'causes',
        'risk factors', 'complications', 'management', 'guidelines',
        'recommendations', 'epidemiology', 'pathophysiology'
    ]
    
    found_sections = []
    text_lower = text.lower()
    
    for keyword in section_keywords:
        # Look for section headers (usually followed by colon or newline)
        pattern = rf'\b{keyword}\b(?:\s*:|\s*\n)'
        if re.search(pattern, text_lower, re.IGNORECASE):
            found_sections.append(keyword.title())
    
    return found_sections

