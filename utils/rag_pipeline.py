"""
RAG Pipeline Module

Orchestrates the complete RAG pipeline: retrieval + generation.
Handles prompt construction, LLM integration, and response formatting.
"""

import logging
from typing import List, Dict, Optional
import re

from .retrieval import RetrievalEngine
from .safety import MedicalSafetyChecker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. Install with: pip install transformers torch")


class RAGPipeline:
    """
    Complete RAG pipeline for medical Q&A.
    
    Features:
    - Query validation
    - Document retrieval
    - Context-aware answer generation
    - Citation formatting
    - Safety checks
    """
    
    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        llm_model_name: Optional[str] = None,
        use_api: bool = False,
        api_key: Optional[str] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            retrieval_engine: RetrievalEngine instance
            llm_model_name: Hugging Face model name (if not using API)
            use_api: Whether to use OpenAI API instead of local model
            api_key: API key if using external API
        """
        self.retrieval_engine = retrieval_engine
        self.safety_checker = MedicalSafetyChecker()
        self.use_api = use_api
        self.api_key = api_key
        
        if use_api:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=api_key) if api_key else None
            except ImportError:
                logger.warning("openai package not available. Install with: pip install openai")
                self.use_api = False
        
        if not use_api:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers required for local LLM. Install with: pip install transformers torch")
            
            # Use a lightweight model for Colab
            self.llm_model_name = llm_model_name or "google/flan-t5-base"
            self._load_local_llm()
    
    def _load_local_llm(self):
        """Load local LLM model."""
        logger.info(f"Loading local LLM: {self.llm_model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            logger.info("Local LLM loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LLM: {e}")
            # Fallback to pipeline
            self.generator = pipeline(
                "text2text-generation",
                model=self.llm_model_name,
                device=0 if torch.cuda.is_available() else -1
            )
    
    def construct_prompt(
        self,
        query: str,
        retrieved_docs: List[Dict],
        include_citations: bool = True
    ) -> str:
        """
        Construct prompt for LLM with retrieved context.
        
        Args:
            query: User query
            retrieved_docs: Retrieved document chunks
            include_citations: Whether to include citation instructions
        
        Returns:
            Formatted prompt string
        """
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source_info = doc.get("metadata", {}).get("source", "Medical Guidelines")
            title = doc.get("metadata", {}).get("title", "")
            
            context_part = f"[Source {i}: {source_info}"
            if title:
                context_part += f" - {title}"
            context_part += "]\n"
            context_part += doc["text"]
            context_parts.append(context_part)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Construct prompt
        prompt = f"""You are a medical information assistant providing evidence-based answers from trusted medical guidelines (WHO, CDC, etc.).

Context from Medical Guidelines:
{context}

Question: {query}

Instructions:
- Provide a clear, accurate answer based ONLY on the provided context
- If the context doesn't contain enough information, say so
- Use medical terminology appropriately
- Include relevant details from the guidelines
"""
        
        if include_citations:
            prompt += "- Cite your sources using [Source 1], [Source 2], etc.\n"
        
        prompt += "\nAnswer:"
        
        return prompt
    
    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        max_length: int = 512,
        include_citations: bool = True
    ) -> Dict:
        """
        Generate answer using RAG pipeline.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            max_length: Maximum answer length
            include_citations: Whether to include citations
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Validate query
        is_valid, warning, safety_flags = self.safety_checker.validate_medical_query(query)
        
        if not is_valid:
            return {
                "answer": warning or "Query validation failed.",
                "sources": [],
                "citations": [],
                "safety_warning": warning,
                "safety_flags": safety_flags,
                "is_valid": False
            }
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieval_engine.retrieve(query, top_k=top_k)
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find relevant information in the medical guidelines for your question. Please consult a healthcare provider for personalized advice.",
                "sources": [],
                "citations": [],
                "is_valid": True
            }
        
        # Construct prompt
        prompt = self.construct_prompt(query, retrieved_docs, include_citations)
        
        # Generate answer
        if self.use_api and self.openai_client:
            answer = self._generate_with_api(prompt, max_length)
        else:
            answer = self._generate_with_local_llm(prompt, max_length)
        
        # Format response with citations
        formatted_answer = self._format_response(answer, retrieved_docs, include_citations)
        
        # Add disclaimer
        final_answer = self.safety_checker.add_medical_disclaimer(formatted_answer)
        
        # Extract citations
        citations = self._extract_citations(retrieved_docs)
        
        return {
            "answer": final_answer,
            "sources": citations,
            "retrieved_docs": retrieved_docs,
            "safety_warning": warning,
            "safety_flags": safety_flags,
            "is_valid": True,
            "query": query
        }
    
    def _generate_with_api(self, prompt: str, max_length: int) -> str:
        """Generate answer using OpenAI API."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a medical information assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API generation error: {e}")
            return "Error generating response. Please try again."
    
    def _generate_with_local_llm(self, prompt: str, max_length: int) -> str:
        """Generate answer using local LLM."""
        try:
            if hasattr(self, 'generator'):
                # Using pipeline
                result = self.generator(
                    prompt,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.3,
                    do_sample=True
                )
                return result[0]['generated_text']
            else:
                # Using model directly
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_return_sequences=1,
                        temperature=0.3,
                        do_sample=True
                    )
                
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the prompt from the answer
                if answer.startswith(prompt):
                    answer = answer[len(prompt):].strip()
                return answer
        except Exception as e:
            logger.error(f"Local LLM generation error: {e}")
            return "Error generating response. Please try again."
    
    def _format_response(self, answer: str, retrieved_docs: List[Dict], include_citations: bool) -> str:
        """Format response with citations."""
        if not include_citations:
            return answer
        
        # Ensure citations are present
        if not re.search(r'\[Source \d+\]', answer):
            # Add citations manually
            citation_text = "\n\n**Sources:**\n"
            for i, doc in enumerate(retrieved_docs, 1):
                source = doc.get("metadata", {}).get("source", "Medical Guidelines")
                title = doc.get("metadata", {}).get("title", "")
                citation_text += f"[{i}] {source}"
                if title:
                    citation_text += f" - {title}"
                citation_text += "\n"
            answer += citation_text
        
        return answer
    
    def _extract_citations(self, retrieved_docs: List[Dict]) -> List[Dict]:
        """Extract citation information from retrieved documents."""
        citations = []
        for i, doc in enumerate(retrieved_docs, 1):
            citations.append({
                "source_number": i,
                "source": doc.get("metadata", {}).get("source", "Medical Guidelines"),
                "title": doc.get("metadata", {}).get("title", ""),
                "chunk_id": doc.get("chunk_id", ""),
                "similarity_score": doc.get("similarity_score", 0.0)
            })
        return citations


def generate_answer(
    query: str,
    retrieval_engine: RetrievalEngine,
    llm_model_name: Optional[str] = None
) -> Dict:
    """
    Convenience function to generate answer.
    
    Args:
        query: User query
        retrieval_engine: RetrievalEngine instance
        llm_model_name: Optional LLM model name
    
    Returns:
        Answer dictionary
    """
    pipeline = RAGPipeline(retrieval_engine, llm_model_name=llm_model_name)
    return pipeline.generate_answer(query)

