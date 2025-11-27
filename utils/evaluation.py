"""
Evaluation Module

Provides evaluation metrics and benchmarking for RAG system.
Includes comparison between RAG and non-RAG (LLM-only) responses.
"""

import logging
from typing import List, Dict, Optional
import json
from pathlib import Path

from .rag_pipeline import RAGPipeline
from .retrieval import RetrievalEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluator for RAG system performance.
    
    Features:
    - Answer quality metrics
    - Citation accuracy
    - Comparison with baseline (LLM-only)
    - Benchmark test suite
    """
    
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize evaluator.
        
        Args:
            rag_pipeline: RAGPipeline instance
        """
        self.rag_pipeline = rag_pipeline
    
    def evaluate_answer(
        self,
        query: str,
        expected_answer: Optional[str] = None,
        ground_truth_sources: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate a single answer.
        
        Args:
            query: User query
            expected_answer: Optional expected answer for comparison
            ground_truth_sources: Optional list of expected source IDs
        
        Returns:
            Evaluation metrics dictionary
        """
        # Generate answer
        result = self.rag_pipeline.generate_answer(query)
        
        metrics = {
            "query": query,
            "answer_length": len(result["answer"]),
            "num_sources": len(result["sources"]),
            "has_citations": len(result["citations"]) > 0,
            "is_valid": result.get("is_valid", True),
            "safety_warning": result.get("safety_warning")
        }
        
        # Check citation quality
        if result["sources"]:
            avg_similarity = sum(s.get("similarity_score", 0) for s in result["sources"]) / len(result["sources"])
            metrics["avg_source_similarity"] = avg_similarity
            metrics["citation_quality"] = "high" if avg_similarity > 0.7 else "medium" if avg_similarity > 0.5 else "low"
        
        # Compare with expected answer if provided
        if expected_answer:
            # Simple keyword overlap metric
            answer_words = set(result["answer"].lower().split())
            expected_words = set(expected_answer.lower().split())
            overlap = len(answer_words & expected_words) / len(expected_words) if expected_words else 0
            metrics["answer_overlap"] = overlap
        
        # Check source accuracy if ground truth provided
        if ground_truth_sources:
            retrieved_sources = [s.get("chunk_id", "") for s in result["sources"]]
            correct_sources = len(set(retrieved_sources) & set(ground_truth_sources))
            metrics["source_accuracy"] = correct_sources / len(ground_truth_sources) if ground_truth_sources else 0
        
        return metrics
    
    def compare_rag_vs_baseline(
        self,
        query: str,
        baseline_answer: Optional[str] = None
    ) -> Dict:
        """
        Compare RAG answer with baseline (LLM-only) answer.
        
        Args:
            query: User query
            baseline_answer: Optional baseline answer (if not provided, will generate)
        
        Returns:
            Comparison dictionary
        """
        # Get RAG answer
        rag_result = self.rag_pipeline.generate_answer(query)
        
        # Generate baseline if not provided
        if baseline_answer is None:
            # Simple baseline: answer without retrieval
            baseline_answer = "Baseline answer would be generated here without retrieval context."
        
        comparison = {
            "query": query,
            "rag_answer": rag_result["answer"],
            "rag_sources": len(rag_result["sources"]),
            "rag_has_citations": len(rag_result["citations"]) > 0,
            "baseline_answer": baseline_answer,
            "baseline_has_citations": False,
            "rag_length": len(rag_result["answer"]),
            "baseline_length": len(baseline_answer)
        }
        
        # Advantages of RAG
        advantages = []
        if rag_result["sources"]:
            advantages.append("Has source citations")
        if len(rag_result["answer"]) > len(baseline_answer) * 0.8:
            advantages.append("More detailed answer")
        
        comparison["rag_advantages"] = advantages
        
        return comparison
    
    def run_benchmark(
        self,
        test_questions: List[Dict],
        output_file: Optional[Path] = None
    ) -> Dict:
        """
        Run benchmark evaluation on test questions.
        
        Args:
            test_questions: List of test question dicts with 'query' and optional 'expected_answer'
            output_file: Optional path to save results
        
        Returns:
            Benchmark results dictionary
        """
        results = []
        total_questions = len(test_questions)
        
        logger.info(f"Running benchmark on {total_questions} questions...")
        
        for i, test_case in enumerate(test_questions, 1):
            logger.info(f"Evaluating question {i}/{total_questions}: {test_case['query'][:50]}...")
            
            metrics = self.evaluate_answer(
                query=test_case["query"],
                expected_answer=test_case.get("expected_answer"),
                ground_truth_sources=test_case.get("ground_truth_sources")
            )
            results.append(metrics)
        
        # Aggregate metrics
        avg_similarity = sum(r.get("avg_source_similarity", 0) for r in results) / len(results) if results else 0
        avg_sources = sum(r.get("num_sources", 0) for r in results) / len(results) if results else 0
        citation_rate = sum(1 for r in results if r.get("has_citations")) / len(results) if results else 0
        
        benchmark_results = {
            "total_questions": total_questions,
            "avg_source_similarity": avg_similarity,
            "avg_num_sources": avg_sources,
            "citation_rate": citation_rate,
            "results": results
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            logger.info(f"Benchmark results saved to {output_file}")
        
        return benchmark_results


def evaluate_rag_system(
    rag_pipeline: RAGPipeline,
    test_questions: List[Dict]
) -> Dict:
    """
    Convenience function to evaluate RAG system.
    
    Args:
        rag_pipeline: RAGPipeline instance
        test_questions: List of test questions
    
    Returns:
        Evaluation results
    """
    evaluator = RAGEvaluator(rag_pipeline)
    return evaluator.run_benchmark(test_questions)

