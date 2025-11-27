"""
Medical Domain Safety Module

Implements safety checks and validations for medical queries
to prevent harmful or inappropriate responses.
"""

import re
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalSafetyChecker:
    """
    Safety checker for medical queries and responses.
    
    Features:
    - Query validation
    - Harmful content detection
    - Medical disclaimer enforcement
    - Emergency situation detection
    """
    
    def __init__(self):
        """Initialize safety checker."""
        # Patterns for emergency situations
        self.emergency_patterns = [
            r'\b(emergency|urgent|immediate|severe pain|chest pain|difficulty breathing|unconscious|bleeding heavily)\b',
            r'\b(call 911|call ambulance|go to hospital|ER|emergency room)\b'
        ]
        
        # Patterns for personal medical advice requests
        self.personal_advice_patterns = [
            r'\b(should i|what should i|tell me what to do|prescribe|diagnose me|my symptoms|i have|i am)\b'
        ]
        
        # Harmful content patterns
        self.harmful_patterns = [
            r'\b(suicide|kill myself|end my life|self harm)\b',
            r'\b(drug abuse|illegal drugs|prescription abuse)\b'
        ]
    
    def validate_medical_query(self, query: str) -> Tuple[bool, Optional[str], Dict]:
        """
        Validate a medical query for safety.
        
        Args:
            query: User query
        
        Returns:
            Tuple of (is_valid, warning_message, safety_flags)
        """
        query_lower = query.lower()
        safety_flags = {
            "is_emergency": False,
            "requests_personal_advice": False,
            "contains_harmful_content": False,
            "requires_disclaimer": True
        }
        warnings = []
        
        # Check for emergency situations
        for pattern in self.emergency_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                safety_flags["is_emergency"] = True
                warnings.append(
                    "⚠️ EMERGENCY DETECTED: This appears to be a medical emergency. "
                    "Please contact emergency services immediately (911 or local emergency number). "
                    "This system is not a substitute for emergency medical care."
                )
                break
        
        # Check for personal medical advice requests
        for pattern in self.personal_advice_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                safety_flags["requests_personal_advice"] = True
                warnings.append(
                    "⚠️ This system provides general medical information only. "
                    "For personal medical advice, please consult a qualified healthcare provider."
                )
                break
        
        # Check for harmful content
        for pattern in self.harmful_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                safety_flags["contains_harmful_content"] = True
                warnings.append(
                    "⚠️ If you are experiencing a mental health crisis, please contact: "
                    "National Suicide Prevention Lifeline: 988 (US) or your local crisis hotline."
                )
                break
        
        # Determine if query is valid
        is_valid = not safety_flags["is_emergency"] and not safety_flags["contains_harmful_content"]
        
        warning_message = "\n".join(warnings) if warnings else None
        
        return is_valid, warning_message, safety_flags
    
    def add_medical_disclaimer(self, response: str) -> str:
        """
        Add medical disclaimer to response.
        
        Args:
            response: Generated response
        
        Returns:
            Response with disclaimer appended
        """
        disclaimer = (
            "\n\n---\n"
            "**Medical Disclaimer**: This information is for educational purposes only "
            "and is not a substitute for professional medical advice, diagnosis, or treatment. "
            "Always seek the advice of your physician or other qualified health provider "
            "with any questions you may have regarding a medical condition. "
            "Never disregard professional medical advice or delay in seeking it because of "
            "something you have read here."
        )
        return response + disclaimer
    
    def check_response_quality(self, response: str, query: str) -> Dict:
        """
        Check quality and safety of generated response.
        
        Args:
            response: Generated response
            query: Original query
        
        Returns:
            Quality metrics dictionary
        """
        quality_metrics = {
            "has_citations": bool(re.search(r'\[.*?\]|\(.*?\)|source|reference', response, re.IGNORECASE)),
            "has_disclaimer": "disclaimer" in response.lower() or "not a substitute" in response.lower(),
            "length_appropriate": 100 <= len(response) <= 2000,
            "mentions_consultation": bool(re.search(r'consult|physician|doctor|healthcare provider', response, re.IGNORECASE))
        }
        
        quality_metrics["quality_score"] = sum(quality_metrics.values()) / len(quality_metrics)
        
        return quality_metrics


def validate_medical_query(query: str) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to validate a medical query.
    
    Args:
        query: User query
    
    Returns:
        Tuple of (is_valid, warning_message)
    """
    checker = MedicalSafetyChecker()
    is_valid, warning, _ = checker.validate_medical_query(query)
    return is_valid, warning

