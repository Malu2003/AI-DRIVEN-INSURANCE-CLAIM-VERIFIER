"""Predictor interface for ICD prediction models.

Defines a minimal API so models (TF-IDF, ClinicalBERT) can be swapped
without changing downstream code.
"""
from typing import Protocol, Dict


class Predictor(Protocol):
    def predict_proba(self, text: str) -> Dict[str, float]:
        """Return a mapping ICD->probability for the given text."""
        ...
