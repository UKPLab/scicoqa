"""
Evaluation utilities for SciCodeQA.

This module contains tools for evaluating and analyzing model performance
on discrepancy detection tasks.
"""

from .compute_recall import MODEL_TO_PRETTY, compute_recall, load_evaluations

__all__ = ["compute_recall", "load_evaluations", "MODEL_TO_PRETTY"]


