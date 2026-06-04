# src/enterprise_ai_cache/utils/__init__.py
"""Utility module."""

from enterprise_ai_cache.utils.embedding import EmbeddingUtility
from enterprise_ai_cache.utils.distance import compute_distance, DistanceMetric

__all__ = ["EmbeddingUtility", "compute_distance", "DistanceMetric"]
