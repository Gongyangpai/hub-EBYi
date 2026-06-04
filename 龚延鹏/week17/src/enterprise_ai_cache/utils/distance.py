# src/enterprise_ai_cache/utils/distance.py
"""Distance computation utilities for vector similarity."""

from enum import Enum
from typing import Union
import numpy as np


class DistanceMetric(str, Enum):
    """Supported distance metrics."""

    COSINE = "cosine"
    L2 = "l2"
    IP = "ip"


def compute_distance(
    vec1: np.ndarray,
    vec2: np.ndarray,
    metric: Union[str, DistanceMetric] = DistanceMetric.COSINE,
) -> float:
    """Compute distance between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.
        metric: Distance metric to use.

    Returns:
        Distance value (lower = more similar for most metrics).
    """
    if isinstance(metric, str):
        metric = DistanceMetric(metric.lower())

    vec1 = np.asarray(vec1).flatten()
    vec2 = np.asarray(vec2).flatten()

    if vec1.shape != vec2.shape:
        raise ValueError(f"Vector dimensions must match: {vec1.shape} vs {vec2.shape}")

    if metric == DistanceMetric.L2:
        return float(np.linalg.norm(vec1 - vec2))

    elif metric == DistanceMetric.COSINE:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return float(1 - np.dot(vec1, vec2) / (norm1 * norm2))

    elif metric == DistanceMetric.IP:
        return float(1 - np.dot(vec1, vec2))

    else:
        raise ValueError(f"Unsupported metric: {metric}")


def compute_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray,
    metric: Union[str, DistanceMetric] = DistanceMetric.COSINE,
) -> float:
    """Compute similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.
        metric: Similarity metric to use.

    Returns:
        Similarity value (higher = more similar).
    """
    return 1.0 - compute_distance(vec1, vec2, metric)


def batch_compute_distances(
    query: np.ndarray,
    candidates: np.ndarray,
    metric: Union[str, DistanceMetric] = DistanceMetric.COSINE,
) -> np.ndarray:
    """Compute distances from query to multiple candidate vectors.

    Args:
        query: Query vector (1D).
        candidates: Candidate vectors (2D, where each row is a vector).

    Returns:
        Array of distances.
    """
    if isinstance(metric, str):
        metric = DistanceMetric(metric.lower())

    candidates = np.asarray(candidates)

    if metric == DistanceMetric.L2:
        return np.linalg.norm(candidates - query, axis=1)

    elif metric == DistanceMetric.COSINE:
        query_norm = np.linalg.norm(query)
        norms = np.linalg.norm(candidates, axis=1)
        dots = np.dot(candidates, query)
        return 1 - dots / (norms * query_norm + 1e-8)

    elif metric == DistanceMetric.IP:
        return 1 - np.dot(candidates, query)

    else:
        raise ValueError(f"Unsupported metric: {metric}")
