# src/enterprise_ai_cache/exceptions.py
"""Custom exceptions for Enterprise AI Cache SDK."""


class EnterpriseAICacheError(Exception):
    """Base exception for all SDK errors."""

    pass


class ConnectionError(EnterpriseAICacheError):
    """Raised when Redis connection fails."""

    pass


class CacheError(EnterpriseAICacheError):
    """Raised when cache operations fail."""

    pass


class VectorIndexError(EnterpriseAICacheError):
    """Raised when vector index operations fail."""

    pass


class ValidationError(EnterpriseAICacheError):
    """Raised when input validation fails."""

    pass


class EmbeddingError(EnterpriseAICacheError):
    """Raised when embedding operations fail."""

    pass


class RouterError(EnterpriseAICacheError):
    """Raised when semantic router operations fail."""

    pass
