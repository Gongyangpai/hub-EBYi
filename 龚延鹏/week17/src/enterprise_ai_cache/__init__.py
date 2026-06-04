# src/enterprise_ai_cache/__init__.py
"""Enterprise AI Cache SDK.

A unified vector retrieval and intelligent caching service platform
based on Redis Stack for enterprise internal use.
"""

from .__version__ import __version__
from .config.settings import Settings, get_settings
from .exceptions import (
    EnterpriseAICacheError,
    ConnectionError,
    CacheError,
    VectorIndexError,
    ValidationError,
)

__all__ = [
    "__version__",
    "Settings",
    "get_settings",
    "EnterpriseAICacheError",
    "ConnectionError",
    "CacheError",
    "VectorIndexError",
    "ValidationError",
]
