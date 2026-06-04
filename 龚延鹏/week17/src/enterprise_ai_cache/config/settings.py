# src/enterprise_ai_cache/config/settings.py
"""Configuration management for Enterprise AI Cache SDK."""

from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field


class RedisSettings(BaseModel):
    """Redis connection settings."""

    host: str = Field(default="localhost", description="Redis host address")
    port: int = Field(default=6379, description="Redis port")
    password: Optional[str] = Field(default=None, description="Redis password")
    db: int = Field(default=0, description="Redis database number")
    ssl: bool = Field(default=False, description="Enable SSL connection")
    username: Optional[str] = Field(default=None, description="Redis username")

    @property
    def url(self) -> str:
        """Build Redis URL from components."""
        scheme = "rediss" if self.ssl else "redis"
        auth = f"{self.username}:{self.password}@" if self.username and self.password else ""
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"


class VectorSettings(BaseModel):
    """Vector index settings."""

    default_dimensions: int = Field(default=768, description="Default embedding dimensions")
    default_metric: str = Field(default="COSINE", description="Default distance metric: COSINE, L2, IP")
    index_type: str = Field(default="HNSW", description="Index type: HNSW, FLAT, IVF")


class CacheSettings(BaseModel):
    """Cache behavior settings."""

    default_ttl: int = Field(default=86400, description="Default TTL in seconds (24 hours)")
    semantic_threshold: float = Field(default=0.1, description="Semantic similarity threshold for cache hit")
    max_results: int = Field(default=100, description="Maximum number of cache results to return")


class Settings(BaseModel):
    """Main settings container."""

    redis: RedisSettings = Field(default_factory=RedisSettings)
    vector: VectorSettings = Field(default_factory=VectorSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)

    model_config = {"extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance (singleton)."""
    return Settings()
