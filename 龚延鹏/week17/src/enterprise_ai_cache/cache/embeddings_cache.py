# src/enterprise_ai_cache/cache/embeddings_cache.py
"""Embeddings caching to avoid redundant vectorization using Redis Stack."""

from typing import Optional, Union, List, Dict, Any
import hashlib
import numpy as np
import redis
from enterprise_ai_cache.config.settings import Settings, get_settings
from enterprise_ai_cache.exceptions import CacheError, ConnectionError, ValidationError


class EmbeddingsCache:
    """Caches text-to-embedding conversions using Redis.

    Uses MD5 hash of text as the cache key and stores embeddings
    as bytes in Redis with optional TTL.
    """

    def __init__(
        self,
        name: str = "embeddings",
        ttl: int = 86400,
        settings: Optional[Settings] = None,
        redis_url: Optional[str] = None,
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
    ):
        """Initialize the embeddings cache.

        Args:
            name: Cache instance name.
            ttl: Time-to-live for cache entries in seconds.
            settings: Configuration settings.
            redis_url: Redis host address.
            redis_port: Redis port.
            redis_password: Redis password.
        """
        self.name = name
        self.ttl = ttl
        self.settings = settings or get_settings()

        try:
            self._redis = redis.Redis(
                host=redis_url or self.settings.redis.host,
                port=redis_port,
                password=redis_password or self.settings.redis.password,
                db=self.settings.redis.db,
                decode_responses=False,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")

    def _make_key(self, text: str) -> str:
        """Generate cache key from text.

        Args:
            text: Input text.

        Returns:
            Cache key string.
        """
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{self.name}:{text_hash}"

    def _embed_to_bytes(self, embedding: np.ndarray) -> bytes:
        """Convert embedding array to bytes for Redis storage.

        Args:
            embedding: Embedding vector.

        Returns:
            Bytes representation of embedding.
        """
        return np.array(embedding, dtype=np.float32).tobytes()

    def _bytes_to_embed(self, data: bytes) -> np.ndarray:
        """Convert bytes back to embedding array.

        Args:
            data: Bytes from Redis.

        Returns:
            Embedding vector as numpy array.
        """
        return np.frombuffer(data, dtype=np.float32)

    def store(self, text: Union[str, List[str]], embedding: Union[np.ndarray, List[np.ndarray]]) -> bool:
        """Store embeddings in the cache.

        Args:
            text: Input text(s).
            embedding: Corresponding embedding vector(s).

        Returns:
            True if stored successfully.
        """
        if isinstance(text, str):
            text = [text]
            embedding = [embedding]

        try:
            pipe = self._redis.pipeline()
            for i, t in enumerate(text):
                key = self._make_key(t)
                value = self._embed_to_bytes(embedding[i])
                pipe.setex(key, self.ttl, value)
            pipe.execute()
            return True
        except Exception as e:
            raise CacheError(f"Failed to store embedding: {e}")

    def call(self, text: Union[str, List[str]]) -> Optional[List[np.ndarray]]:
        """Retrieve embeddings from the cache.

        Args:
            text: Input text(s) to look up.

        Returns:
            List of embeddings if found, None if cache miss.
        """
        if isinstance(text, str):
            text = [text]

        try:
            keys = [self._make_key(t) for t in text]
            results = self._redis.mget(keys)

            if not results or all(r is None for r in results):
                return None

            embeddings = []
            for result in results:
                if result is None:
                    embeddings.append(None)
                else:
                    embeddings.append(self._bytes_to_embed(result))

            return embeddings
        except Exception as e:
            raise CacheError(f"Failed to retrieve embedding: {e}")

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get a single embedding from the cache.

        Args:
            text: Input text.

        Returns:
            Embedding if found, None otherwise.
        """
        results = self.call([text])
        return results[0] if results else None

    def delete(self, text: Union[str, List[str]]) -> int:
        """Delete embeddings from the cache.

        Args:
            text: Input text(s) to delete.

        Returns:
            Number of keys deleted.
        """
        if isinstance(text, str):
            text = [text]

        try:
            keys = [self._make_key(t) for t in text]
            return self._redis.delete(*keys)
        except Exception as e:
            raise CacheError(f"Failed to delete embedding: {e}")

    def exists(self, text: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """Check if text has a cached embedding.

        Args:
            text: Input text(s) to check.

        Returns:
            True/False or list of True/False for each text.
        """
        if isinstance(text, str):
            key = self._make_key(text)
            return self._redis.exists(key) > 0

        keys = [self._make_key(t) for t in text]
        results = self._redis.exists(*keys)
        return [r > 0 for r in results]

    def clear(self) -> bool:
        """Clear all embeddings from this cache.

        Returns:
            True if cleared successfully.
        """
        try:
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match=f"{self.name}:*", count=100)
                if keys:
                    self._redis.delete(*keys)
                if cursor == 0:
                    break
            return True
        except Exception as e:
            raise CacheError(f"Failed to clear cache: {e}")

    def count(self) -> int:
        """Get the number of cached embeddings.

        Returns:
            Number of cached embeddings.
        """
        try:
            cursor = 0
            count = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match=f"{self.name}:*", count=100)
                count += len(keys)
                if cursor == 0:
                    break
            return count
        except Exception as e:
            raise CacheError(f"Failed to count cache entries: {e}")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        return {
            "name": self.name,
            "count": self.count(),
            "ttl": self.ttl,
        }
