# src/enterprise_ai_cache/cache/semantic_cache.py
"""Semantic caching for LLM responses using Redis Stack vector similarity."""

from typing import Optional, Union, List, Callable, Any, Dict
import json
import numpy as np
import redis
from enterprise_ai_cache.config.settings import Settings, get_settings
from enterprise_ai_cache.exceptions import CacheError, ConnectionError, ValidationError


class SemanticCache:
    """Caches LLM responses based on semantic similarity of prompts.

    Uses Redis Stack (RediSearch) for vector similarity search with
    configurable distance threshold for cache hits.
    """

    INDEX_SUFFIX = ":vec_idx"
    PROMPTS_SUFFIX = ":prompts"
    RESPONSE_SUFFIX = ":response"

    def __init__(
        self,
        name: str = "semantic_cache",
        embedding_method: Optional[Callable[[Union[str, List[str]]], Any]] = None,
        ttl: int = 86400,
        distance_threshold: float = 1.0,
        dimensions: int = 768,
        settings: Optional[Settings] = None,
        redis_url: Optional[str] = None,
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        embedding_util: Optional[Any] = None,
    ):
        """Initialize the semantic cache.

        Args:
            name: Cache instance name (used as prefix for keys).
            embedding_method: Function that converts text to embedding vectors.
            ttl: Time-to-live for cache entries in seconds.
            distance_threshold: Maximum distance for a cache hit.
            dimensions: Embedding vector dimensions.
            settings: Configuration settings.
            redis_url: Redis host address.
            redis_port: Redis port.
            redis_password: Redis password.
            embedding_util: EmbeddingUtility instance (alternative to embedding_method).
        """
        if distance_threshold < 0:
            raise ValidationError("distance_threshold must be non-negative")

        self.name = name
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        if embedding_util is not None:
            self.embedding_method = getattr(embedding_util, 'embed', embedding_util)
        else:
            self.embedding_method = embedding_method or self._default_embedding
        self.dimensions = dimensions
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

        self._index_name = f"{name}{self.INDEX_SUFFIX}"
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Ensure the vector index exists in Redis."""
        try:
            self._redis.ft(self._index_name).info()
        except redis.ResponseError:
            try:
                from redis.commands.search.field import VectorField, TextField, TagField
                try:
                    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
                except ImportError:
                    from redis.commands.search.index_definition import IndexDefinition, IndexType

                vector_attrs = {
                    "DIM": self.dimensions,
                    "TYPE": "FLOAT32",
                    "DISTANCE_METRIC": "COSINE",
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                    "EF_RUNTIME": 200
                }

                schema_fields = [
                    VectorField(
                        name="vector",
                        algorithm="HNSW",
                        attributes=vector_attrs
                    ),
                    TagField(name="prompt", sortable=True),
                    TextField(name="response"),
                ]

                definition = IndexDefinition(
                    prefix=[f"{self.name}{self.PROMPTS_SUFFIX}:"],
                    index_type=IndexType.HASH
                )

                self._redis.ft(self._index_name).create_index(
                    fields=schema_fields,
                    definition=definition
                )
            except Exception:
                pass

    def store(self, prompt: Union[str, List[str]], response: Union[str, List[str]]) -> bool:
        """Store a prompt-response pair in the cache.

        Args:
            prompt: User prompt(s).
            response: LLM response(s).

        Returns:
            True if stored successfully.
        """
        if isinstance(prompt, str):
            prompt = [prompt]
            response = [response]

        if len(prompt) != len(response):
            raise ValidationError("prompt and response must have the same length")

        try:
            pipe = self._redis.pipeline()
            embeddings = self.embedding_method(prompt)

            for q, a, emb in zip(prompt, response, embeddings):
                key = f"{self.name}{self.PROMPTS_SUFFIX}:{q}"
                vector_bytes = np.array(emb, dtype=np.float32).tobytes()

                pipe.hset(key, mapping={
                    "vector": vector_bytes,
                    "prompt": q,
                    "response": a,
                })
                pipe.expire(key, self.ttl)

            pipe.execute()
            return True
        except Exception as e:
            raise CacheError(f"Failed to store in cache: {e}")

    def check(self, prompt: str, top_k: int = 5) -> Optional[List[Dict[str, Any]]]:
        """Check if a similar prompt exists in the cache.

        Args:
            prompt: User prompt to check.
            top_k: Number of similar results to retrieve.

        Returns:
            List of matching results with prompt, response, and distance,
            or None if no match above threshold.
        """
        try:
            query_embedding = self.embedding_method([prompt])
            query_bytes = np.array(query_embedding, dtype=np.float32).tobytes()

            # 使用 execute_command 执行 FT.SEARCH（redis-py 5.x 兼容）
            return_fields_list = ["prompt", "response", "score"]

            result = self._redis.execute_command(
                'FT.SEARCH', self._index_name,
                f'*=>[KNN {top_k} @vector $vec AS score]',
                'PARAMS', '2', 'vec', query_bytes,
                'RETURN', str(len(return_fields_list)), *return_fields_list,
                'DIALECT', '2'
            )

            # 解析结果
            matches = []

            if isinstance(result, dict):
                results_key = b'results' if b'results' in result else ('results' if 'results' in result else None)
                if results_key:
                    results_list = result[results_key]

                    for item in results_list:
                        extra_attrs = item.get(b'extra_attributes', {})

                        # 获取距离分数
                        score_bytes = extra_attrs.get(b'score')
                        if score_bytes:
                            score_str = score_bytes.decode() if isinstance(score_bytes, bytes) else score_bytes
                            distance = float(score_str)
                        else:
                            continue

                        # 检查是否在阈值内
                        if distance <= self.distance_threshold:
                            # 获取 prompt 和 response
                            prompt_val = extra_attrs.get(b'prompt', b'').decode() if isinstance(
                                extra_attrs.get(b'prompt'), bytes) else extra_attrs.get(b'prompt', '')
                            response_val = extra_attrs.get(b'response', b'').decode() if isinstance(
                                extra_attrs.get(b'response'), bytes) else extra_attrs.get(b'response', '')

                            matches.append({
                                "prompt": prompt_val,
                                "response": response_val,
                                "distance": distance,
                            })

            if not matches:
                return None

            matches.sort(key=lambda x: x["distance"])
            return matches

        except Exception as e:
            raise CacheError(f"Failed to check cache: {e}")

    def get(self, prompt: str) -> Optional[str]:
        """Get exact response for a prompt (no similarity search).

        Args:
            prompt: User prompt.

        Returns:
            Cached response if found, None otherwise.
        """
        try:
            key = f"{self.name}{self.PROMPTS_SUFFIX}:{prompt}"
            data = self._redis.hget(key, "response")
            return data.decode() if data else None
        except Exception as e:
            raise CacheError(f"Failed to get from cache: {e}")

    def delete(self, prompt: str) -> bool:
        """Delete a specific prompt from the cache.

        Args:
            prompt: Prompt to delete.

        Returns:
            True if deleted successfully.
        """
        try:
            key = f"{self.name}{self.PROMPTS_SUFFIX}:{prompt}"
            self._redis.delete(key)
            return True
        except Exception as e:
            raise CacheError(f"Failed to delete from cache: {e}")

    def clear(self) -> bool:
        """Clear all entries from the cache.

        Returns:
            True if cleared successfully.
        """
        try:
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match=f"{self.name}{self.PROMPTS_SUFFIX}:*", count=100)
                if keys:
                    self._redis.delete(*keys)
                if cursor == 0:
                    break
            return True
        except Exception as e:
            raise CacheError(f"Failed to clear cache: {e}")

    def count(self) -> int:
        """Get the number of cached entries.

        Returns:
            Number of cached prompt-response pairs.
        """
        try:
            cursor = 0
            count = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match=f"{self.name}{self.PROMPTS_SUFFIX}:*", count=100)
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
        try:
            info = self._redis.ft(self._index_name).info()
            return {
                "name": self.name,
                "count": self.count(),
                "distance_threshold": self.distance_threshold,
                "ttl": self.ttl,
                "dimensions": self.dimensions,
            }
        except Exception:
            return {
                "name": self.name,
                "count": self.count(),
                "distance_threshold": self.distance_threshold,
                "ttl": self.ttl,
                "dimensions": self.dimensions,
            }
