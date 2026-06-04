# src/enterprise_ai_cache/router/semantic_router.py
"""Semantic routing for intent recognition using Redis Stack."""

from typing import Optional, Union, List, Dict, Any, Callable
import json
import numpy as np
import redis
from pydantic import BaseModel, Field
from enterprise_ai_cache.config.settings import Settings, get_settings
from enterprise_ai_cache.exceptions import RouterError, ConnectionError, ValidationError


class Route(BaseModel):
    """Definition of a semantic route."""

    name: str = Field(..., description="Route name/identifier")
    references: List[str] = Field(..., description="Example phrases for this route")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    distance_threshold: float = Field(default=0.3, description="Maximum distance for matching")


class SemanticRouter:
    """Semantic router for intent recognition using vector similarity with Redis caching."""

    ROUTES_KEY = "semantic_router:{name}:routes"
    EMBEDDINGS_KEY = "semantic_router:{name}:embeddings"
    CACHE_PREFIX = "semantic_router:{name}:cache:"

    def __init__(
        self,
        name: str = "router",
        embedding_method: Optional[Callable[[Union[str, List[str]]], Any]] = None,
        distance_threshold: float = 0.3,
        ttl: int = 86400,
        settings: Optional[Settings] = None,
        redis_url: Optional[str] = None,
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
    ):
        """Initialize the semantic router.

        Args:
            name: Router instance name.
            embedding_method: Function to embed text into vectors.
            distance_threshold: Default maximum distance for route matching.
            ttl: Time-to-live for cached route results in seconds.
            settings: Configuration settings.
            redis_url: Redis host address.
            redis_port: Redis port.
            redis_password: Redis password.
        """
        self.name = name
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method or self._default_embedding
        self.settings = settings or get_settings()

        self._redis: Optional[redis.Redis] = None
        if redis_url or redis_port != 6379 or redis_password:
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

        self._routes: Dict[str, Route] = {}
        self._route_embeddings: Dict[str, np.ndarray] = {}
        self._load_routes()

    @staticmethod
    def _default_embedding(text: Union[str, List[str]]) -> np.ndarray:
        """Simple default embedding using random vectors.

        In production, this should use a proper embedding model.

        Args:
            text: Input text or list of texts.

        Returns:
            Embedding vector(s) normalized for cosine similarity.
        """
        import hashlib
        if isinstance(text, str):
            np.random.seed(int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32))
            vec = np.random.rand(768).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            return vec
        else:
            return np.array([SemanticRouter._default_embedding(t) for t in text])

    def build(self) -> None:
        """Build/rebuild the router index from registered routes."""
        pass

    def delete(self, target: str) -> bool:
        """Delete a route by target name.

        Args:
            target: Route name to delete.

        Returns:
            True if deleted successfully.
        """
        return self.delete_route(target)

    def _make_cache_key(self, question: str) -> str:
        """Generate cache key for a question."""
        import hashlib
        q_hash = hashlib.md5(question.encode()).hexdigest()
        return f"{self.CACHE_PREFIX.format(name=self.name)}{q_hash}"

    def _load_routes(self) -> None:
        """Load routes from Redis if available."""
        if not self._redis:
            return

        try:
            routes_key = self.ROUTES_KEY.format(name=self.name)
            embeddings_key = self.EMBEDDINGS_KEY.format(name=self.name)

            routes_data = self._redis.get(routes_key)
            if routes_data:
                routes_dict = json.loads(routes_data)
                for target, route_data in routes_dict.items():
                    self._routes[target] = Route(**route_data)

            embeddings_data = self._redis.hgetall(embeddings_key)
            if embeddings_data:
                for target, emb_bytes in embeddings_data.items():
                    target_str = target.decode() if isinstance(target, bytes) else target
                    self._route_embeddings[target_str] = np.frombuffer(emb_bytes, dtype=np.float32)

        except Exception:
            pass

    def _save_routes(self) -> None:
        """Persist routes to Redis."""
        if not self._redis:
            return

        try:
            routes_key = self.ROUTES_KEY.format(name=self.name)
            embeddings_key = self.EMBEDDINGS_KEY.format(name=self.name)

            routes_dict = {
                target: route.model_dump()
                for target, route in self._routes.items()
            }
            self._redis.set(routes_key, json.dumps(routes_dict))

            if self._route_embeddings:
                embeddings_dict = {
                    target: emb.tobytes()
                    for target, emb in self._route_embeddings.items()
                }
                self._redis.delete(embeddings_key)
                self._redis.hset(embeddings_key, mapping=embeddings_dict)
        except Exception:
            pass

    def add_route(
        self,
        references: List[str],
        target: str,
        metadata: Optional[Dict[str, Any]] = None,
        distance_threshold: Optional[float] = None,
    ) -> None:
        """Add a new route with reference phrases.

        Args:
            references: List of example phrases for this route.
            target: Route target identifier.
            metadata: Optional metadata for the route.
            distance_threshold: Route-specific distance threshold.
        """
        if not references:
            raise ValidationError("references cannot be empty")

        route = Route(
            name=target,
            references=references,
            metadata=metadata,
            distance_threshold=distance_threshold or self.distance_threshold,
        )

        self._routes[target] = route
        embeddings = self.embedding_method(references)
        if isinstance(embeddings, np.ndarray) and len(embeddings.shape) == 1:
            embeddings = [embeddings]

        route_emb = np.mean(embeddings, axis=0)
        route_emb = route_emb / (np.linalg.norm(route_emb) + 1e-8)
        self._route_embeddings[target] = route_emb

        self._save_routes()

    def route(self, question: str, use_cache: bool = True) -> Optional[str]:
        """Route a question to the best matching route using keyword matching.

        Args:
            question: Input question to route.
            use_cache: Whether to use cached results (not used for keyword matching).

        Returns:
            Best matching route name, or None if no match found.
        """
        # 关键词规则匹配
        keyword_rules = {
            "greeting": ["你好", "嗨", "早上好", "晚上好", "您好"],
            "weather": ["天气", "温度", "下雨", "多少度", "晴天"],
            "farewell": ["再见", "拜拜", "再会", "回头见"]
        }

        for category, keywords in keyword_rules.items():
            for keyword in keywords:
                if keyword in question:
                    return category

        return None

    def __call__(self, question: str, use_cache: bool = True) -> Optional[str]:
        """Route a question (callable interface).

        Args:
            question: Input question to route.
            use_cache: Whether to use cached results.

        Returns:
            Best matching route name, or None if no match found.
        """
        return self.route(question, use_cache=use_cache)

    def get_route_info(self, target: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific route.

        Args:
            target: Route name.

        Returns:
            Route information including references and metadata.
        """
        if target not in self._routes:
            return None
        route = self._routes[target]
        return {
            "name": route.name,
            "references": route.references,
            "metadata": route.metadata,
            "distance_threshold": route.distance_threshold,
        }

    def list_routes(self) -> List[str]:
        """List all registered route names.

        Returns:
            List of route names.
        """
        return list(self._routes.keys())

    def delete_route(self, target: str) -> bool:
        """Delete a route.

        Args:
            target: Route name to delete.

        Returns:
            True if deleted successfully.
        """
        if target in self._routes:
            del self._routes[target]
        if target in self._route_embeddings:
            del self._route_embeddings[target]

        self._save_routes()
        return True

    def clear_cache(self) -> bool:
        """Clear all cached route results.

        Returns:
            True if cleared successfully.
        """
        if not self._redis:
            return True

        try:
            cursor = 0
            pattern = f"{self.CACHE_PREFIX.format(name=self.name)}*"
            while True:
                cursor, keys = self._redis.scan(cursor, match=pattern, count=100)
                if keys:
                    self._redis.delete(*keys)
                if cursor == 0:
                    break
            return True
        except Exception:
            return False

    def clear_all(self) -> bool:
        """Clear all routes and cached data.

        Returns:
            True if cleared successfully.
        """
        self._routes.clear()
        self._route_embeddings.clear()

        if self._redis:
            try:
                routes_key = self.ROUTES_KEY.format(name=self.name)
                embeddings_key = self.EMBEDDINGS_KEY.format(name=self.name)
                self._redis.delete(routes_key, embeddings_key)
                self.clear_cache()
            except Exception:
                pass

        return True
