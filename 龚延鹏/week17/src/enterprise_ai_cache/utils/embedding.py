# src/enterprise_ai_cache/utils/embedding.py
"""Embedding utility functions with caching support."""

from typing import Union, List, Optional, Callable, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from enterprise_ai_cache.cache import EmbeddingsCache


class EmbeddingUtility:
    """Utility class for text embedding operations with optional caching."""

    def __init__(
        self,
        embedding_fn: Optional[Callable[[Union[str, List[str]]], Any]] = None,
        cache: Optional["EmbeddingsCache"] = None,
        dimensions: int = 768,
    ):
        """Initialize the embedding utility.

        Args:
            embedding_fn: Function to generate embeddings.
            cache: Optional cache for embeddings.
            dimensions: Embedding vector dimensions.
        """
        self.embedding_fn = embedding_fn or self._default_embedding
        self.cache = cache
        self.dimensions = dimensions

    @staticmethod
    def _default_embedding(text: Union[str, List[str]]) -> np.ndarray:
        """Default embedding function (random vectors).

        In production, replace with actual embedding model.

        Args:
            text: Input text or list of texts.

        Returns:
            Embedding vector(s).
        """
        if isinstance(text, str):
            np.random.seed(hash(text) % (2**32))
            return np.random.rand(768).astype(np.float32)
        else:
            return np.array([
                EmbeddingUtility._default_embedding(t) for t in text
            ])

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text, with optional caching.

        Args:
            text: Input text.

        Returns:
            Embedding vector.
        """
        if self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                return cached

        embedding = self.embedding_fn(text)

        if self.cache is not None and isinstance(embedding, np.ndarray):
            self.cache.store(text, embedding)

        return embedding

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts, with optional caching.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        uncached_texts = []
        uncached_indices = []
        results = [None] * len(texts)

        if self.cache:
            exists_list = self.cache.exists(texts)
            for i, (text, exists) in enumerate(zip(texts, exists_list)):
                if exists:
                    cached = self.cache.get(text)
                    if cached is not None:
                        results[i] = cached
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        if uncached_texts:
            embeddings = self.embedding_fn(uncached_texts)

            if isinstance(embeddings, np.ndarray) and len(embeddings.shape) > 1:
                for idx, emb in zip(uncached_indices, embeddings):
                    results[idx] = emb
            elif isinstance(embeddings, list):
                for idx, emb in zip(uncached_indices, embeddings):
                    results[idx] = emb
            else:
                for idx, emb in zip(uncached_indices, [embeddings]):
                    results[idx] = emb

            if self.cache:
                self.cache.store(uncached_texts, embeddings if isinstance(embeddings, np.ndarray) else np.array(embeddings))

        return [r for r in results if r is not None]

    def embed_with_cache_check(self, text: str) -> tuple[np.ndarray, bool]:
        """Generate embedding and indicate if it was cached.

        Args:
            text: Input text.

        Returns:
            Tuple of (embedding, was_cached).
        """
        was_cached = False

        if self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                return cached, True

        embedding = self.embedding_fn(text)
        was_cached = False

        if self.cache is not None:
            self.cache.store(text, embedding)

        return embedding, was_cached
