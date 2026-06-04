"""Hybrid search engine combining vector similarity with metadata filtering and full-text search."""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import redis
from enterprise_ai_cache.config.settings import Settings, get_settings
from enterprise_ai_cache.exceptions import VectorIndexError, ConnectionError, ValidationError


class HybridSearchEngine:
    """Hybrid search combining vector similarity, metadata filters, and full-text search."""
    INDEX_PREFIX = "vec:"

    def __init__(self, index_name: str, settings: Optional[Settings] = None,
                 redis_client: Optional[redis.Redis] = None):
        """Initialize the hybrid search engine.

        Args:
            index_name: Name of the vector index.
            settings: Configuration settings.
            redis_client: Optional pre-configured Redis client.
        """
        self.index_name = index_name
        self.full_index_name = f"{self.INDEX_PREFIX}{index_name}"
        self.settings = settings or get_settings()

        if redis_client:
            self._redis = redis_client
        else:
            try:
                self._redis = redis.Redis(
                    host=self.settings.redis.host,
                    port=self.settings.redis.port,
                    password=self.settings.redis.password,
                    db=self.settings.redis.db,
                    decode_responses=False,
                )
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Redis: {e}")

    def add(
            self,
            id: str,
            vector: Union[np.ndarray, List[float]],
            metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a vector to the index.

        Args:
            id: Unique identifier for the vector.
            vector: Embedding vector.
            metadata: Optional metadata dictionary.

        Returns:
            True if added successfully.
        """
        key = f"doc:{self.index_name}:{id}"

        # 🔥 修复点 1：存入时做归一化（和你的路由保持一致）
        vec_np = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec_np) + 1e-8
        vec_np = vec_np / norm
        vector_bytes = vec_np.tobytes()

        try:
            mapping = {
                "vector": vector_bytes,
                "id": id,
            }
            if metadata:
                for field_name, field_value in metadata.items():
                    mapping[field_name] = str(field_value) if not isinstance(field_value, str) else field_value
                mapping["_text_meta"] = " ".join([str(v) for v in metadata.values()])

            self._redis.hset(key, mapping=mapping)
            return True
        except Exception as e:
            raise VectorIndexError(f"Failed to add vector: {e}")

    def search(
            self,
            query_vector: Union[np.ndarray, List[float]],
            top_k: int = 10,
            vector_filter: Optional[Dict[str, Any]] = None,
            text_query: Optional[str] = None,
            return_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors with optional metadata filters and full-text query.

        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            vector_filter: Metadata filters (e.g., {"category": "books", "price": ">10"}).
            text_query: Full-text search query string.
            return_metadata: Whether to return metadata.

        Returns:
            List of search results with scores and optional metadata.
        """
        # 🔥 修复点 2：查询向量也归一化！（必须和存入一致）
        vec_np = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(vec_np) + 1e-8
        vec_np = vec_np / norm
        query_bytes = vec_np.tobytes()

        query_parts = []

        if text_query:
            escaped_query = text_query.replace('"', '\\"')
            query_parts.append(f'(@_text_meta:{escaped_query} | @id:{escaped_query})')

        if vector_filter:
            filter_conditions = self._build_filter_conditions(vector_filter)
            if filter_conditions:
                query_parts.append(filter_conditions)

        base_query = f"*=>[KNN {top_k} @vector $vec AS score]"

        if query_parts:
            combined = " AND ".join(query_parts)
            search_query = f"({combined})=>[KNN {top_k} @vector $vec AS score]"
        else:
            search_query = base_query

        try:
            # 构建 RETURN 字段
            return_fields_list = ["id", "score"]
            if return_metadata:
                return_fields_list.extend(["metadata", "_text_meta"])

            # 使用 execute_command（redis-py 5.x 兼容）
            result = self._redis.execute_command(
                'FT.SEARCH', self.full_index_name,
                search_query,
                'PARAMS', '2', 'vec', query_bytes,
                'RETURN', str(len(return_fields_list)), *return_fields_list,
                'DIALECT', '2'
            )

            # 解析结果 - 支持多种格式
            docs = []

            # 格式 1: 字典格式（redis-py 5.x）
            if isinstance(result, dict):
                results_key = b'results' if b'results' in result else ('results' if 'results' in result else None)
                if results_key:
                    results_list = result[results_key]

                    for item in results_list:
                        # 获取文档 ID
                        doc_id_bytes = item.get(b'id')

                        if doc_id_bytes:
                            doc_id = doc_id_bytes.decode() if isinstance(doc_id_bytes, bytes) else doc_id_bytes
                        else:
                            continue

                        # 获取额外属性
                        extra_attrs = item.get(b'extra_attributes', {})

                        # 获取 score
                        score_bytes = extra_attrs.get(b'score')
                        if score_bytes:
                            score_str = score_bytes.decode() if isinstance(score_bytes, bytes) else score_bytes
                            score = float(score_str)
                        else:
                            score = 0.0

                        doc_data = {"id": doc_id, "score": score}

                        if return_metadata:
                            metadata = {}
                            for k, v in extra_attrs.items():
                                key = k.decode() if isinstance(k, bytes) else k
                                if key not in ['id', 'score']:
                                    val = v.decode() if isinstance(v, bytes) else v
                                    metadata[key] = val
                            doc_data["metadata"] = metadata

                        docs.append(doc_data)

            # 格式 2: 列表格式（旧版本兼容）
            elif isinstance(result, list) and len(result) > 0:
                total = result[0]
                for i in range(1, len(result), 2):
                    doc_id = result[i]
                    if isinstance(doc_id, bytes):
                        doc_id = doc_id.decode()

                    fields = result[i + 1] if i + 1 < len(result) else []

                    doc_data = {"id": doc_id, "score": 0.0}
                    metadata = {}

                    for j in range(0, len(fields), 2):
                        field_name = fields[j].decode() if isinstance(fields[j], bytes) else fields[j]
                        field_value = fields[j + 1].decode() if isinstance(fields[j + 1], bytes) else fields[j + 1]

                        if field_name == 'score':
                            doc_data["score"] = float(field_value)
                        elif field_name not in ['id']:
                            metadata[field_name] = field_value

                    if return_metadata:
                        doc_data["metadata"] = metadata

                    docs.append(doc_data)

            return docs

        except redis.ResponseError as e:
            if "no such index" in str(e).lower():
                raise VectorIndexError(
                    f"Index '{self.index_name}' does not exist. Create it with VectorIndexManager first.")
            raise VectorIndexError(f"Search failed: {e}")
        except Exception as e:
            raise VectorIndexError(f"Search failed: {e}")

    def _build_filter_conditions(self, filters: Dict[str, Any]) -> str:
        conditions = []
        for field, value in filters.items():
            if isinstance(value, dict):
                for op, val in value.items():
                    conditions.append(f"@{field}{op}{val}")
            else:
                conditions.append(f"@{field}:{value}")
        return " AND ".join(conditions)

    def text_search(
            self,
            query: str,
            fields: Optional[List[str]] = None,
            top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Perform pure full-text search without vector similarity."""
        try:
            from redis.commands.search.query import Query

            search_query = query
            if fields:
                field_query = " | ".join([f"@{f}" for f in fields])
                search_query = f"({field_query})({query})"

            query_obj = Query(search_query) \
                .return_fields("id", "_text_meta") \
                .dialect(2)

            results = self._redis.ft(self.full_index_name).search(query_obj)

            return [
                {
                    "id": doc.id,
                    "metadata": doc.metadata if hasattr(doc, "metadata") else None,
                }
                for doc in results.docs
            ]
        except Exception as e:
            raise VectorIndexError(f"Text search failed: {e}")

    def delete(self, id: str) -> bool:
        key = f"doc:{self.index_name}:{id}"
        try:
            self._redis.delete(key)
            return True
        except Exception as e:
            raise VectorIndexError(f"Failed to delete vector: {e}")

    def count(self) -> int:
        try:
            info = self._redis.ft(self.full_index_name).info()
            return info.get("num_docs", 0)
        except Exception:
            return 0