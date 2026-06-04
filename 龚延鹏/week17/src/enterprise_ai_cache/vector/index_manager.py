# src/enterprise_ai_cache/vector/index_manager.py
"""Vector index management for Redis Stack."""

from typing import Optional, List, Dict, Any
import redis
from enterprise_ai_cache.config.settings import Settings, get_settings
from enterprise_ai_cache.exceptions import VectorIndexError, ConnectionError


class VectorIndexManager:
    """Manages vector indexes in Redis Stack using RedisSearch."""

    INDEX_PREFIX = "vec:"

    def __init__(self, settings: Optional[Settings] = None, redis_client: Optional[redis.Redis] = None):
        """Initialize the index manager.

        Args:
            settings: Configuration settings. Uses default if not provided.
            redis_client: Optional pre-configured Redis client.
        """
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

    def create_index(
        self,
        index_name: str,
        dimensions: int,
        metric: str = "COSINE",
        index_type: str = "HNSW",
        fields: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Create a vector index.

        Args:
            index_name: Name of the index.
            dimensions: Vector embedding dimensions.
            metric: Distance metric (COSINE, L2, IP).
            index_type: Index type (HNSW, FLAT, IVF).
            fields: Additional fields to index.

        Returns:
            True if index was created successfully.
        """
        key = f"{self.INDEX_PREFIX}{index_name}"
        '''
        field_defs = [
            {"name": "vector", "type": "VECTOR", "dims": dimensions, "metric": metric, "algorithm": index_type},
            {"name": "id", "type": "TAG"},
            {"name": "metadata", "type": "TEXT"},
        ]

        if fields:
            field_defs.extend(fields)

        try:
            ft_commands = [
                "FT.CREATE", key,
                "ON", "HASH",
                "SCHEMA",
            ]
            for field in field_defs:
                # if field["type"] == "VECTOR":
                #     num_params = 4
                #     ft_commands.extend([
                #         field["name"], "VECTOR", field["algorithm"],
                #         num_params,
                #         "dims", field["dims"],
                #         "type", "FLOAT32",
                #         "distance_metric", field["metric"]
                #     ])
                if field["type"] == "VECTOR":
                    # 构建向量字段定义
                    vector_params = [
                        field["name"],
                        "VECTOR",
                        field["algorithm"].upper(),
                        6,
                        "DIM",
                        field["dims"],  # 维度数（整数）
                        "FLOAT32",  # 数据类型
                        "TYPE",
                        field["metric"].upper(),  # 距离度量
                        "DISTANCE_METRIC"
                    ]

                    # 添加 HNSW 特定参数（可选）
                    if field["algorithm"].upper() == "HNSW":
                        vector_params.extend([
                            "M", 16,
                            "EF_CONSTRUCTION", 200,
                            "EF_RUNTIME", 200
                        ])

                    ft_commands.extend(vector_params)
                elif field["type"] == "TAG":
                    ft_commands.extend([field["name"], "TAG"])
                elif field["type"] == "TEXT":
                    ft_commands.extend([field["name"], "TEXT"])

            self._redis.execute_command(*ft_commands)
            return True
        except Exception as e:
            raise VectorIndexError(f"Failed to create index: {e}")
            '''
        try:
            from redis.commands.search.field import VectorField, TextField, TagField
            try:
                from redis.commands.search.indexDefinition import IndexDefinition, IndexType
            except ImportError:
                from redis.commands.search.index_definition import IndexDefinition, IndexType
                # 定义向量字段
            vector_attributes = {
                "DIM": dimensions,
                "TYPE": "FLOAT32",
                "DISTANCE_METRIC": metric.upper()
            }

            # 如果是 HNSW，添加额外参数
            if index_type.upper() == "HNSW":
                vector_attributes.update({
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                    "EF_RUNTIME": 200
                })

            # 创建字段列表
            schema_fields = [
                VectorField(
                    name="vector",
                    algorithm=index_type.upper(),
                    attributes=vector_attributes
                ),
                TagField(name="id"),
                TextField(name="metadata"),
                TextField(name="_text_meta")  # 添加全文搜索元数据字段
            ]

            # 添加额外字段
            if fields:
                for extra_field in fields:
                    if extra_field.get("type") == "TAG":
                        schema_fields.append(TagField(name=extra_field["name"]))
                    elif extra_field.get("type") == "TEXT":
                        schema_fields.append(TextField(name=extra_field["name"]))

            # 创建索引定义
            definition = IndexDefinition(
                prefix=[f"doc:{index_name}:"],
                index_type=IndexType.HASH
            )

            # 创建索引
            self._redis.ft(key).create_index(
                fields=schema_fields,
                definition=definition
            )
            return True
        except Exception as e:
            raise VectorIndexError(f"Failed to create index: {e}")
    def delete_index(self, index_name: str) -> bool:
        """Delete a vector index.

        Args:
            index_name: Name of the index to delete.

        Returns:
            True if index was deleted successfully.
        """
        key = f"{self.INDEX_PREFIX}{index_name}"
        try:
            self._redis.execute_command("FT.DROPINDEX", key)
            return True
        except Exception as e:
            raise VectorIndexError(f"Failed to delete index: {e}")

    def index_exists(self, index_name: str) -> bool:
        """Check if an index exists.

        Args:
            index_name: Name of the index.

        Returns:
            True if index exists.
        """
        key = f"{self.INDEX_PREFIX}{index_name}"
        try:
            self._redis.execute_command("FT.INFO", key)
            return True
        except redis.ResponseError:
            return False

    def list_indexes(self) -> List[str]:
        """List all vector indexes.

        Returns:
            List of index names.
        """
        try:
            result = self._redis.execute_command("FT._LIST")
            prefix_bytes = self.INDEX_PREFIX.encode() if isinstance(self.INDEX_PREFIX, str) else self.INDEX_PREFIX
            return [idx.decode() if isinstance(idx, bytes) else idx for idx in result if (idx.decode() if isinstance(idx, bytes) else idx).startswith(prefix_bytes.decode() if isinstance(prefix_bytes, bytes) else prefix_bytes)]
        except Exception as e:
            raise VectorIndexError(f"Failed to list indexes: {e}")
