# src/enterprise_ai_cache/vector/schema.py
"""Schema definitions for vector indexes."""

from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class FieldType(str, Enum):
    """Supported field types for vector indexes."""

    VECTOR = "VECTOR"
    TAG = "TAG"
    TEXT = "TEXT"
    NUMERIC = "NUMERIC"
    GEOSHAPE = "GEOSHAPE"


class DistanceMetric(str, Enum):
    """Supported distance metrics for vector similarity."""

    COSINE = "COSINE"
    L2 = "L2"
    IP = "IP"


class IndexType(str, Enum):
    """Supported index types."""

    HNSW = "HNSW"
    FLAT = "FLAT"
    IVF = "IVF"


class FieldDefinition(BaseModel):
    """Definition of a field in a vector index schema."""

    name: str = Field(..., description="Field name")
    field_type: FieldType = Field(..., description="Type of the field")
    sortable: bool = Field(default=False, description="Whether the field is sortable")
    no_index: bool = Field(default=False, description="Whether to skip indexing")
    optional: bool = Field(default=False, description="Whether the field is optional")


class VectorSchema(BaseModel):
    """Schema definition for a vector index."""

    name: str = Field(..., description="Index name")
    dimensions: int = Field(..., description="Vector embedding dimensions")
    distance_metric: DistanceMetric = Field(default=DistanceMetric.COSINE)
    index_type: IndexType = Field(default=IndexType.HNSW)
    fields: List[FieldDefinition] = Field(default_factory=list)

    model_config = {"extra": "ignore"}

    def to_redis_schema(self) -> List[Dict[str, Any]]:
        """Convert schema to Redis search schema format.

        Returns:
            List of field definitions in Redis format.
        """
        schema = [
            {
                "name": "vector",
                "type": "VECTOR",
                "dims": self.dimensions,
                "metric": self.distance_metric.value,
                "algorithm": self.index_type.value,
            }
        ]

        for field in self.fields:
            schema.append({
                "name": field.name,
                "type": field.field_type.value,
                "sortable": field.sortable,
                "no_index": field.no_index,
                "optional": field.optional,
            })

        return schema
