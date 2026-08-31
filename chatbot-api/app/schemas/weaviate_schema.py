"""Schemas for Weaviate catalog endpoints (custom filter builder)."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class WeaviateCollectionInfo(BaseModel):
    name: str
    object_count: Optional[int] = Field(None, description="Approximate object count in Weaviate")


class WeaviateCollectionsResponse(BaseModel):
    collections: List[WeaviateCollectionInfo]
    cached: bool
    generated_at: datetime


class WeaviateFieldsResponse(BaseModel):
    collection: str
    fields: List[str]
    cached: bool
    generated_at: datetime


class WeaviateValuesResponse(BaseModel):
    collection: str
    field: str
    values: List[str]
    count: int = Field(..., description="Number of values returned in this page")
    total: int = Field(..., description="Total unique values matching query")
    page: int
    per_page: int
    cached: bool
    generated_at: datetime
