"""Schemas for the person catalog endpoint."""

from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel, Field


class PersonListResponse(BaseModel):
    """Unique person names aggregated from Weaviate."""

    persons: List[str] = Field(..., description="Sorted unique person names (case-sensitive)")
    count: int = Field(..., description="Number of unique names")
    sources: Dict[str, int] = Field(
        ...,
        description="Objects scanned per Weaviate collection",
    )
    cached: bool = Field(..., description="Whether the response was served from in-memory cache")
    generated_at: datetime = Field(..., description="UTC timestamp when the list was built")
