"""Schemas for city/country/organisation catalog endpoint."""

from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel, Field


class MetadataFiltersResponse(BaseModel):
    """Unique city/country/organisation values aggregated from Weaviate."""

    cities: List[str] = Field(..., description="Sorted unique city names (case-sensitive)")
    countries: List[str] = Field(..., description="Sorted unique country names (case-sensitive)")
    organisations: List[str] = Field(
        ...,
        description="Sorted unique organisation names (case-sensitive)",
    )
    counts: Dict[str, int] = Field(
        ...,
        description="Unique value counts per field (city, country, organisation)",
    )
    sources: Dict[str, int] = Field(
        ...,
        description="Objects scanned per Weaviate collection",
    )
    cached: bool = Field(..., description="Whether the response was served from in-memory cache")
    generated_at: datetime = Field(..., description="UTC timestamp when the list was built")
