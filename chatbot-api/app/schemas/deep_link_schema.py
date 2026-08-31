"""
Pydantic schemas for deep link URL shortening.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class DeepLinkCreate(BaseModel):
    """Request to create a short deep link."""
    section_text: str = Field(..., description="Section content for context matching")
    sentences: List[str] = Field(..., min_items=1, description="List of matched sentences to highlight")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata (url, title, query)")


class DeepLinkBatchCreate(BaseModel):
    """Request to create multiple deep links in batch."""
    links: List[DeepLinkCreate] = Field(..., min_items=1)


class DeepLinkCreateResponse(BaseModel):
    """Response after creating a deep link."""
    id: str = Field(..., description="Short alphanumeric ID (e.g., 'abc123XYZ')")
    expires_in: int = Field(..., description="TTL in seconds (e.g., 2592000 for 30 days)")


class DeepLinkBatchCreateResponse(BaseModel):
    """Response after creating multiple deep links."""
    ids: List[str] = Field(..., description="List of short IDs in same order as request")


class DeepLinkData(BaseModel):
    """Deep link data retrieved by ID."""
    section_text: str = Field(..., description="Section content")
    sentences: List[str] = Field(..., description="List of sentences to highlight")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata")
