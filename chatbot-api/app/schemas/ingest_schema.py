from pydantic import BaseModel, Field


class IngestTopicRequest(BaseModel):
    """Request to ingest a single topic."""
    topic_slug: str = Field(..., description="Topic slug, e.g. 'digital-diplomacy'")
    site: str = Field(default="diplomacy.edu", description="Site domain")


class IngestAllTopicsRequest(BaseModel):
    """Request to reingest all topics from scratch."""
    site: str = Field(default="diplomacy.edu", description="Site domain")
    skip_existing: bool = Field(
        default=False,
        description="Skip topics already in Weaviate (True=incremental, False=full reingest)",
    )


class IngestTopicResponse(BaseModel):
    """Response for single topic ingest."""
    success: bool
    message: str
    data: dict


class IngestAllTopicsResponse(BaseModel):
    """Response for bulk topic ingest."""
    success: bool
    message: str
    data: dict
