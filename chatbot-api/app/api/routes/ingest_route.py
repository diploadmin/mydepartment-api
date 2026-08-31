"""
Ingest routes — topic ingestion via chunking API.

Endpoints:
  POST /ingest/topic          — ingest a single topic
  POST /ingest/topics/all     — reingest ALL topics from scratch
"""

from fastapi import APIRouter, HTTPException
from app.core.logging import logger
from app.schemas.ingest_schema import (
    IngestTopicRequest,
    IngestTopicResponse,
    IngestAllTopicsRequest,
    IngestAllTopicsResponse,
)
from app.services.ingest_service import (
    ingest_single_topic,
    ingest_all_topics,
)

router = APIRouter()


@router.post(
    "/topic",
    summary="Ingest a single topic",
    description=(
        "Calls the chunking API to ingest a single topic from WordPress. "
        "If the topic already exists, it will be updated (old data replaced)."
    ),
    response_model=IngestTopicResponse,
)
async def ingest_topic(request: IngestTopicRequest):
    """Ingest a single topic by slug."""
    try:
        logger.info(f"[ingest] Single topic: {request.topic_slug} (site={request.site})")
        result = ingest_single_topic(
            topic_slug=request.topic_slug,
            site=request.site,
        )

        if "error" in result:
            raise HTTPException(status_code=502, detail=result["error"])

        return IngestTopicResponse(
            success=True,
            message=(
                f"Ingested '{request.topic_slug}': "
                f"{result['paragraphs_created']} paragraphs, "
                f"{result['chunks_created']} chunks"
            ),
            data=result,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[ingest] Error ingesting topic '{request.topic_slug}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/topics/all",
    summary="Reingest ALL topics",
    description=(
        "Full reingest: deletes all existing topic data from Weaviate, "
        "calls chunking API to bulk-ingest all topics from WordPress. "
        "This is a long-running operation (typically 5-15 minutes)."
    ),
    response_model=IngestAllTopicsResponse,
)
async def ingest_topics_all(request: IngestAllTopicsRequest):
    """Full reingest of all topics."""
    try:
        logger.info(
            f"[ingest] Full topics reingest: site={request.site}, "
            f"skip_existing={request.skip_existing}"
        )
        result = ingest_all_topics(
            site=request.site,
            skip_existing=request.skip_existing,
        )

        if "error" in result:
            raise HTTPException(status_code=502, detail=result["error"])

        return IngestAllTopicsResponse(
            success=True,
            message=(
                f"Ingested {result['topics_ingested']} topics: "
                f"{result['paragraphs_created']} paragraphs, "
                f"{result['chunks_created']} chunks "
                f"({result['total_elapsed_s']}s)"
            ),
            data=result,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[ingest] Error in full topics reingest: {e}")
        raise HTTPException(status_code=500, detail=str(e))
