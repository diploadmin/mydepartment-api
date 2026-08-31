"""
Metadata filter catalog — unique city/country/organisation values for WordPress sync.
"""

import asyncio

from fastapi import APIRouter, HTTPException, Query

from app.core.logging import logger
from app.schemas.metadata_schema import MetadataFiltersResponse
from app.services.metadata_catalog_service import list_metadata_filters

router = APIRouter()


@router.get(
    "",
    response_model=MetadataFiltersResponse,
    summary="List unique city, country, and organisation values from Weaviate",
    description=(
        "Scans DiploDocument_contextual only, deduplicates values, and returns "
        "sorted lists for multi-select filters (``city_filter_names``, "
        "``country_filter_names``, ``organisation_filter_names`` in retrieval_config)."
    ),
)
async def list_metadata(
    refresh: bool = Query(
        False,
        description="Bypass in-memory cache and rescan Weaviate",
    ),
) -> MetadataFiltersResponse:
    try:
        payload = await asyncio.to_thread(list_metadata_filters, refresh=refresh)
    except RuntimeError as exc:
        logger.error(f"[metadata] {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return MetadataFiltersResponse(**payload)
