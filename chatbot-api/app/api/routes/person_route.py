"""
Person catalog route — unique names from Weaviate for WordPress sync.
"""

import asyncio

from fastapi import APIRouter, HTTPException, Query

from app.core.logging import logger
from app.schemas.person_schema import PersonListResponse
from app.services.person_catalog_service import list_unique_persons

router = APIRouter()


@router.get(
    "",
    response_model=PersonListResponse,
    summary="List unique person names from Weaviate",
    description=(
        "Scans DiploDocument_contextual only (document authors via ``person`` TEXT[]), "
        "deduplicates names, and returns a sorted list. "
        "Protected by ALLOWED_IPS / ALLOWED_HOSTS (CORS) like other API routes."
    ),
)
async def list_person(
    refresh: bool = Query(
        False,
        description="Bypass in-memory cache and rescan Weaviate",
    ),
) -> PersonListResponse:
    try:
        payload = await asyncio.to_thread(list_unique_persons, refresh=refresh)
    except RuntimeError as exc:
        logger.error(f"[person] {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PersonListResponse(**payload)
