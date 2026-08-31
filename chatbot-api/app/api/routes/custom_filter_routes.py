"""Custom filter catalog routes for WordPress admin sync."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.auth.sync_auth import verify_wp_weaviate_basic_auth
from app.core.logging import logger
from app.schemas.weaviate_schema import (
    WeaviateCollectionsResponse,
    WeaviateFieldsResponse,
    WeaviateValuesResponse,
)
from app.services.custom_filter_service import (
    list_collection_fields,
    list_field_values,
    list_weaviate_collections,
)

router = APIRouter(dependencies=[Depends(verify_wp_weaviate_basic_auth)])


@router.get(
    "/collections",
    response_model=WeaviateCollectionsResponse,
    summary="List Weaviate collections for custom filters",
)
async def get_collections(
    refresh: bool = Query(False, description="Bypass in-memory cache"),
) -> WeaviateCollectionsResponse:
    try:
        payload = await asyncio.to_thread(list_weaviate_collections, refresh=refresh)
    except RuntimeError as exc:
        logger.error(f"[custom-filter] collections: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return WeaviateCollectionsResponse(**payload)


@router.get(
    "/collections/{collection}/fields",
    response_model=WeaviateFieldsResponse,
    summary="List filterable fields for a collection",
)
async def get_collection_fields(
    collection: str,
    refresh: bool = Query(False, description="Bypass in-memory cache"),
) -> WeaviateFieldsResponse:
    try:
        payload = await asyncio.to_thread(list_collection_fields, collection, refresh=refresh)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error(f"[custom-filter] fields: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return WeaviateFieldsResponse(**payload)


@router.get(
    "/collections/{collection}/fields/{field}/values",
    response_model=WeaviateValuesResponse,
    summary="Search unique field values (max 10 per page)",
)
async def get_field_values(
    collection: str,
    field: str,
    q: str = Query("", description="Substring search (case-insensitive); longer query = narrower matches"),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=10),
    refresh: bool = Query(False, description="Bypass in-memory cache and rescan Weaviate"),
) -> WeaviateValuesResponse:
    try:
        payload = await asyncio.to_thread(
            list_field_values,
            collection,
            field,
            q=q,
            page=page,
            per_page=per_page,
            refresh=refresh,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error(f"[custom-filter] values: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return WeaviateValuesResponse(**payload)
