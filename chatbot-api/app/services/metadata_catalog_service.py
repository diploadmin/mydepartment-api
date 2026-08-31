"""
Aggregate unique country values from Weaviate Documents.

oneweaviate exposes ``countries`` TEXT[]; city/organisation are not present
and return empty lists for API compatibility.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

import weaviate

from app.core.collections import DOCUMENT
from app.core.config import PERSON_LIST_CACHE_TTL_SECONDS
from app.core.logging import logger
from app.core.singleton import Singleton
from app.core.weaviate_props import COUNTRIES

_SCAN_COLLECTION = DOCUMENT
_BATCH_SIZE = 500

_cache_lock = threading.Lock()
_cache_payload: Optional[Dict[str, Any]] = None
_cache_expires_at: float = 0.0


def _collect_text_values(value: Any, values: Set[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            values.add(stripped)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_text_values(item, values)


def _scan_collection(
    client: weaviate.WeaviateClient,
    collection_name: str,
    countries: Set[str],
) -> int:
    collection = client.collections.get(collection_name)
    scanned = 0

    for obj in collection.iterator(
        return_properties=[COUNTRIES],
        cache_size=_BATCH_SIZE,
    ):
        scanned += 1
        props = obj.properties or {}
        _collect_text_values(props.get(COUNTRIES), countries)

    logger.info(f"[metadata] {collection_name}: scanned {scanned:,} objects")
    return scanned


def _build_catalog() -> Dict[str, Any]:
    client = Singleton().weaviate_client
    countries: Set[str] = set()
    sources: Dict[str, int] = {}

    logger.info(f"[metadata] Scanning {_SCAN_COLLECTION} for countries...")
    sources[_SCAN_COLLECTION] = _scan_collection(client, _SCAN_COLLECTION, countries)

    return {
        "cities": [],
        "countries": sorted(countries),
        "organisations": [],
        "counts": {
            "city": 0,
            "country": len(countries),
            "organisation": 0,
        },
        "sources": sources,
        "cached": False,
        "generated_at": datetime.now(timezone.utc),
    }


def list_metadata_filters(*, refresh: bool = False) -> Dict[str, Any]:
    """Return unique country values from Weaviate (city/org empty on oneweaviate)."""
    global _cache_payload, _cache_expires_at

    now = time.monotonic()
    if not refresh:
        with _cache_lock:
            if _cache_payload is not None and now < _cache_expires_at:
                return {**_cache_payload, "cached": True}

    try:
        payload = _build_catalog()
    except Exception as exc:
        logger.exception("[metadata] Weaviate scan failed")
        raise RuntimeError(f"Weaviate metadata catalog scan failed: {exc}") from exc

    with _cache_lock:
        _cache_payload = payload
        _cache_expires_at = time.monotonic() + PERSON_LIST_CACHE_TTL_SECONDS

    return payload
