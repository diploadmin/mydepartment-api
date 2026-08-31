"""
Aggregate unique author names from Weaviate document ``persons`` TEXT[] fields.

Scans only ``Documents``. Authors are denormalized onto chunks at ingest time;
document-level scan is sufficient for the person catalog.
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
from app.core.weaviate_props import PERSONS

_SCAN_COLLECTION = DOCUMENT
_BATCH_SIZE = 500

_cache_lock = threading.Lock()
_cache_payload: Optional[Dict[str, Any]] = None
_cache_expires_at: float = 0.0


def _collect_person_names(value: Any, names: Set[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            names.add(stripped)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_person_names(item, names)


def _scan_collection(client: weaviate.WeaviateClient, collection_name: str, names: Set[str]) -> int:
    """Full collection scan via iterator."""
    collection = client.collections.get(collection_name)
    scanned = 0

    for obj in collection.iterator(
        return_properties=[PERSONS],
        cache_size=_BATCH_SIZE,
    ):
        scanned += 1
        props = obj.properties or {}
        _collect_person_names(props.get(PERSONS), names)

    logger.info(
        f"[person] {collection_name}: scanned {scanned:,} objects, "
        f"{len(names):,} unique names so far"
    )
    return scanned


def _build_catalog() -> Dict[str, Any]:
    client = Singleton().weaviate_client
    names: Set[str] = set()
    sources: Dict[str, int] = {}

    logger.info(f"[person] Scanning {_SCAN_COLLECTION} for author names...")
    sources[_SCAN_COLLECTION] = _scan_collection(client, _SCAN_COLLECTION, names)

    persons = sorted(names)
    return {
        "persons": persons,
        "count": len(persons),
        "sources": sources,
        "cached": False,
        "generated_at": datetime.now(timezone.utc),
    }


def list_unique_persons(*, refresh: bool = False) -> Dict[str, Any]:
    """
    Return unique person names from Weaviate.

    Uses an in-memory cache unless ``refresh=True`` or TTL expired.
    """
    global _cache_payload, _cache_expires_at

    now = time.monotonic()
    if not refresh:
        with _cache_lock:
            if _cache_payload is not None and now < _cache_expires_at:
                return {**_cache_payload, "cached": True}

    try:
        payload = _build_catalog()
    except Exception as exc:
        logger.exception("[person] Weaviate scan failed")
        raise RuntimeError(f"Weaviate person catalog scan failed: {exc}") from exc

    with _cache_lock:
        _cache_payload = payload
        _cache_expires_at = time.monotonic() + PERSON_LIST_CACHE_TTL_SECONDS

    return payload
