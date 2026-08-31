"""
Custom filter catalog service.

Admin/sync layer for the WordPress custom filter builder: lists allowed
collections, filterable fields, and unique field values via the v4 client.

Runtime filter application (chat) lives in retrievers/utils.py
(custom_filters_for_collection), not here.
"""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.collections import ALL_DATA_COLLECTIONS
from app.core.config import PERSON_LIST_CACHE_TTL_SECONDS
from app.core.logging import logger
from app.core.singleton import Singleton

_BATCH_SIZE = 500
_MAX_SCAN_OBJECTS = 100_000
_MAX_UNIQUE_VALUES = 10_000
_ALLOWED_COLLECTIONS = frozenset(ALL_DATA_COLLECTIONS)

_cache_lock = threading.Lock()
_collections_cache: Optional[Dict[str, Any]] = None
_collections_expires: float = 0.0
_fields_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
_values_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _validate_collection(collection: str) -> str:
    name = (collection or "").strip()
    if name not in _ALLOWED_COLLECTIONS:
        raise ValueError(f"Unknown or disallowed collection: {collection}")
    return name


def _cache_get(store: Dict[str, Tuple[Dict[str, Any], float]], key: str, refresh: bool) -> Optional[Dict[str, Any]]:
    if refresh:
        return None
    with _cache_lock:
        entry = store.get(key)
        if entry and time.monotonic() < entry[1]:
            payload = dict(entry[0])
            payload["cached"] = True
            return payload
    return None


def _cache_set(store: Dict[str, Tuple[Dict[str, Any], float]], key: str, payload: Dict[str, Any]) -> None:
    with _cache_lock:
        store[key] = (payload, time.monotonic() + PERSON_LIST_CACHE_TTL_SECONDS)


def _collect_value(value: Any, values: Set[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            values.add(stripped)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_value(item, values)


def _collection_object_count(collection_name: str) -> Optional[int]:
    try:
        client = Singleton().weaviate_client
        collection = client.collections.get(collection_name)
        result = collection.aggregate.over_all(total_count=True)
        if result and result.total_count is not None:
            return int(result.total_count)
    except Exception as exc:
        logger.warning(f"[custom-filter] count failed for {collection_name}: {exc}")
    return None


def list_weaviate_collections(*, refresh: bool = False) -> Dict[str, Any]:
    global _collections_cache, _collections_expires

    now = time.monotonic()
    if not refresh:
        with _cache_lock:
            if _collections_cache is not None and now < _collections_expires:
                return {**_collections_cache, "cached": True}

    items: List[Dict[str, Any]] = []
    for name in sorted(_ALLOWED_COLLECTIONS):
        items.append({
            "name": name,
            "object_count": _collection_object_count(name),
        })

    payload = {
        "collections": items,
        "cached": False,
        "generated_at": _now_utc(),
    }
    with _cache_lock:
        _collections_cache = payload
        _collections_expires = time.monotonic() + PERSON_LIST_CACHE_TTL_SECONDS
    return payload


def _filterable_fields(collection_name: str) -> List[str]:
    """Return text-like property names from the Weaviate collection schema."""
    client = Singleton().weaviate_client
    collection = client.collections.get(collection_name)
    config = collection.config.get()
    fields: List[str] = []
    skip = {"id", "vector", "_additional"}
    for prop in getattr(config, "properties", []) or []:
        name = getattr(prop, "name", None)
        if not name or name in skip:
            continue
        data_type = getattr(prop, "data_type", None)
        type_name = str(data_type).lower() if data_type is not None else "text"
        if "text" in type_name or "string" in type_name or data_type is None:
            fields.append(name)
    return sorted(set(fields))


def list_collection_fields(collection: str, *, refresh: bool = False) -> Dict[str, Any]:
    collection_name = _validate_collection(collection)
    cache_key = collection_name

    cached = _cache_get(_fields_cache, cache_key, refresh)
    if cached:
        return cached

    try:
        fields = _filterable_fields(collection_name)
    except Exception as exc:
        logger.exception(f"[custom-filter] fields failed for {collection_name}")
        raise RuntimeError(f"Failed to list fields for {collection_name}: {exc}") from exc

    payload = {
        "collection": collection_name,
        "fields": fields,
        "cached": False,
        "generated_at": _now_utc(),
    }
    _cache_set(_fields_cache, cache_key, payload)
    return payload


def _scan_unique_values(collection_name: str, field: str) -> List[str]:
    client = Singleton().weaviate_client
    collection = client.collections.get(collection_name)
    values: Set[str] = set()
    scanned = 0

    for obj in collection.iterator(
        return_properties=[field],
        cache_size=_BATCH_SIZE,
    ):
        scanned += 1
        if scanned > _MAX_SCAN_OBJECTS:
            logger.warning(
                f"[custom-filter] value scan capped at {_MAX_SCAN_OBJECTS} for "
                f"{collection_name}.{field}"
            )
            break
        props = obj.properties or {}
        _collect_value(props.get(field), values)
        if len(values) >= _MAX_UNIQUE_VALUES:
            break

    logger.info(
        f"[custom-filter] {collection_name}.{field}: scanned {scanned:,}, "
        f"{len(values):,} unique values"
    )
    return sorted(values)


def list_field_values(
    collection: str,
    field: str,
    *,
    q: str = "",
    page: int = 1,
    per_page: int = 10,
    refresh: bool = False,
) -> Dict[str, Any]:
    collection_name = _validate_collection(collection)
    field_name = re.sub(r"[^a-zA-Z0-9_]", "", (field or "").strip())
    if not field_name:
        raise ValueError("Field name is required")

    allowed_fields = list_collection_fields(collection_name, refresh=False)["fields"]
    if field_name not in allowed_fields:
        raise ValueError(f"Field '{field_name}' is not available on {collection_name}")

    per_page = max(1, min(100, int(per_page)))
    page = max(1, int(page))
    query = (q or "").strip().lower()

    cache_key = f"{collection_name}|{field_name}"
    full_values: List[str]

    with _cache_lock:
        cached_entry = _values_cache.get(cache_key)
        if not refresh and cached_entry and time.monotonic() < cached_entry[1]:
            cached_all = cached_entry[0].get("all_values")
            if cached_all is not None:
                full_values = cached_all
            else:
                cached_entry = None
        else:
            cached_entry = None

    if cached_entry is None:
        try:
            full_values = _scan_unique_values(collection_name, field_name)
        except Exception as exc:
            logger.exception(
                f"[custom-filter] values failed for {collection_name}.{field_name}"
            )
            raise RuntimeError(
                f"Failed to scan values for {collection_name}.{field_name}: {exc}"
            ) from exc
        _cache_set(_values_cache, cache_key, {
            "collection": collection_name,
            "field": field_name,
            "all_values": full_values,
            "cached": False,
            "generated_at": _now_utc(),
        })
        served_from_cache = False
    else:
        served_from_cache = True

    if query:
        filtered = [v for v in full_values if query in v.lower()]
    else:
        filtered = full_values

    total = len(filtered)
    start = (page - 1) * per_page
    page_values = filtered[start:start + per_page]

    return {
        "collection": collection_name,
        "field": field_name,
        "values": page_values,
        "count": len(page_values),
        "total": total,
        "page": page,
        "per_page": per_page,
        "cached": served_from_cache,
        "generated_at": _now_utc(),
    }
