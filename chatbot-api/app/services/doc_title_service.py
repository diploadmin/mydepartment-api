"""Recover display titles for documents whose ``document_title`` is an ingest artifact.

Dump-based ingests store the encoded source URL as the title (e.g.
``https%3A%2F%2F...%2Fwhat-is-pd.txt``), which renders as an ugly slug on source
cards. The real title is the first heading line of ``Documents.full_text``, so it
is fetched once per document hash and cached for the process lifetime.
"""

from weaviate.classes.query import Filter

from app.core.collections import DOCUMENT
from app.core.logging import logger
from app.core.singleton import Singleton
from app.core.weaviate_props import (
    DOCUMENT_HASH,
    FULL_TEXT,
    PARENT_DOCUMENT_HASH,
    doc_title,
    has_real_title,
    title_from_full_text,
)

# document_hash -> resolved title ('' when unavailable, to avoid retry storms)
_title_cache: dict[str, str] = {}


def stored_document_title(document_hash: str) -> str:
    """Title read from the document's full_text, or '' when it cannot be resolved."""
    doc_hash = (document_hash or "").strip()
    if not doc_hash:
        return ""
    if doc_hash in _title_cache:
        return _title_cache[doc_hash]

    title = ""
    try:
        from app.ai.ai_services.retrievers.utils import resolve_collection_name

        client = Singleton().weaviate_client
        collection = client.collections.get(resolve_collection_name(DOCUMENT))
        for prop in (DOCUMENT_HASH, PARENT_DOCUMENT_HASH):
            result = collection.query.fetch_objects(
                limit=1,
                filters=Filter.by_property(prop).equal(doc_hash),
                return_properties=[FULL_TEXT],
            )
            if result.objects:
                props = result.objects[0].properties or {}
                title = title_from_full_text(str(props.get(FULL_TEXT) or ""))
                break
    except Exception as exc:
        logger.warning(f"[doc-title] lookup failed for {doc_hash}: {exc}")

    _title_cache[doc_hash] = title
    return title


def display_title(props: dict) -> str:
    """Card title: the document's own title, else one recovered from full_text."""
    title = doc_title(props)
    if has_real_title(props):
        return title
    recovered = stored_document_title(
        props.get(PARENT_DOCUMENT_HASH) or props.get(DOCUMENT_HASH) or ""
    )
    return recovered or title
