"""
Ingest Service — handles topic ingestion via the chunking API.

Shared by both the API endpoint and CLI scripts.
"""

import time
import requests
from urllib.parse import urlparse

import weaviate
from weaviate.config import AdditionalConfig, Timeout
from weaviate.classes.query import Filter

from app.core.config import (
    WV_CLIENT_URL, WV_KEY, WV_GRPC_PORT,
)
from app.core.collections import CHUNK, PARAGRAPH
from app.core.logging import logger


_parsed = urlparse(WV_CLIENT_URL)
_WV_HOST = _parsed.hostname
if not _WV_HOST:
    raise ValueError("WV_CLIENT_URL must include a hostname")
if _parsed.port is None:
    raise ValueError(
        "WV_CLIENT_URL must include a port, e.g. http://ai6000.diplomacy.edu:8591"
    )
_WV_PORT = _parsed.port
if WV_GRPC_PORT <= 0:
    raise ValueError("WV_GRPC_PORT must be set in .env")
_WV_GRPC = WV_GRPC_PORT

CHUNKING_API_PORT = 8563
CHUNKING_API_URL = f"http://localhost:{CHUNKING_API_PORT}/api/v1"

PARAGRAPH_COLLECTION = PARAGRAPH
CHUNK_COLLECTION = CHUNK


def _connect_weaviate():
    """Create a new Weaviate client connection."""
    return weaviate.connect_to_local(
        host=_WV_HOST,
        port=_WV_PORT,
        grpc_port=_WV_GRPC,
        auth_credentials=weaviate.auth.AuthApiKey(WV_KEY),
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=120, insert=300)
        ),
    )


def delete_topic_data(url: str) -> dict:
    """Delete contextual paragraph/chunk data for a URL."""
    client = _connect_weaviate()
    try:
        results = {}
        url_filter = Filter.by_property("link").equal(url)

        for coll_name in [PARAGRAPH_COLLECTION, CHUNK_COLLECTION]:
            coll = client.collections.get(coll_name)
            before = coll.aggregate.over_all(total_count=True, filters=url_filter).total_count
            del_result = coll.data.delete_many(where=url_filter)
            results[coll_name] = {"deleted": del_result.successful, "before": before}

        return results
    finally:
        client.close()


def delete_all_topic_data() -> dict:
    """Delete ALL post_type=topic data from contextual paragraph/chunk collections."""
    client = _connect_weaviate()
    try:
        results = {}
        topic_filter = Filter.by_property("post_type").equal("topic")

        for coll_name in [PARAGRAPH_COLLECTION, CHUNK_COLLECTION]:
            coll = client.collections.get(coll_name)
            before = coll.aggregate.over_all(total_count=True, filters=topic_filter).total_count
            del_result = coll.data.delete_many(where=topic_filter)
            results[coll_name] = {"deleted": del_result.successful, "before": before}

        return results
    finally:
        client.close()


def ingest_single_topic(topic_slug: str, site: str = "diplomacy.edu") -> dict:
    """Ingest a single topic via chunking API."""
    t0 = time.time()

    logger.info(f"Ingesting topic '{topic_slug}' from {site}...")
    try:
        resp = requests.post(
            f"{CHUNKING_API_URL}/ingest/topic",
            json={"site": site, "topic_slug": topic_slug},
            timeout=300,
        )
    except requests.exceptions.ConnectionError:
        return {"error": f"Chunking API not reachable at {CHUNKING_API_URL}"}
    except requests.exceptions.Timeout:
        return {"error": "Chunking API request timed out (300s)"}

    if resp.status_code != 200:
        return {"error": f"Chunking API HTTP {resp.status_code}: {resp.text[:300]}"}

    ingest_data = resp.json().get("data", {})
    url = ingest_data.get("link", "")
    logger.info(
        f"  Ingested: {ingest_data.get('paragraphs_created', 0)} paragraphs, "
        f"{ingest_data.get('chunks_created', 0)} chunks"
    )

    elapsed = time.time() - t0
    return {
        "topic_slug": topic_slug,
        "site": site,
        "url": url,
        "paragraphs_created": ingest_data.get("paragraphs_created", 0),
        "chunks_created": ingest_data.get("chunks_created", 0),
        "was_update": ingest_data.get("was_update", False),
        "elapsed_s": round(elapsed, 1),
    }


def ingest_all_topics(
    site: str = "diplomacy.edu",
    skip_existing: bool = False,
) -> dict:
    """Full reingest of all topics via chunking API."""
    t0 = time.time()

    try:
        resp = requests.get(
            f"{CHUNKING_API_URL}/ingest/topics/stats?site={site}", timeout=10
        )
        stats = resp.json().get("data", {})
        total_topics = stats.get("topics_with_content", 0)
        logger.info(f"Chunking API: {total_topics} topics with content on {site}")
    except Exception as e:
        return {"error": f"Chunking API not reachable at {CHUNKING_API_URL}: {e}"}

    delete_result = {}
    if not skip_existing:
        logger.info("Deleting all old topic data from Weaviate...")
        delete_result = delete_all_topic_data()
        for coll, info in delete_result.items():
            logger.info(f"  {coll}: deleted {info['deleted']} (was {info['before']})")

    logger.info(f"Calling bulk ingest (skip_existing={skip_existing})...")
    t_ingest = time.time()

    try:
        resp = requests.post(
            f"{CHUNKING_API_URL}/ingest/topics/bulk",
            json={
                "site": site,
                "all_topics": True,
                "skip_existing": skip_existing,
            },
            timeout=1800,
        )
    except requests.exceptions.Timeout:
        return {"error": "Bulk ingest timed out (1800s)"}
    except Exception as e:
        return {"error": f"Bulk ingest failed: {e}"}

    if resp.status_code != 200:
        return {"error": f"Bulk ingest HTTP {resp.status_code}: {resp.text[:500]}"}

    data = resp.json()
    results = data.get("data", {}).get("results", [])
    ingest_elapsed = time.time() - t_ingest

    ingested = sum(1 for r in results if not r.get("skipped") and "error" not in r)
    skipped = sum(1 for r in results if r.get("skipped"))
    errors_count = sum(1 for r in results if "error" in r)
    total_paragraphs = sum(r.get("paragraphs_created", 0) for r in results)
    total_chunks = sum(r.get("chunks_created", 0) for r in results)

    logger.info(
        f"Bulk ingest done in {ingest_elapsed:.1f}s: "
        f"ingested={ingested}, skipped={skipped}, errors={errors_count}, "
        f"paragraphs={total_paragraphs}, chunks={total_chunks}"
    )

    error_details = []
    for r in results:
        if "error" in r:
            detail = f"{r.get('title', r.get('topic_slug', '?'))}: {r['error'][:100]}"
            error_details.append(detail)
            logger.warning(f"  Ingest error: {detail}")

    elapsed = time.time() - t0
    return {
        "site": site,
        "topics_ingested": ingested,
        "topics_skipped": skipped,
        "topics_errors": errors_count,
        "paragraphs_created": total_paragraphs,
        "chunks_created": total_chunks,
        "error_details": error_details,
        "deleted": delete_result,
        "ingest_elapsed_s": round(ingest_elapsed, 1),
        "total_elapsed_s": round(elapsed, 1),
    }
