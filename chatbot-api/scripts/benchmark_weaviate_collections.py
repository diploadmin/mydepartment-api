#!/usr/bin/env python3
"""
Benchmark: DiploParagraph vs DiploChunk query speed.

Measures raw Weaviate hybrid search latency (embedding + query) for both
collections side-by-side, with identical parameters.

Usage:
    cd /opt/prod/humainism_ai_chatbot_api/chatbot-api
    ENV_FILE=../.env python scripts/benchmark_weaviate_collections.py
"""
import os
import sys
import time
import statistics
import requests
from urllib.parse import urlparse

import weaviate
from weaviate.config import AdditionalConfig, Timeout
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery

# ── Config from .env ──────────────────────────────────────────────
env_file = os.getenv("ENV_FILE", "../.env")
from starlette.config import Config
config = Config(env_file)

WV_CLIENT_URL = config("WV_CLIENT_URL", cast=str)
WV_KEY = config("WV_KEY", cast=str)
WV_GRPC_PORT = config("WV_GRPC_PORT", cast=int, default=0)

LOCAL_EMBEDDING_URL = config("LOCAL_EMBEDDING_URL", cast=str, default="")
LOCAL_EMBEDDING_KEY = config("LOCAL_EMBEDDING_KEY", cast=str, default="")
HYBRID_ALPHA = config("HYBRID_ALPHA", cast=float, default=0.75)

USE_CONTEXTUAL_COLLECTIONS = config("USE_CONTEXTUAL_COLLECTIONS", cast=bool, default=True)


def _effective_collection(base_name: str) -> str:
    if USE_CONTEXTUAL_COLLECTIONS and not base_name.endswith("_contextual"):
        return f"{base_name}_contextual"
    return base_name


PARAGRAPH_COLLECTION = _effective_collection(
    config("INDEX_NAME", cast=str, default="DiploParagraph")
)
CHUNK_COLLECTION = _effective_collection(
    config("SENTENCE_INDEX_NAME", cast=str, default="DiploChunk")
)

_parsed = urlparse(WV_CLIENT_URL)
WV_HOST = _parsed.hostname
if not WV_HOST:
    raise SystemExit("WV_CLIENT_URL must include a hostname")
if _parsed.port is None:
    raise SystemExit("WV_CLIENT_URL must include a port, e.g. http://host:8591")
WV_PORT = _parsed.port
if WV_GRPC_PORT <= 0:
    raise SystemExit("WV_GRPC_PORT must be set in .env")

# ── Test queries ──────────────────────────────────────────────────
QUERIES = [
    "What is digital diplomacy?",
    "How does AI affect international relations?",
    "What are the main principles of the Vienna Convention?",
    "Explain cybersecurity in diplomatic context",
    "What is the role of UNESCO in digital governance?",
    "How do small island states participate in global negotiations?",
    "What is the impact of social media on diplomacy?",
    "Explain the concept of digital sovereignty",
]

REPEATS = 3
LIMITS = [50, 100, 200]


def connect_weaviate():
    client = weaviate.connect_to_local(
        host=WV_HOST, port=WV_PORT, grpc_port=WV_GRPC_PORT,
        auth_credentials=weaviate.auth.AuthApiKey(WV_KEY),
        additional_config=AdditionalConfig(timeout=Timeout(init=30, query=120)),
    )
    meta = client.get_meta()
    print(f"  Weaviate: {WV_HOST}:{WV_PORT} (gRPC:{WV_GRPC_PORT}) — v{meta.get('version', '?')}")
    return client


def embed_query(text: str) -> list[float]:
    url = f"{LOCAL_EMBEDDING_URL.rstrip('/')}/embed"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {LOCAL_EMBEDDING_KEY}",
            "Content-Type": "application/json",
        },
        json={"inputs": [text]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()[0]


def run_hybrid_query(collection, query_text, query_vector, limit, query_props):
    t0 = time.perf_counter()
    results = collection.query.hybrid(
        query=query_text,
        vector=query_vector,
        alpha=HYBRID_ALPHA,
        fusion_type=HybridFusion.RELATIVE_SCORE,
        limit=limit,
        query_properties=query_props,
        return_metadata=["score"],
    )
    elapsed = time.perf_counter() - t0
    return elapsed, len(results.objects)


def run_sentence_hybrid(collection, query_text, query_vector, limit):
    """DiploChunk with chunk_level=sentence filter (mirrors real usage)."""
    base_filter = Filter.by_property("chunk_level").equal("sentence")
    t0 = time.perf_counter()
    results = collection.query.hybrid(
        query=query_text,
        vector=query_vector,
        alpha=HYBRID_ALPHA,
        fusion_type=HybridFusion.RELATIVE_SCORE,
        limit=limit,
        filters=base_filter,
        query_properties=["sentence"],
        return_metadata=["score"],
    )
    elapsed = time.perf_counter() - t0
    return elapsed, len(results.objects)


def collection_count(collection) -> int:
    return collection.aggregate.over_all(total_count=True).total_count


def main():
    print("=" * 72)
    print("  BENCHMARK: DiploParagraph vs DiploChunk — Raw Weaviate Query Speed")
    print("=" * 72)
    print()

    # Connect
    client = connect_weaviate()
    para_coll = client.collections.get(PARAGRAPH_COLLECTION)
    chunk_coll = client.collections.get(CHUNK_COLLECTION)

    para_count = collection_count(para_coll)
    chunk_count = collection_count(chunk_coll)
    print(f"  {PARAGRAPH_COLLECTION}: {para_count:,} objects")
    print(f"  {CHUNK_COLLECTION}: {chunk_count:,} objects")
    print(f"  Hybrid alpha: {HYBRID_ALPHA}")
    print(f"  Queries: {len(QUERIES)} × {REPEATS} repeats × {len(LIMITS)} limits")
    print()

    # Pre-embed all queries (so embedding time doesn't pollute query benchmark)
    print("  Embedding queries...", end=" ", flush=True)
    t0 = time.perf_counter()
    query_vectors = {}
    for q in QUERIES:
        query_vectors[q] = embed_query(q)
    t_embed = time.perf_counter() - t0
    print(f"done in {t_embed:.2f}s ({t_embed/len(QUERIES)*1000:.0f}ms/query)")
    print()

    # Warmup — run one query per collection to warm caches
    print("  [Warmup]", end=" ", flush=True)
    run_hybrid_query(para_coll, QUERIES[0], query_vectors[QUERIES[0]], 50, ["text"])
    run_sentence_hybrid(chunk_coll, QUERIES[0], query_vectors[QUERIES[0]], 50)
    print("done")
    print()

    # ── Run benchmark ─────────────────────────────────────────────
    all_results = {limit: {"paragraph": [], "sentence": []} for limit in LIMITS}

    for limit in LIMITS:
        print(f"─── limit={limit} {'─' * 50}")

        for q_idx, query in enumerate(QUERIES):
            vec = query_vectors[query]

            para_times = []
            sent_times = []

            for r in range(REPEATS):
                pt, pn = run_hybrid_query(para_coll, query, vec, limit, ["text"])
                st, sn = run_sentence_hybrid(chunk_coll, query, vec, limit)
                para_times.append(pt)
                sent_times.append(st)

            p_med = statistics.median(para_times) * 1000
            s_med = statistics.median(sent_times) * 1000
            diff = s_med - p_med
            pct = (diff / p_med * 100) if p_med > 0 else 0

            all_results[limit]["paragraph"].append(p_med)
            all_results[limit]["sentence"].append(s_med)

            faster = "PARA" if p_med < s_med else "SENT"
            print(f"  Q{q_idx+1:>2}: Para {p_med:6.1f}ms | Sent {s_med:6.1f}ms | "
                  f"Δ {abs(diff):+5.1f}ms ({abs(pct):4.1f}% {faster} faster) | "
                  f"\"{query[:45]}\"")

        print()

    # ── Summary ───────────────────────────────────────────────────
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print()
    print(f"  {'limit':>7}  {'Para med':>10}  {'Sent med':>10}  {'Para mean':>10}  {'Sent mean':>10}  {'Δ mean':>10}")
    print(f"  {'─'*7}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    for limit in LIMITS:
        p_vals = all_results[limit]["paragraph"]
        s_vals = all_results[limit]["sentence"]

        p_med = statistics.median(p_vals)
        s_med = statistics.median(s_vals)
        p_mean = statistics.mean(p_vals)
        s_mean = statistics.mean(s_vals)
        diff = s_mean - p_mean

        print(f"  {limit:>7}  {p_med:>8.1f}ms  {s_med:>8.1f}ms  "
              f"{p_mean:>8.1f}ms  {s_mean:>8.1f}ms  {diff:>+8.1f}ms")

    print()
    print(f"  Object counts: {PARAGRAPH_COLLECTION}={para_count:,}  |  {CHUNK_COLLECTION}={chunk_count:,}")
    print(f"  Note: DiploChunk queries include chunk_level='sentence' filter")
    print("=" * 72)

    client.close()


if __name__ == "__main__":
    main()
