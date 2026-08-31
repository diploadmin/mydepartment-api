#!/usr/bin/env python3
"""Export a random sentence chunk about trade agreements from DiploChunk_contextual."""
import os
import json
from urllib.parse import urlparse

import weaviate
from weaviate.config import AdditionalConfig, Timeout
from weaviate.classes.query import Filter, HybridFusion

env_file = os.getenv("ENV_FILE", "../.env")
from starlette.config import Config
config = Config(env_file)

WV_CLIENT_URL = config("WV_CLIENT_URL", cast=str)
WV_KEY = config("WV_KEY", cast=str)
WV_GRPC_PORT = config("WV_GRPC_PORT", cast=int, default=0)
if WV_GRPC_PORT <= 0:
    raise SystemExit("WV_GRPC_PORT must be set in .env")

_parsed = urlparse(WV_CLIENT_URL)
WV_HOST = _parsed.hostname
if not WV_HOST:
    raise SystemExit("WV_CLIENT_URL must include a hostname")
if _parsed.port is None:
    raise SystemExit("WV_CLIENT_URL must include a port, e.g. http://host:8591")
WV_PORT = _parsed.port

client = weaviate.connect_to_local(
    host=WV_HOST, port=WV_PORT, grpc_port=WV_GRPC_PORT,
    auth_credentials=weaviate.auth.AuthApiKey(WV_KEY),
    additional_config=AdditionalConfig(timeout=Timeout(init=30, query=120)),
)

# List all collections to find contextual ones
all_colls = list(client.collections.list_all().keys())
contextual_colls = [c for c in all_colls if "contextual" in c.lower()]
print(f"All collections: {all_colls}")
print(f"Contextual collections: {contextual_colls}")

# Try DiploChunk_contextual
COLL_NAME = "DiploChunk_contextual"
if COLL_NAME not in all_colls:
    print(f"\n{COLL_NAME} not found. Trying DiploChunk instead...")
    COLL_NAME = "DiploChunk"

coll = client.collections.get(COLL_NAME)
total = coll.aggregate.over_all(total_count=True).total_count
print(f"\n{COLL_NAME}: {total:,} objects")

# Embed query
import requests as http_requests
LOCAL_EMBEDDING_URL = config("LOCAL_EMBEDDING_URL", cast=str, default="")
LOCAL_EMBEDDING_KEY = config("LOCAL_EMBEDDING_KEY", cast=str, default="")

resp = http_requests.post(
    f"{LOCAL_EMBEDDING_URL.rstrip('/')}/embed",
    headers={"Authorization": f"Bearer {LOCAL_EMBEDDING_KEY}", "Content-Type": "application/json"},
    json={"inputs": ["trade agreements"]},
    timeout=30,
)
resp.raise_for_status()
query_vector = resp.json()[0]
print(f"Embedded query ({len(query_vector)} dims)")

# Search for trade agreements
trade_filter = Filter.by_property("chunk_level").equal("sentence")
results = coll.query.hybrid(
    query="trade agreements",
    vector=query_vector,
    alpha=0.75,
    fusion_type=HybridFusion.RELATIVE_SCORE,
    limit=20,
    filters=trade_filter,
    query_properties=["sentence"],
    return_metadata=["score"],
)

print(f"\nFound {len(results.objects)} results for 'trade agreements'")

if results.objects:
    import random
    obj = random.choice(results.objects)
    props = obj.properties

    # Build export dict with all properties
    export = {
        "uuid": str(obj.uuid),
        "collection": COLL_NAME,
        "score": round(obj.metadata.score, 6) if obj.metadata else None,
        "properties": {k: str(v) if not isinstance(v, (str, int, float, bool, list, type(None))) else v
                       for k, v in props.items()},
    }

    output_path = "/opt/prod/humainism_ai_chatbot_api/trade-agreements-sentence-chunk.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)

    print(f"\nExported to: {output_path}")
    print(f"Sentence: {props.get('sentence', '')[:200]}")
    print(f"Link: {props.get('link', '')}")
    print(f"H1: {props.get('h1', '')}")
    print(f"Score: {obj.metadata.score:.6f}" if obj.metadata else "")
else:
    print("No results found!")

client.close()
