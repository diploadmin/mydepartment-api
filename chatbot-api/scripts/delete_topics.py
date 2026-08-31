#!/usr/bin/env python3
"""Delete all topic-type content from dev Weaviate (contextual paragraph/chunk collections)."""
import sys, os, time
from urllib.parse import urlparse

sys.path.insert(0, "/opt/humainism_ai_chatbot_api")
os.chdir("/opt/humainism_ai_chatbot_api")

from starlette.config import Config
config = Config(".env")

import weaviate
from weaviate.config import AdditionalConfig, Timeout
from weaviate.classes.query import Filter

_wv_url = config("WV_CLIENT_URL", cast=str)
_parsed = urlparse(_wv_url)
host = _parsed.hostname
if not host:
    raise SystemExit("WV_CLIENT_URL must include a hostname")
if _parsed.port is None:
    raise SystemExit("WV_CLIENT_URL must include a port, e.g. http://host:8591")
port = _parsed.port
grpc_port = config("WV_GRPC_PORT", cast=int, default=0)
if grpc_port <= 0:
    raise SystemExit("WV_GRPC_PORT must be set in .env")

client = weaviate.connect_to_local(
    host=host, port=port, grpc_port=grpc_port,
    auth_credentials=weaviate.auth.AuthApiKey(config("WV_KEY", cast=str)),
    additional_config=AdditionalConfig(timeout=Timeout(init=30, query=120, insert=300)),
)

COLLECTIONS = ["DiploParagraph_contextual", "DiploChunk_contextual"]

print("=" * 60)
print("DELETE ALL TOPIC CONTENT FROM DEV WEAVIATE")
print(f"Weaviate: {host}:{port}")
print("=" * 60)

for coll_name in COLLECTIONS:
    print(f"\n--- {coll_name} ---")
    
    try:
        coll = client.collections.get(coll_name)
    except Exception as e:
        print(f"  Collection not found: {e}")
        continue
    
    # Count total objects
    total = coll.aggregate.over_all(total_count=True).total_count
    
    # Count topic objects
    topic_filter = Filter.by_property("post_type").equal("topic")
    topic_count = coll.aggregate.over_all(total_count=True, filters=topic_filter).total_count
    
    print(f"  Total objects: {total}")
    print(f"  Topic objects: {topic_count}")
    
    if topic_count == 0:
        print(f"  Nothing to delete.")
        continue
    
    # Delete all topic objects
    t0 = time.time()
    result = coll.data.delete_many(
        where=topic_filter,
    )
    elapsed = time.time() - t0
    
    print(f"  Deleted: {result.successful} objects in {elapsed:.2f}s")
    if result.failed:
        print(f"  Failed: {result.failed}")
    
    # Verify
    remaining = coll.aggregate.over_all(total_count=True).total_count
    topic_remaining = coll.aggregate.over_all(total_count=True, filters=topic_filter).total_count
    print(f"  Remaining total: {remaining} (topics: {topic_remaining})")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

client.close()
