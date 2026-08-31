"""
Retrieval Cache (Redis).

Caches post-reranker documents so repeat queries skip embedding + search + reranker.
Key = sha256(normalized_query), value = JSON-serialized docs, TTL from config.
"""

import hashlib
import json as _json
from langchain_core.documents import Document

# context: this is a helper function to create the cache key for the user question - this helps Redis to find the cached documents for the user question
# description: function to create the cache key for the user question
# input given by: graph.py - from state ['messages']
# params: query - the user question - given in graph.py - from state ['messages'] 
# return: cache key - format: ret_cache:hash - this will be used for lookup in Redis
def _cache_key(query: str) -> str:
    """Deterministic Redis key for a retrieval query (user_type independent)."""
    # normalize the query - remove whitespace and convert to lowercase
    normalized = query.strip().lower()
    # hash the normalized query - use sha256 and take the first 24 characters
    h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    # return the cache key - format: ret_cache:hash
    return f"ret_cache:{h}"

# context: this is a helper function to serialize the documents to JSON - this helps to store the documents in Redis - 
# input given by: graph.py - active_retrieval_node.invoke(user_question) - used to make catched documents from Redis
def _serialize_docs(docs) -> str:
    """Serialize a list of LangChain Documents to JSON."""
    items = []
    for d in docs:
        items.append({
            "page_content": d.page_content,
            "metadata": {k: v for k, v in d.metadata.items() if _is_json_serializable(v)}
        })
    return _json.dumps(items, ensure_ascii=False)

# context: this is a helper function to deserialize the documents from JSON - this helps to get the documents from Redis - 
# input given by: graph.py - _catched - this is the output from Redis.get(_ckey) if there is a cache hit
def _deserialize_docs(raw: str):
    """Deserialize JSON back to a list of LangChain Documents."""
    items = _json.loads(raw)
    return [Document(page_content=it["page_content"], metadata=it["metadata"]) for it in items]


def _is_json_serializable(value) -> bool:
    """Quick check — skip values that can't round-trip through JSON."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_serializable(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_serializable(v) for k, v in value.items())
    return False
