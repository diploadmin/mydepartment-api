# Pipeline Improvements & Refactoring Plan

**Date:** 2026-02-16  
**Status:** Phase 1 Completed (2026-02-16)

---

## 1. Pipeline Performance Improvements

### 1.1 Query-Level Cache ✅ DONE

**Implemented:** `direct_retrieval_node` in `chatbot.py` (post-reranker, pre-label-weights).

- **Cache key:** `ret_cache:{sha256(normalized_query)[:24]}` — user_type independent (label weights applied after cache, <1ms)
- **What's cached:** JSON-serialized `List[Document]` (page_content + metadata)
- **TTL:** `RETRIEVAL_CACHE_TTL` env var (default: 120s, 0 = disabled)
- **Helpers:** `_cache_key()`, `_serialize_docs()`, `_deserialize_docs()` at top of `chatbot.py`
- **Cache hit fallback:** `handle_chat_request_WS` recovers docs from `_last_reranked_docs` when `on_retriever_end` doesn't fire (cache skips retriever invocation)

**Measured savings on cache HIT:** retrieval=0.000s, reranker=0.000s (vs ~8.9s MISS)

---

### 1.2 Batch Deep Link Creation ✅ DONE

**Implemented:**

- `build_deep_link_urls_batch()` in `chat_service.py` — collects all docs, calls `create_deep_links_batch()` once
- `DeepLinkService.create_deep_links_batch()` upgraded to use Redis pipeline (`pipe.setex()` × N → `pipe.execute()` = 1 round-trip)
- Single `DeepLinkService` instance per WS request (`dl_service` in `handle_chat_request_WS`)
- Batch used in both `on_retriever_end` (all docs) and `filter_related_sources` (cited docs)

---

### 1.3 Parallel Injection (Step 1c) (MEDIUM PRIORITY)

**Problem:** Sentence injection iterates sequentially over up to 10 URLs, making one `near_vector` call per URL (~5-15ms each). Total: ~50-150ms.

**Proposed solution:** Use `concurrent.futures.ThreadPoolExecutor` to parallelize injection calls (Weaviate Python client v4.12 is sync-only, so asyncio isn't an option).

**Estimated savings:** ~50-80ms (10 sequential calls → 1 parallel batch)

---

### 1.4 Background Related Questions (LOW PRIORITY)

**Problem:** Related questions generation (~5s LLM call) happens after the main response. User waits for both.

**Proposed solution:** Fire related questions generation as a background task; send via WebSocket when ready. User sees sources immediately.

**Estimated savings:** Better perceived UX, no actual latency change.

---

### 1.5 Batch URL Redirect Resolution (LOW PRIORITY)

**Problem:** `resolve_url_redirects()` does HTTP HEAD per URL. For 8 documents = 8 HTTP calls.

**Current mitigation:** In-memory cache (`_url_redirect_cache`) already covers repeated URLs. After first few requests, most URLs are cached.

**Proposed solution:** Parallel resolution for uncached URLs using `asyncio.gather` or `ThreadPoolExecutor`. Only matters on cold start.

**Estimated savings:** ~100-400ms on first request with many new URLs, negligible after warm-up.

---

## 2. Code Refactoring

### 2.1 Extract `build_deep_link_url()` ✅ DONE

**Implemented:** Top-level function in `chat_service.py` replacing all 4 duplicated copies.

- `_normalize_url_with_redirect(url)` — HTTP→HTTPS, fragment extraction, redirect resolution
- `_build_deep_link_text(page_content, metadata, title, url)` — sentence/paragraph mode text builder
- `build_deep_link_url(doc_or_dict, query, deep_link_service)` — full pipeline: normalize → text → Redis/URL-param
- `build_deep_link_urls_batch(docs, query, deep_link_service)` — batch version using Redis pipeline

**Bugs fixed:** Copy 3 now uses `get_metadata_field()` (was `.get('title','')`), all copies send `query` in Redis metadata, fallback: 2-sentence cap everywhere.

**Impact:** ~200 lines of duplicated code eliminated.

---

### 2.2 Unified URL Normalization ✅ DONE

Implemented as `_normalize_url_with_redirect(url)` in `chat_service.py` — used by both `build_deep_link_url()` and `__repack_source_documents()`.

---

### 2.3 Split `chatbot.py` ✅ DONE

**Implemented:** Modular structure under `app/ai/ai_services/`:
```
app/ai/ai_services/
  graph.py               # LangGraph agent, tool nodes, graph construction
  label_weights.py       # Static profile label weights + get_label_weights()
  query_intent.py        # Dynamic label weight boosting via prototype embeddings
  reranker.py            # TEIReranker
  embeddings.py          # TEIEmbeddings
  retrievers/
    sentence_first.py    # SentenceFirstRetriever
    combined.py          # CombinedRetriever
    two_phase.py         # TwoPhaseRetriever
    label_rescoring.py   # LabelRescoringRetriever
    factory.py           # create_retriever(), resolve_retrieval_mode()
```

---

### 2.4 Replace `print()` with `logging` (LOW PRIORITY)

**Problem:** 50+ `print(f"DEBUG: ...", flush=True)` statements in production code. Cannot control log levels.

**Proposed solution:** Replace with `logger.debug()`, `logger.info()`, `logger.warning()`. Control via `LOG_LEVEL` env var.

**Note:** Keep current prints during active development. Convert when pipeline is stable.

---

## 3. Implementation Order

| Phase | Task | Priority | Status |
|-------|------|----------|--------|
| **1** | Extract `build_deep_link_url()` | HIGH | ✅ Done |
| **1** | Query-level Redis cache | HIGH | ✅ Done |
| **1** | Batch deep link creation | MEDIUM | ✅ Done |
| **1** | Unified URL normalization | MEDIUM | ✅ Done |
| **2** | Parallel injection (Step 1c) | MEDIUM | Pending |
| **3** | Split `chatbot.py` into modules | LOW | ✅ Done |
| **3** | Dynamic label weight boosting | MEDIUM | ✅ Done |
| **3** | Replace print with logging | LOW | Pending |
| **3** | Background related questions | LOW | Pending |
| **3** | Batch URL redirect resolution | LOW | Pending |

---

## 4. Monitoring After Changes

- Track `_matched_sentences` count distribution after injection limit change (100→20)
- Monitor cache hit rate for query-level cache
- Compare end-to-end latency before/after each phase
- Watch for stale cache issues after content ingestion

---

## 5. Recent Changes

### 2026-02-14

- **Weaviate-level blocklist filter:** Replaced Python post-filtering (over-fetch 1000 → filter in Python) with Weaviate `not_equal` filter. Hybrid search now returns 200 clean sentences directly (limit=200), eliminating ~643 boilerplate copies. Hybrid search time reduced from ~2.4s to ~1.5s.
  - New function: `build_weaviate_blocklist_filter()` in `app/core/blocklist.py`
  - Blocklist patterns loaded from `config/sentence_blocklist.txt`
  - Configurable via `USE_SENTENCE_BLOCKLIST` in `.env`
- **Timing instrumentation:** Added `TIMING` log lines to `SentenceFirstRetriever`, `direct_retrieval_node`, `generate_final_node`, and `generate_related_questions_from_sources` for performance profiling.
- **Alpha tuned to 0.6:** Changed `HYBRID_ALPHA` from 0.85 to 0.6 in `.env` for better balance between semantic and keyword search.
- **Hardcoded alpha removed:** `SentenceFirstRetriever` and `CombinedRetriever` now use `HYBRID_ALPHA` from config instead of hardcoded `alpha=0.6`.
- **Fusion type explicit:** Added `fusion_type=HybridFusion.RELATIVE_SCORE` to hybrid search calls.

### 2026-02-16

- **Injection limit:** Changed from `limit=100` to `limit=20` in Step 1c (prevents flooding `_matched_sentences` with entire page contents)
- **CORS fix:** Added `www.diplomacy.edu` to `ALLOWED_HOSTS` with full URL format
- **IP allowlist bypass:** Added `/api/deep-link/` to public endpoints in `IPAllowlistMiddleware`
- **Plugin fix:** Updated `API_BASE` URL in `diplo-deep-link-finder.js` and deployed to v43.diplomacy.edu
- **Phase 1 refactor completed:**
  - `build_deep_link_url()` extracted as top-level function, replacing 4 duplicated copies (~200 lines removed)
  - `_normalize_url_with_redirect()` and `_build_deep_link_text()` helpers shared by all URL/text building code
  - `build_deep_link_urls_batch()` with Redis pipeline (1 round-trip instead of N)
  - `DeepLinkService.create_deep_links_batch()` upgraded to use `redis.pipeline()`
  - Single `DeepLinkService` instance per request
  - Query-level Redis cache in `direct_retrieval_node` (post-reranker, pre-label-weights, TTL 120s)
  - Cache key: `ret_cache:{sha256(query)[:24]}` — user_type independent
  - Cache hit fallback in WS handler to recover docs when `on_retriever_end` doesn't fire
  - New config: `RETRIEVAL_CACHE_TTL` in `.env` and `config.py`