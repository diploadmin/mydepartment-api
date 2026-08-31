# Retrieval Pipeline

This document describes the retrieval pipeline for the Humainism AI Chatbot API. The pipeline supports multiple modes; the recommended production mode is `sentence` (SentenceFirstRetriever).

> **2026-03 update:** `DiploHeading` and Steps 1b/1c (heading boost + injection) were removed. Heading hierarchy is encoded in `DiploChunk_contextual` / `DiploParagraph_contextual` embeddings (`USE_CONTEXTUAL_COLLECTIONS=true`).

## Pipeline Overview (sentence mode)

```
User Query
    |
    v
[Embed] TEI embedder → query_vector (1024-dim)
    |  ~100ms (external API call to TEI service)
    v
[Step 1a] Hybrid Search (DiploChunk_contextual, sentence-level)
    |  alpha=0.6 (60% vector + 40% BM25)
    |  fusion: relativeScoreFusion
    |  limit=200, filtered by chunk_level="sentence"
    |  Weaviate-level blocklist: not_equal filter excludes boilerplate
    |  ~1.5-2.4s
    v
[Step 2] Group by (URL, section) + Aggregate scores
    |  aggregate = max_weight*max + avg_weight*avg + count_weight*log(1+n)
    |  Top 5 sentences per section cap
    |  <5ms
    v
[Step 3] Deduplicate + cap sections per URL
    |  MAX_SECTIONS_PER_URL=3, content dedup by hash
    |  --> Top 50 section candidates
    |  <1ms
    v
[Reranker] TEI cross-encoder reranker
    |  Score query-document pairs, format: [Label] Title > Section
    |  --> Top SENTENCE_RERANKER_TOP_K (default: 50)
    |  ~1.1-1.5s
    v
[Redis Cache] Query-level cache (post-reranker, pre-label-weights)
    |  Key: sha256(normalized_query), TTL: 120s
    |  On HIT: skip Steps 1-3 + Reranker (~3-9s saved)
    |  On MISS: store result for next identical query
    v
[Label Weights] Profile-based re-scoring + dynamic boost
    |  weight = max(static_weight, similarity * alpha)  [if dynamic enabled]
    |  final_score = reranker_score * weight
    |  --> Top RETRIEVAL_CHUNKS (default: 8)
    |  <1ms
    v
[Deep Links] build_deep_link_urls_batch()
    |  Batch Redis pipeline: 1 round-trip for all docs
    |  Short link IDs via DeepLinkService (single instance per request)
    v
[LLM] Generate answer with citations [1], [2], etc.
    |  ~5-6s (local LLM or OpenAI)
    v
[Related Questions] LLM generates 3-5 follow-up questions
    |  ~2-3s (parallel to response delivery via WebSocket)
```

## Hybrid Search Configuration

### BM25 Query Properties

BM25 search is restricted to the primary content property only, not all text properties on the collection. This prevents metadata fields (headings, section paths, post types) from inflating BM25 scores:

| Collection | Property searched by BM25 | Rationale |
|------------|--------------------------|-----------|
| DiploChunk | `sentence` | Actual sentence content only |
| DiploParagraph | `text` | Paragraph content only |

Without `query_properties`, BM25 would search all TEXT fields including `h1`-`h6`, `section_path`, `post_type`, `site`, etc. In a diplomacy-focused corpus, this causes noise because terms like "diplomacy" appear in nearly every metadata field, inflating scores for irrelevant documents.

### Alpha Parameter

The `HYBRID_ALPHA` parameter controls the balance between vector (semantic) and BM25 (keyword) search in Weaviate's hybrid query. In Weaviate: `alpha=0` is pure BM25, `alpha=1` is pure vector.

| Alpha | Vector Weight | BM25 Weight | Use Case |
|-------|--------------|-------------|----------|
| 0.0 | 0% | 100% | Pure keyword search |
| 0.5 | 50% | 50% | Equal balance |
| **0.6** | **60%** | **40%** | **Current production** |
| 0.75 | 75% | 25% | Previously recommended |
| 0.85 | 85% | 15% | Used in TwoPhase mode |
| 1.0 | 100% | 0% | Pure semantic search |

**Why 0.6?** A moderate balance that leverages semantic understanding for domain-specific terms while keeping enough BM25 signal to match specific keywords and proper nouns. Higher alpha values (0.75, 0.85) were tested but 0.6 provided better diversity in results for diplomacy-related queries.

### Fusion Type

All hybrid searches use `HybridFusion.RELATIVE_SCORE` which normalizes BM25 and vector scores to [0,1] before combining. This produces better-calibrated combined scores compared to the default Ranked Fusion.

## Per-request content filters

Optional filters are sent in `retrieval_config` on chat requests (see [API.md](API.md)). They are applied at Weaviate query time in all retrieval modes (`sentence`, `combined`, `twophase`, `paragraph`).

| `retrieval_config` field | Weaviate property | Behavior |
|--------------------------|-------------------|----------|
| `site_filter_name` | `site` | `equal` |
| `parent_filter_name` | `parent_document_hash` | Resolved from `DiploDocument.name`, then `equal` |
| `person_filter_name` | `person` (TEXT[]) | `contains_any` — matches if the name is **one element** of the array (multi-person chunks included) |
| `person_filter_names` | `person` (TEXT[]) | Same as above; OR across all listed names (merged with `person_filter_name` if both set) |
| `city_filter_name` | `city` (TEXT) | `contains_any` — property value equals the given city |
| `city_filter_names` | `city` (TEXT) | OR across listed cities (merged with `city_filter_name` if both set) |
| `country_filter_name` | `country` (TEXT) | `contains_any` — property value equals the given country |
| `country_filter_names` | `country` (TEXT) | OR across listed countries (merged with `country_filter_name` if both set) |
| `organisation_filter_name` | `organisation` (TEXT) | `contains_any` — property value equals the given organisation |
| `organisation_filter_names` | `organisation` (TEXT) | OR across listed organisations (merged with `organisation_filter_name` if both set) |
| `post_types` | `post_type` | `contains_any` whitelist |

Always applied: `visibility != "private"` (`visibility_filter()` in `retrievers/utils.py`). Person/city/country/organisation filters are applied via `append_content_filters()` in the same file (reads `*_filter_name` / `*_filter_names` from per-request overrides). Multiple filter fields are ANDed; values within one field are ORed.

## Sentence Blocklist

Boilerplate sentences (e.g., "Would you like to learn more about AI, tech and digital diplomacy?") are excluded from retrieval using `config/sentence_blocklist.txt`.

### How it works

The blocklist is applied as a **Weaviate-level `not_equal` filter** during hybrid search. This means boilerplate sentences are excluded *before* the search, not after. Weaviate never returns them, so no over-fetching is needed.

**Previous approach (deprecated):** Over-fetch `limit=k_sentences*5` (1000 sentences), then filter in Python. This was slow (~2.4s) because Weaviate had to score 1000 results including ~643 identical boilerplate copies.

**Current approach:** `build_weaviate_blocklist_filter()` generates a Weaviate `Filter.by_property("sentence").not_equal(pattern)` for each blocklist pattern. The filter is combined with the `chunk_level="sentence"` filter via AND. Limit stays at `k_sentences` (200).

### Configuration

- **File:** `config/sentence_blocklist.txt` — one pattern per line, `#` for comments
- **Toggle:** `USE_SENTENCE_BLOCKLIST` in `.env` (default: `True`)
- **Code:** `app/core/blocklist.py` — `build_weaviate_blocklist_filter()`, `get_blocklist()`, `reload_blocklist()`
- **Matching:** Weaviate `not_equal` is exact match (case-sensitive). Each pattern must match the exact sentence text in the database.

### Performance impact

| Metric | Without filter (limit=1000) | With Weaviate filter (limit=200) |
|--------|---------------------------|----------------------------------|
| Hybrid search time | ~2.4s | ~1.5s |
| Sentences returned | 1000 (643 boilerplate) | 200 (all clean) |
| Post-filtering needed | Yes (Python) | No |

### Adding new patterns

1. Add the exact sentence text to `config/sentence_blocklist.txt`
2. Restart the service (patterns are loaded at startup and cached)

## Retrieval Modes

### `sentence` (SentenceFirstRetriever) -- Recommended

Searches `DiploChunk_contextual` at sentence granularity, groups by (URL, section), aggregates scores. Heading hierarchy is encoded in contextual embeddings (no separate heading index).

### `twophase` (TwoPhaseRetriever)

Two-phase approach: Phase 1 discovers top URLs via paragraph-level hybrid search on `DiploParagraph_contextual`; Phase 2 selects the best section per URL via embedding similarity. Higher latency but better for structured multi-section pages.

### `combined` (CombinedRetriever)

Merges results from SentenceFirstRetriever and a parallel paragraph-level search. Best coverage but highest latency.

### `paragraph` (WeaviateHybridRetriever)

Simple paragraph-level hybrid search. Fast but less precise.

## Reranker

**Service:** TEI (Text Embeddings Inference) cross-encoder reranker

**Text formatting per document:**
- `[Label] Title > Section Title`
- `[BEST MATCH]: best_sentence`
- `[FULL CONTEXT]: section_content`
- Truncated to 4000 chars

**Output:** Top `RERANKER_TOP_K` or `SENTENCE_RERANKER_TOP_K` documents sorted by cross-encoder relevance.

## Label Weights

Profile-based label weights adjust the final ranking after reranking:

```
final_score = reranker_score * label_weight[user_profile][document_label]
```

Static weights range from 1.00 to 1.50 across all profiles:

| Profile | Top Priority Labels |
|---------|-------------------|
| General | Topic (1.50) > Course (1.42) > Blog (1.33) > Event (1.25) |
| Student | Course (1.50) > Topic (1.42) > Blog (1.33) > Resources (1.25) |
| Diplomat | Topic (1.50) > Blog (1.42) > Updates (1.33) > Resources (1.25) |
| Researcher | Topic (1.50) > Blog (1.42) > Resources (1.33) > Technologies (1.33) |
| Historian | History (1.50) > Topic (1.50) > Blog (1.42) > Resources (1.33) |
| Journalist | Updates (1.50) > Blog (1.42) > Topic (1.33) > Resources (1.25) |

### Dynamic Label Weight Boosting

When `USE_DYNAMIC_LABEL_WEIGHTS=True` (default), static weights are augmented at query time using prototype embeddings:

1. **Startup:** Prototype questions from `config/prototype_questions.txt` are embedded and averaged into one vector per post_type (11 prototypes total).
2. **Query time:** Cosine similarity is computed between the query vector and each prototype. The dynamic boost is `similarity * DYNAMIC_WEIGHT_ALPHA`.
3. **Formula:** `final_weight = max(static_weight, dynamic_boost)` -- boost can only lift a weight, never reduce it.

This addresses the problem of semantically relevant content being down-ranked by static profile biases. For example, "Who is Jovan Kurbalija?" has high similarity to the `people` prototype (~0.61), producing a dynamic boost of ~1.34 that lifts People (static 1.00) above Event (static 1.25).

**Configuration:**
- `DYNAMIC_WEIGHT_ALPHA` (default: 2.2) -- scale factor for the boost
- `DYNAMIC_WEIGHT_MIN_SIM` (default: 0.3) -- similarity threshold below which no boost is applied
- `config/prototype_questions.txt` -- INI-style file with prototype questions per `[post_type]`

**Per-request overrides:** All three parameters (`use_dynamic_label_weights`, `dynamic_weight_alpha`, `dynamic_weight_min_sim`) can be overridden via `retrieval_config` in the request body.

## Query-Level Redis Cache

Identical queries are cached in Redis to skip the entire retrieval + reranking pipeline on repeat hits.

- **Cache point:** Post-reranker, pre-label-weights (in `direct_retrieval_node`)
- **Cache key:** `ret_cache:{sha256(normalized_query)[:24]}` — user_type independent (label weights are cheap, applied after cache)
- **TTL:** `RETRIEVAL_CACHE_TTL` (default: 120s, 0 = disabled)
- **What's cached:** JSON-serialized `List[Document]` (page_content + metadata)
- **Storage:** Redis via `RedisClient.get_client()` singleton

**On cache HIT:** Skips embedding + Weaviate search + reranker. Label weights are still applied per-request (profile-dependent, <1ms).

**Cache invalidation:** TTL-based only. Flush `ret_cache:*` keys manually after content ingestion if needed.

## Deep Link URL Building

Deep link URLs are built by a single top-level function `build_deep_link_url()` in `chat_service.py`. This replaces 4 previously duplicated copies.

- **Helpers:** `_normalize_url_with_redirect(url)` and `_build_deep_link_text(content, metadata, title, url)`
- **Batch mode:** `build_deep_link_urls_batch(docs, query, deep_link_service)` collects all docs, then calls `DeepLinkService.create_deep_links_batch()` using a single Redis pipeline (1 round-trip instead of N)
- **Single instance:** One `DeepLinkService` per request, not per document

## Timing Breakdown (measured, sentence mode)

Measured on production server with local TEI embedder, local TEI reranker, and local LLM (`gpt-oss-20b`). Query: "what is digital diplomacy?"

| Phase | Cold | Warm | Cache HIT | Notes |
|-------|------|------|-----------|-------|
| Embedding | ~0.11s | ~0.11s | 0s | TEI /embed API call |
| Step 1a (hybrid search) | ~1.8s | ~1.5s | 0s | 200 sentences on `DiploChunk_contextual`, blocklist filter |
| Step 2-3 (aggregation + dedup) | <0.01s | <0.01s | 0s | In-memory |
| **SentenceRetriever total** | **~3.9s** | **~2.1s** | **0s** | Steps 1b/1c (DiploHeading) removed |
| Reranker | ~1.1s | ~1.1s | 0s | TEI cross-encoder, 50 docs |
| Redis cache read | — | — | <2ms | `ret_cache:` key lookup |
| Label weights | <0.001s | <0.001s | <0.001s | In-memory re-sort |
| **Total retrieval** | **~6.2s** | **~3.5s** | **<5ms** | |
| Deep link batch | ~50ms | ~30ms | ~30ms | Redis pipeline for all docs |
| LLM generation | ~5-6s | ~5-6s | ~5-6s | Local LLM, 8 chunks context |
| Related questions | ~2-3s | ~2-3s | ~2-3s | Separate LLM call, local model |
| **End-to-end** | **~11-12s** | **~9-10s** | **~8-9s** | |

### Timing instrumentation

Debug timing output is enabled via `TIMING` prefixed log lines:
```
TIMING SentenceRetriever: embed=0.107s hybrid=1.447s TOTAL=2.337s
TIMING direct_retrieval: retrieval=2.339s reranker=1.131s cache=MISS
TIMING direct_retrieval: retrieval=0.000s reranker=0.000s cache=HIT
CACHE HIT for 'what is digital diplomacy?' (50 docs)
CACHE STORE for 'what is digital diplomacy?' (50 docs, TTL=120s)
DEBUG: Batch created 8 short links
TIMING generate_final: LLM=5.049s
TIMING related_questions: 2.345s
```
