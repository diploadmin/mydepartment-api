# Humainism AI Chatbot API

RAG-powered chatbot API for [diplomacy.edu](https://www.diplomacy.edu), [humainism.ai](https://humainism.ai), and [faicon.ai](https://faicon.ai). Retrieves relevant content from contextual Weaviate collections using a multi-strategy pipeline (hybrid search, cross-encoder reranking, profile-based label weights) and generates answers via LangGraph agents.

## Quick Start

```bash
# Install dependencies
poetry install

# Configure environment
cp .env.example .env
# Edit .env with your credentials (see docs/CONFIGURATION.md)

# Run
poetry run python run_chatbot.py
```

**Prerequisites:** Python 3.12+, Poetry, MongoDB, Redis, Weaviate, TEI Embedder, TEI Reranker (optional)

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System overview, components, LangGraph agent, data layer |
| [Configuration](docs/CONFIGURATION.md) | All environment variables, grouped by category |
| [Observability](docs/OBSERVABILITY.md) | Langfuse tracing for chat, LLM, and LangGraph runs |
| [Retrieval Pipeline](docs/RETRIEVAL_PIPELINE.md) | Full pipeline: contextual hybrid search, section selection, reranker, label weights |
| [API Reference](docs/API.md) | REST + WebSocket endpoints, request/response schemas |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/conversation/get_id` | Create or validate conversation ID |
| POST | `/api/chat/{conversation_id}` | Send message, receive answer + sources |
| WS | `/api/chat/ws/{conversation_id}` | Streaming chat via WebSocket |
| PUT | `/api/chat/feedback/{message_id}` | Submit feedback on a response |
| POST | `/api/ingest/topic` | Ingest a single topic by slug (via chunking API) |
| POST | `/api/ingest/topics/all` | Full reingest of all topics |
| POST | `/api/deep-link/create` | Create a short deep link for sentence highlighting |
| POST | `/api/deep-link/create-batch` | Batch create multiple deep links |
| GET | `/api/deep-link/{dl_id}` | Retrieve deep link data |
| POST | `/api/debug/retrieve` | Debug retrieval (full pipeline introspection) |

## MyDepartment Chatbot Gateway

Calls a chatbot built in the Chatbot Generator, identified either by its uid or
by the public share link it is handed out with. Runs go through the Department
backend's public endpoints — the same path the shareable link takes in a
browser — so only publicly shared chatbots are reachable, and answers match
what that link produces.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/chatbot/list` | Every chatbot: name, organization, owner, public link |
| GET | `/api/chatbot/{chatbot_uid}` | Chatbot name and default model |
| POST | `/api/chatbot/{chatbot_uid}` | Ask by uid, returns the finished answer |
| POST | `/api/chatbot/{chatbot_uid}/stream` | Same, as server-sent events |
| POST | `/api/chatbot/invoke` | Ask by `chatbot_uid` or `public_link` in the body |
| POST | `/api/chatbot/stream` | Same, as server-sent events |

Authenticate with `CHATBOT_INVOKE_API_KEY` as either `X-API-Key` or
`Authorization: Bearer`. With no key configured the routes stay behind the
`ALLOWED_IPS` allowlist instead.

```bash
curl -X POST https://mydepartment-api.mydepartment.ai/api/chatbot/invoke \
  -H "X-API-Key: $CHATBOT_INVOKE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"public_link": "https://mydepartment.ai/coworkers/chatbot/public/<uid>",
       "message": "What can you help me with?"}'
```

Optional `prompt` replaces the chatbot's stored system prompt for that turn
only. Leave it out (or send it blank) and the prompt saved on the chatbot is
used. It is not written back to the chatbot.

```bash
curl -X POST https://mydepartment-api.mydepartment.ai/api/chatbot/invoke \
  -H "X-API-Key: $CHATBOT_INVOKE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"chatbot_uid": "<uid>",
       "prompt": "Answer in one sentence, in Serbian.",
       "message": "What can you help me with?"}'
```

The response carries a `thread_id`; send it back in the next request to
continue that conversation. The streaming routes return it as `X-Thread-Id`
before the first event.

`/api/chatbot/list` covers private chatbots too, since it is the catalog an
operator needs; `access_level` says which ones actually answer on their
`public_link`. It reads the Department backend's `/assistants/catalog`, so it
needs `DEPARTMENT_API_KEY` (that backend's own service key) on top of the
gateway key.

```bash
curl https://mydepartment-api.mydepartment.ai/api/chatbot/list \
  -H "X-API-Key: $CHATBOT_INVOKE_API_KEY"
```

```json
{"count": 1146,
 "chatbots": [{"chatbot_uid": "155fba2a-...", "name": "Zimbabwe",
               "organization_id": "5a592e74-...", "organization_name": "Diplo Team",
               "owner": "Marko Markovic", "owner_email": "markom@diplomacy.edu",
               "public_link": "https://mydepartment.ai/coworkers/chatbot/public/155fba2a-...",
               "access_level": "public"}]}
```

## Retrieval Strategies

Four retrieval modes, configurable via `RETRIEVAL_MODE`:

- **`sentence`** (recommended) -- Sentence-level hybrid search on `DiploChunk_contextual`, grouped by (URL, section), cross-encoder reranking, and URL cap/dedup.
- **`combined`** -- Merges SentenceFirstRetriever + parallel paragraph-level hybrid search on `DiploParagraph_contextual`, deduplicates by URL.
- **`twophase`** -- Phase 1: URL discovery via paragraph hybrid search. Phase 2: parallel fetch + embedding-based section selection.
- **`paragraph`** -- Simple paragraph-level hybrid search from contextual collections with label rescoring.

All strategies include a **visibility filter** that excludes chunks marked as `visibility=private` (e.g. transcripts), while including all objects with null or non-private visibility.

### Pipeline Flow (Sentence Mode)

```
Query
  -> Embed (TEI)
  -> Hybrid search on DiploChunk_contextual (200 sentences, alpha=0.6)
  -> Group by (URL, section)
  -> Aggregate scores (max*0.6 + avg*0.3 + log(1+count)*0.1)
  -> Cross-encoder best-sentence selection (TEI reranker)
  -> URL cap + dedup
  -> Label weights (profile-based + dynamic query-intent boost)
  -> Top K sections -> LangGraph agent -> LLM response with citations
```

## Per-Request Configuration

The `RetrievalConfig` schema allows overriding 25+ retrieval parameters per chat request, including hybrid alpha, reranker settings, sentence retrieval K, contextual collections toggle, content filters (`site_filter_name`, `parent_filter_name`, `person_filter_name`, `person_filter_names`, `city_filter_name`, `city_filter_names`, `country_filter_name`, `country_filter_names`, `organisation_filter_name`, `organisation_filter_names`, `post_types`), label weight parameters, and optional **recency boost** (`use_recency_boost`, default off).

### Recency boost (optional, default off)

When `retrieval_config.use_recency_boost` is `true`, documents published within the last 5 years receive a soft score multiplier after reranking (newer = higher; posts older than 5 years or missing dates stay at multiplier 1.0). All content remains searchable — this is not a Weaviate date filter.

```json
"retrieval_config": {
  "use_recency_boost": true,
  "recency_boost_max_age_days": 1825,
  "recency_half_life_days": 900,
  "recency_max_boost": 0.35
}
```

## User Profiles

Eight user types with distinct label weight priorities: `general`, `student`, `diplomat`, `researcher`, `historian`, `philosopher`, `contrarian`, `journalist`.

## Key Features

- **Multi-website support** -- diplomacy.edu, humainism.ai, faicon.ai with per-site system prompts
- **Streaming** -- WebSocket token-by-token streaming with live citation renumbering
- **Deep links** -- Redis-based URL shortening for sentence/paragraph highlighting
- **LLM failsafe** -- Automatic fallback to secondary LLM provider on failure
- **Retrieval cache** -- Redis-backed post-reranker cache (120s TTL)
- **Sentence blocklist** -- Weaviate-level filter for boilerplate content
- **Force-include** -- Guarantees URLs with matching titles/headings enter the reranker pool
- **Dynamic label weights** -- Query-intent prototype embeddings boost relevant post types at query time
- **Recency boost** -- Optional post-reranker multiplier for posts newer than 5 years (`use_recency_boost`, off by default)
- **Visibility filtering** -- Chunks with `visibility=private` are excluded from retrieval by default
- **Contextual collections** -- Optional `_contextual` Weaviate collections with enriched embeddings
- **IP allowlist** -- Middleware supporting CIDR notation
- **Langfuse tracing** -- Optional observability to self-hosted Langfuse (`LANGFUSE_*` env vars)

## Tech Stack

- **FastAPI** -- REST + WebSocket API
- **LangGraph** -- Stateful agent with tool calling
- **LangChain** -- LLM orchestration (OpenAI, Anthropic, local)
- **Weaviate** -- Vector database (contextual collections: Document, Paragraph, Chunk, Turn)
- **TEI** -- Text Embeddings Inference (embedder + cross-encoder reranker)
- **MongoDB** -- Chat persistence (messages, feedback)
- **Redis** -- Retrieval cache + deep link URL shortening
- **SQLite** -- Conversation memory (LangGraph checkpointer)

## Deployment

Docker Compose with two services:

1. **`api`** -- Multi-stage Python 3.12-slim, non-root user, port 8561
2. **`redis`** -- Redis 7 Alpine with AOF persistence, 2GB max memory, port 6380

```bash
docker-compose up -d
```

External dependencies (not in compose): Weaviate, MongoDB, TEI Embedder, TEI Reranker, LLM provider, Langfuse (optional).

## Project Structure

```
app/
  ai/
    ai_services/
      graph.py                  # LangGraph agent, tool nodes, graph construction
      chatbot.py                # Legacy chatbot entry point
      label_weights.py          # Static profile-based label weights
      query_intent.py           # Dynamic label weight boosting via prototype embeddings
      reranker.py               # TEI cross-encoder reranker
      embeddings.py             # TEI Embeddings wrapper (LangChain-compatible)
      retrieval_cache.py        # Redis cache helpers
      retrievers/
        factory.py              # Retriever factory (mode-based creation)
        sentence_first.py       # SentenceFirstRetriever
        combined.py             # CombinedRetriever (sentence + paragraph merge)
        twophase.py             # TwoPhaseRetriever (URL discovery + section selection)
        label_rescoring.py      # LabelRescoringRetriever (paragraph-level)
        utils.py                # Shared: grouping, scoring, section selection, visibility_filter
    prompts/                    # System prompts per website
  api/
    routes/
      chat_route.py             # Chat REST + WebSocket endpoints
      conversation_route.py     # Conversation ID management
      ingest_route.py           # Topic ingestion (chunking API proxy)
      deep_link_route.py        # Deep link URL shortening (Redis)
      debug_route.py            # Debug retrieval endpoint
    errors/                     # HTTP + validation error handlers
  core/
    config.py                   # Environment configuration (~70+ variables)
    events.py                   # Startup/shutdown lifecycle handlers
    singleton.py                # Shared state (agent, retrievers, WS manager, Weaviate client)
    websocket_manager.py        # WebSocket connection pool
    retrieval_context.py        # Per-request retrieval overrides (contextvars)
    blocklist.py                # Sentence blocklist (Weaviate-level filtering)
  db/                           # MongoDB connection management
  middleware/                   # IP allowlist middleware (CIDR support)
  models/                       # MongoEngine document models
  repositories/                 # MongoDB CRUD operations
  schemas/                      # Pydantic request/response models
  services/
    chat_service.py             # Chat orchestration, streaming, citations, deep links
    deep_link_service.py        # Redis-based deep link shortening
    ingest_service.py           # Topic ingestion pipeline
    conversation_service.py     # Conversation ID management
    user_type_service.py        # User type -> prompt mapping
  utils/redis_client.py         # Redis singleton client
config/
  sentence_blocklist.txt        # Boilerplate sentences excluded from retrieval
  prototype_questions.txt       # Prototype questions per post_type (dynamic label weights)
scripts/
  ingest_topics.py              # CLI for topic ingestion
  delete_topics.py              # Delete topic data from Weaviate
  benchmark_speed.py            # API speed benchmarking
docs/                           # Architecture, Configuration, Pipeline, API docs
main.py                         # FastAPI app factory
run_chatbot.py                # Uvicorn launcher
```
