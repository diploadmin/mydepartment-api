# Architecture

## Overview

The Humainism AI Chatbot API is a FastAPI-based RAG (Retrieval-Augmented Generation) system. It receives user questions via REST/WebSocket, retrieves relevant content from a Weaviate vector database using a multi-phase pipeline, and generates answers with an LLM via LangGraph.

```
Client (REST / WebSocket)
        |
   FastAPI + Middleware (CORS, IP allowlist)
        |
   +-----------+-----------+
   |                       |
ChatService          IngestService
   |                       |
LangGraph Agent      Chunking API + Weaviate
   |
   +-----------+----------+
   |                      |
Retrieval Pipeline     LLM (OpenAI / Anthropic / Local)
   |
Weaviate (DiploParagraph_contextual, DiploChunk_contextual, DiploDocument_contextual)
   |
TEI Reranker --> Label Weights --> Final Context
```

## Core Components

### 1. FastAPI Application (`main.py`)

- Entry point: `get_application()` creates the FastAPI app
- Middleware: CORS (`ALLOWED_HOSTS`), IP allowlist (`ALLOWED_IPS`)
- Routes registered under `/api` prefix
- Error handlers for HTTP and validation errors

### 2. API Routes (`app/api/routes/`)

| Route | Method | Description |
|-------|--------|-------------|
| `/api/chat/{conversation_id}` | POST | Main chat endpoint |
| `/api/chat/ws/{conversation_id}` | WS | Streaming chat via WebSocket |
| `/api/chat/feedback/{message_id}` | PUT | Submit feedback on a response |
| `/api/conversation/get_id` | POST | Create or validate conversation ID |
| `/api/ingest/topic` | POST | Ingest a single topic via chunking API |
| `/api/ingest/topics/all` | POST | Full reingest of all topics |

### 3. Service Layer (`app/services/`)

- **ChatService** -- Orchestrates the chat flow: receives request, invokes LangGraph agent, saves messages/responses to MongoDB, streams via WebSocket with live citation renumbering. Deep link URL building via top-level `build_deep_link_url()` and `build_deep_link_urls_batch()` functions (with helpers `_normalize_url_with_redirect()`, `_build_deep_link_text()`)
- **IngestService** -- Topic ingestion pipeline: calls external chunking API; chunking API writes contextual Weaviate collections
- **ConversationService** -- Manages conversation IDs and validation
- **UserTypeService** -- Maps user type to profile-specific prompts and label weights
- **RepositoryService** -- Abstraction over MongoDB operations

### 4. AI Layer (`app/ai/`)

#### LangGraph Agent (`app/ai/ai_services/chatbot.py`)

Two graph compilation modes controlled by `SKIP_TOOL_DECISION`:

**SKIP_TOOL_DECISION=True** (recommended, saves ~1s):
```
START --> direct_retrieval --> generate_final --> END
```
Skips the LLM call that decides to use `diplo_tool`. Directly calls the retrieval pipeline and passes results to the LLM for answer generation.

**SKIP_TOOL_DECISION=False** (full ReAct agent):
```
START --> chatbot --> tools_condition --> tools --> after_tools_condition --> generate_final --> END
```
LLM decides which tool to call. `MAX_TOOL_CALLS` limits the ReAct loop (default: 1).

#### State Definition
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_count: int
```

#### Retrieval Modes

Selected by `RETRIEVAL_MODE` env variable:

| Mode | Retriever Class | Description |
|------|----------------|-------------|
| **`sentence`** | **`SentenceFirstRetriever`** | **Sentence-level search with aggregation (recommended)** |
| `twophase` | `TwoPhaseRetriever` | Hybrid search + section selection |
| `combined` | `CombinedRetriever` | Sentence + paragraph search merged |
| `paragraph` | `LabelRescoringRetriever` | Direct paragraph-level search |

See [RETRIEVAL_PIPELINE.md](RETRIEVAL_PIPELINE.md) for the full pipeline breakdown.

### 5. Redis (Cache & Deep Links)

Redis serves two purposes:

1. **Query-level retrieval cache** (`ret_cache:*` keys) — Caches post-reranker documents in `direct_retrieval_node`. On cache HIT, skips embedding + Weaviate + reranker (~3-9s saved). TTL: `RETRIEVAL_CACHE_TTL` (default 120s). Cache helpers in `chatbot.py`: `_cache_key()`, `_serialize_docs()`, `_deserialize_docs()`.

2. **Deep link URL shortening** (`deep_link:*` keys) — Stores highlight data (section text + matched sentences) for short `?diplo-hl-id=XYZ` URLs instead of long URL-encoded text. TTL: 30 days. Managed by `DeepLinkService` in `app/services/deep_link_service.py`. Batch creation via `create_deep_links_batch()` uses a Redis pipeline (1 round-trip).

Connection: `RedisClient` singleton in `app/utils/redis_client.py`.

### 6. Embeddings & Reranker

#### TEI Embeddings
Custom `TEIEmbeddings` class calls a Text Embeddings Inference (TEI) service for embedding generation. Falls back to OpenAI embeddings when `EMBEDDING_PROVIDER=openai`.

#### TEI Reranker
`TEIReranker` class calls a cross-encoder reranker service. Enabled via `USE_RERANKER=True`. Receives candidate documents, sends query+text pairs, returns re-scored top-K results.

#### Profile-Based Label Weights + Dynamic Boosting
After reranking, documents are re-scored based on user profile label preferences. 8 profiles: General, Student, Diplomat, Researcher, Historian, Philosopher, Contrarian, Journalist. Each profile assigns different multipliers (1.00-1.50 range) to content labels (Topic, Course, Blog, Event, etc.).

When `USE_DYNAMIC_LABEL_WEIGHTS=True`, static weights are augmented by query-intent prototype embeddings. At startup, prototype questions from `config/prototype_questions.txt` are embedded and averaged into one vector per post_type. At query time, cosine similarity between the query and each prototype determines a dynamic boost (`similarity * alpha`). The final weight is `max(static, dynamic)`, so the boost can only lift a label weight, never reduce it. See [RETRIEVAL_PIPELINE.md](../docs/RETRIEVAL_PIPELINE.md#dynamic-label-weight-boosting).

### 7. Data Layer

#### Weaviate Collections

| Collection | Purpose | Key Fields |
|-----------|---------|--------|
| `DiploParagraph_contextual` | Paragraph-level content (enriched embeddings) | text, link, h1-h6, section, post_type, last_h_title |
| `DiploChunk_contextual` | Sentence-level content (enriched embeddings) | sentence, context, link, h1, section_path, post_type |
| `DiploDocument_contextual` | Document metadata | name, link, site, post_type, full_text |

#### MongoDB
- Chat messages: user_ip, message, timestamp, conversation_id
- Chat responses: message_id, response, response_time, feedback_type, feedback_message

#### SQLite
- LangGraph conversation memory (thread-based checkpointing)

### 8. Ingestion Pipeline

The ingestion pipeline handles populating Weaviate with content from WordPress:

1. **Chunking API** (`nested_chunking_api`) -- Ingests topic content and writes `DiploDocument_contextual`, `DiploParagraph_contextual`, and `DiploChunk_contextual` with heading hierarchy baked into contextual embeddings.

Available via:
- **API:** `POST /api/ingest/topic`, `POST /api/ingest/topics/all`
- **CLI:** `scripts/ingest_topics.py`

## Observability (Langfuse)

When `LANGFUSE_ENABLED=true`, chat requests emit traces to the configured Langfuse host. `ChatService` attaches a LangChain `CallbackHandler` to LangGraph `astream_events` and to auxiliary LLM calls (related questions, source filter). Startup initializes the Langfuse client from env; shutdown flushes pending events.

See [OBSERVABILITY.md](OBSERVABILITY.md) for configuration and troubleshooting.

## Project Structure

```
app/
  ai/
    ai_services/
      graph.py                 # LangGraph agent, tool nodes, graph construction
      label_weights.py         # Static profile label weights + get_label_weights()
      query_intent.py          # Dynamic label weight boosting via prototype embeddings
      reranker.py              # TEIReranker cross-encoder
      embeddings.py            # TEIEmbeddings wrapper
      retrievers/              # SentenceFirst, Combined, TwoPhase, LabelRescoring retrievers
    prompts/                   # System prompts per website
  api/
    routes/
      chat_route.py            # Chat REST + WebSocket endpoints
      conversation_route.py    # Conversation ID management
      ingest_route.py          # Topic ingestion + heading management
    errors/                    # HTTP + validation error handlers
  core/
    config.py                  # All env variable loading (~60 variables)
    langfuse_tracing.py        # Langfuse CallbackHandler and run config
    blocklist.py               # Sentence blocklist: load, filter, Weaviate filter builder
    events.py                  # Startup/shutdown handlers
    logging.py                 # Loguru setup
    singleton.py               # Shared state (agent, retrievers, WS manager, Weaviate client)
    websocket_manager.py       # WebSocket connection management
  db/                          # MongoDB connection management
  middleware/ip_check.py       # IP allowlist middleware
  models/                      # MongoEngine document models
  repositories/                # MongoDB CRUD
  schemas/                     # Pydantic request/response schemas
  services/
    chat_service.py            # Chat orchestration, streaming, citations, deep link URL building
    deep_link_service.py       # Redis-based deep link shortening (create, batch, retrieve)
    ingest_service.py          # Topic ingestion pipeline
    conversation_service.py    # Conversation ID management
    user_type_service.py       # User type to prompt mapping
  utils/
    redis_client.py            # Redis singleton client
config/
  sentence_blocklist.txt       # Boilerplate sentences excluded from retrieval
  prototype_questions.txt      # Prototype questions per post_type for dynamic label weights
docs/                          # Documentation
scripts/                       # CLI utilities (ingest, headings, benchmarks)
main.py                        # FastAPI app factory
run_chatbot.py               # Uvicorn runner with ENV_FILE
```
