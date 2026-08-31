# Configuration

All configuration is loaded from an `.env` file via `app/core/config.py`. The env file path is set by the `ENV_FILE` environment variable (default: `.env`).

## Application

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WEBSITE_NAME` | str | *required* | Website name identifier |
| `HOST` | str | `127.0.0.1` | Server bind address |
| `PORT` | int | `8560` | Server port |
| `RELOAD` | bool | `True` | Uvicorn auto-reload |
| `DEBUG` | bool | `False` | Debug mode (enables DEBUG-level logging) |
| `PROJECT_NAME` | str | `FastAPI example application` | FastAPI app title |
| `SECRET_KEY` | str | *required* | Application secret key |

## Database (MongoDB)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DB_CONNECTION` | str | *required* | MongoDB connection URI |
| `DB_HOST` | str | `127.0.0.1` | MongoDB host |
| `DB_PORT` | int | `3306` | MongoDB port |
| `DB_USER` | str | *required* | MongoDB username |
| `DB_PWD` | str | `""` | MongoDB password |
| `DB_NAME` | str | `""` | MongoDB database name |
| `MAX_CONNECTIONS_COUNT` | int | `10` | Connection pool max |
| `MIN_CONNECTIONS_COUNT` | int | `10` | Connection pool min |

## Weaviate

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WV_CLIENT_URL` | str | *required* | Weaviate HTTP URL (e.g. `http://127.0.0.1:8080`) |
| `WV_KEY` | str | *required* | Weaviate API key |
| `WV_GRPC_PORT` | int | `0` | Weaviate gRPC port (0 = auto-detect) |
| `INDEX_NAME` | str | *required* | Primary Weaviate collection (e.g. `DiploParagraph`) |
| `INDEX_NAME2` | str | *required* | Secondary collection (can be same as INDEX_NAME) |
| `SENTENCE_INDEX_NAME` | str | `DiploChunk` | Collection for sentence-level retrieval |

## LLM

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LLM_MODEL_PROVIDER` | str | `openai` | LLM provider: `openai`, `anthropic`, `local`, or `deepseek` |
| `LLM_MODEL_NAME` | str | `gpt-4o` | Model name for OpenAI/Anthropic/DeepSeek |
| `LOCAL_LLM_URL` | str | `""` | Base URL for local LLM (OpenAI-compatible) |
| `LOCAL_LLM_MODEL` | str | `""` | Local LLM model name |
| `LOCAL_LLM_KEY` | str | `""` | Local LLM API key |
| `DEEPSEEK_API_KEY` | str | `""` | DeepSeek API key |
| `DEEPSEEK_API_URL` | str | `https://api.deepseek.com/v1` | DeepSeek API base URL |

## Embedding

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EMBEDDING_PROVIDER` | str | `openai` | Embedding provider: `openai` or `local` |
| `EMBEDDING_MODEL_NAME` | str | `text-embedding-3-large` | OpenAI embedding model |
| `LOCAL_EMBEDDING_URL` | str | `""` | TEI embedder service URL |
| `LOCAL_EMBEDDING_KEY` | str | `""` | TEI embedder API key |

## Retrieval

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RETRIEVAL_MODE` | str | `sentence` | Retrieval strategy: `twophase`, `combined`, `sentence`, `paragraph` |
| `RETRIEVAL_CHUNKS` | int | `8` | Final number of chunks sent to LLM |
| `HYBRID_ALPHA` | float | `0.85` | Vector/BM25 balance (Weaviate): 0.0=pure BM25, 0.6=current production, 1.0=pure vector |
| `SENTENCE_RETRIEVAL_K` | int | `200` | Sentences fetched in sentence/combined mode |
| `PARAGRAPH_RETRIEVAL_K` | int | `20` | Paragraphs fetched in paragraph/combined mode |
| `PARAGRAPH_FETCH_LIMIT` | int | `500` | Raw paragraphs from hybrid search before aggregation |
| `USE_SENTENCE_BLOCKLIST` | bool | `True` | Weaviate-level blocklist filter via `config/sentence_blocklist.txt` (see [RETRIEVAL_PIPELINE.md](RETRIEVAL_PIPELINE.md#sentence-blocklist)) |
| `USE_SENTENCE_RETRIEVAL` | bool | `False` | *Deprecated.* Use `RETRIEVAL_MODE` instead |

## TwoPhase Retrieval

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TWOPHASE_K_URLS` | int | `40` | URLs passed from Phase 1 to Phase 2 |
| `HEADING_BOOST_WEIGHT` | float | `0.5` | *Unused* (legacy; DiploHeading removed) |

## Force-Include

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FORCE_INCLUDE_H1` | bool | `True` | Match query terms against page title (h1) |
| `FORCE_INCLUDE_SECTION` | bool | `False` | Match query terms against section headings |
| `FORCE_INCLUDE_OVERLAP` | float | `0.6` | Minimum term overlap ratio to trigger force-include |
| `FORCE_INCLUDE_MAX` | int | `7` | Maximum force-included URLs |

## Legacy heading boost/injection (unused)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HEADING_INJECT_MIN_SIM` | float | `0.65` | *Unused* — DiploHeading collection removed |
| `HEADING_INJECT_MAX` | int | `10` | *Unused* |
| `HEADING_INJECT_MIN_SENTS` | int | `5` | *Unused* |

## Reranker

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_RERANKER` | bool | `False` | Enable TEI reranker |
| `RERANKER_URL` | str | `""` | TEI reranker service URL |
| `RERANKER_API_KEY` | str | `""` | TEI reranker API key |
| `RERANKER_TOP_K` | int | `20` | Documents sent to reranker |
| `TWOPHASE_RERANKER_TOP_K` | int | `0` | TwoPhase-specific reranker candidates (0 = use global) |
| `SENTENCE_RERANKER_TOP_K` | int | `0` | Sentence-mode reranker candidates (0 = use global) |
| `MAX_SECTIONS_PER_URL` | int | `0` | Max sections per URL sent to reranker (0 = no limit) |
| `TOP_N_PER_SECTION` | int | `0` | Max scoring sentences per section for aggregation (0 = no limit) |

## Dynamic Label Weights

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_DYNAMIC_LABEL_WEIGHTS` | bool | `True` | Enable query-intent dynamic label weight boosting. Computes cosine similarity between query and post_type prototype embeddings at query time. Uses `max(static, dynamic)` so boost can only lift, never reduce. |
| `DYNAMIC_WEIGHT_ALPHA` | float | `2.2` | Boost scale factor: `dynamic_boost = similarity * alpha`. With alpha=2.2, similarity 0.7 produces boost 1.54. |
| `DYNAMIC_WEIGHT_MIN_SIM` | float | `0.3` | Minimum cosine similarity threshold. Below this, no dynamic boost is applied for that post_type. |

Prototype questions are loaded from `config/prototype_questions.txt` at startup. The file uses INI-style sections (`[post_type]`) with one question per line.

## Caching

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RETRIEVAL_CACHE_TTL` | int | `120` | Query-level Redis cache TTL in seconds. Caches post-reranker docs to skip retrieval+reranker on repeat queries. 0 = disabled. |

## Deep Links

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_SHORT_DEEP_LINKS` | bool | `True` | Use Redis-based short deep link IDs instead of URL-encoded text |
| `DEEP_LINK_HIGHLIGHT_MODE` | str | `paragraph` | Deep link highlight mode: `sentence` or `paragraph` |
| `SENTENCE_HIGHLIGHT_MODE` | str | `single` | Sentence highlight mode: `single` (best sentence) or `multi` (all matched) |
| `MULTI_HIGHLIGHT_MAX_SENTS` | int | `5` | Max sentences in multi-highlight deep links (0 = no limit) |
| `REDIS_HOST` | str | `localhost` | Redis host for deep links and cache |
| `REDIS_PORT` | int | `6379` | Redis port |
| `REDIS_DB` | int | `0` | Redis database number |
| `DEEP_LINK_TTL_DAYS` | int | `30` | Deep link TTL in days |

## Agent Behavior

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SKIP_TOOL_DECISION` | bool | `False` | Skip LLM tool-selection call, directly retrieve (saves ~1s) |
| `MAX_TOOL_CALLS` | int | `1` | Max tool calls per ReAct loop |
| `CITE_SOURCES` | bool | `True` | LLM cites sources as [1], [2], etc. |
| `USE_SOURCE_FILTER` | bool | `True` | Additional LLM call to filter irrelevant sources |

## Related Questions

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RELATED_QUESTIONS_PROVIDER` | str | `openai` | Provider: `openai`, `local`, or `deepseek` |
| `RELATED_QUESTIONS_OPENAI_MODEL` | str | `gpt-4o-mini` | OpenAI model for related questions |
| `RELATED_QUESTIONS_DEEPSEEK_MODEL` | str | `deepseek-v4-flash` | DeepSeek model for related questions |
| `RELATED_QUESTIONS_LOCAL_URL` | str | `""` | Local model URL |
| `RELATED_QUESTIONS_LOCAL_MODEL` | str | `""` | Local model name |
| `RELATED_QUESTIONS_LOCAL_KEY` | str | `""` | Local model API key |

## Langfuse (observability)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LANGFUSE_ENABLED` | bool | `False` | Enable Langfuse tracing for LLM / LangGraph runs |
| `LANGFUSE_PUBLIC_KEY` | str | `""` | Project public key from Langfuse UI |
| `LANGFUSE_SECRET_KEY` | str | `""` | Project secret key from Langfuse UI |
| `LANGFUSE_HOST` | str | `https://langfuse.diplomacy.edu` | Langfuse API base URL (self-hosted) |
| `LANGFUSE_DEBUG` | bool | `False` | Log Langfuse SDK debug output (useful for credential issues) |

Traces include session id (`conversation_id`), user id (`user_ip`), and tags (`WEBSITE_NAME`, flow name, `user_type`).

Full setup, traced flows, and troubleshooting: [OBSERVABILITY.md](OBSERVABILITY.md).

**Docker:** `env_file: .env` in `docker-compose.yml` injects variables into the container; the file is not mounted at `/app/.env`. The app reads process environment in that case (see `app/core/config.py`).

### Dev stack (`/opt/dev/chatbot-api-dev`)

Root `docker-compose.yml` for this workspace:

| Service | Container | Host port | Notes |
|---------|-----------|-----------|-------|
| `api` | `chatbot-api-dev` | `8561` | `WV_CLIENT_URL=http://ai6000.diplomacy.edu:8591` (override in compose) |
| `subgraphs-nginx` | `chatbot-api-dev_subgraphs_nginx` | `127.0.0.1:8565` | Serves `./subgraphs` HTML exports |
| `redis` | `chatbot-api-dev_redis` | `127.0.0.1:6380` | Data volume: `./redis_data` |

Docker network: `chatbot-api-dev_network`. Generated subgraph HTML under `./subgraphs/` is local debug output (not committed).

## Security

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ALLOWED_HOSTS` | CSV | `""` | CORS allowed origins (comma-separated) |
| `ALLOWED_IPS` | CSV | `""` | IP allowlist (comma-separated) |
| `OPENAI_KEY` | str | *required* | OpenAI API key |
| `DEEPSEEK_API_KEY` | str | `""` | DeepSeek API key (required when provider is `deepseek`) |
| `OPENAI_ORGANIZATION` | str | *required* | OpenAI organization ID |
