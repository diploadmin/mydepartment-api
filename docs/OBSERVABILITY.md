# Observability (Langfuse)

The API can send LLM and LangGraph traces to a self-hosted [Langfuse](https://langfuse.com/) instance (e.g. `https://langfuse.diplomacy.edu`).

## What is traced

| Run name | When |
|----------|------|
| `chat_http` | POST `/api/chat/{conversation_id}` — full LangGraph RAG pipeline |
| `chat_ws` | WebSocket `/api/chat/ws/{conversation_id}` — same pipeline, streaming |
| `related_questions` | LLM call to generate follow-up questions when metadata has none |
| `source_filter` | LLM call when `USE_SOURCE_FILTER=true` filters cited sources |

Each trace includes nested spans for retrieval, reranking, graph expansion, and LLM generations (via LangChain `CallbackHandler`).

## Metadata in Langfuse

| Field | Source |
|-------|--------|
| **Session** | `conversation_id` (`langfuse_session_id`) |
| **User** | `user_ip` from the chat request (`langfuse_user_id`) |
| **Tags** | `WEBSITE_NAME`, run name (`chat_http` / `chat_ws`), `user_type` |

## Configuration

See [CONFIGURATION.md](CONFIGURATION.md#langfuse-observability) for environment variables.

```env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://langfuse.diplomacy.edu
LANGFUSE_DEBUG=false   # set true to debug SDK / credential issues
```

Keys are created in Langfuse → **Project** → **Settings** → **API Keys**.

## Implementation

- `app/core/langfuse_tracing.py` — `CallbackHandler`, run config builders, client init, flush on shutdown
- `app/core/events.py` — `init_langfuse()` on startup, `flush_langfuse()` on shutdown
- `app/services/chat_service.py` — passes Langfuse config into `astream_events`, related questions, and source filter

Langfuse Python SDK **v3** reads `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST` from the process environment; only `public_key` is passed to `CallbackHandler()`.

## Docker

`docker-compose.yml` loads variables with `env_file: .env`. The `.env` file is **not** copied into the image; vars are injected into the container environment. `app/core/config.py` uses a `.env` file only when it exists on disk, otherwise Starlette `Config()` reads process env only (no spurious “file not found” warning).

## Troubleshooting

| Symptom | Action |
|---------|--------|
| Chat returns 500 right after enabling Langfuse | Check API logs; ensure SDK v3 (no `secret_key` on `CallbackHandler`). Rebuild image after code updates. |
| No traces in UI | Verify keys and `LANGFUSE_HOST`; set `LANGFUSE_DEBUG=true` and inspect container logs for `UnauthorizedError`. |
| Traces missing LLM spans | Confirm `LANGFUSE_ENABLED=true` and both keys are set; restart API after `.env` changes. |
