# Neo4j Graph Expansion — Production Deployment

Record of the change that brought Neo4j graph expansion to
`chat-api.humainism.ai` (container `humainism_api` on port `8561`, served by
nginx upstream `chat-api-humainism_backend`).

> **Date:** 2026-04-21
> **Env:** `/opt/dev/humainism_ai_chatbot_api/` on the production host
> **Public URL:** <https://chat-api.humainism.ai>

---

## 1. Background

Two Humainism API deployments coexist on this host:

| Container            | Port | Public URL                             | Repo path                                       |
|----------------------|------|----------------------------------------|-------------------------------------------------|
| `humainism_api`      | 8561 | `https://chat-api.humainism.ai`        | `/opt/dev/humainism_ai_chatbot_api/`            |
| `humainism_api_test` | 8570 | `https://humainism.ai/knowledge-graph` | `/opt/dev/humainism_ai_chatbot_api anja/`       |

Both clone the same GitHub repo (`diploadmin/humainisim_ai_chatbot_api`) but
had diverged:

- `anja` (test) had the Neo4j graph expansion commit
  (`ecb11a9 Add Neo4j graph expansion service and related configurations`)
  plus uncommitted working-tree polish on top of it (extended
  `neo4j_config.py`, `chat_service.py` subgraph build with hop3 relations,
  tweaks to `graph_expansion*.py`, `neo4j_graph_client.py`,
  `debug_route.py`, `schemas/chat_schema.py`).
- `prod` was ahead of `anja` on three commits
  (`44d64f0 post_types whitelist filter`, `94b065d resource/histories labels`,
  `cbcc765 visibility_filter is_none drop`) but had **no** Neo4j code.

Goal: give `chat-api.humainism.ai` the same graph-expansion capabilities the
test deployment already had, while keeping the post_types/visibility fixes
that only live on prod.

Neo4j itself is an **external** service at
`bolt://nimani.diplomacy.edu:7687` (two databases: `weaviatediplo`,
`weaviatedw`). No Neo4j container was added on this host.

---

## 2. What changed in the repo

### 2.1 Cherry-picked commit

```
b84ec1b  Add Neo4j graph expansion service and related configurations
         (cherry-picked from anja/master ecb11a9)
```

Conflict on `chatbot-api/app/ai/ai_services/retrievers/utils.py` resolved by
keeping the HEAD version: prod already had the same `visibility_filter`
fix plus the newer `post_types_filter()` helper; the Neo4j commit only
re-applied the visibility fix, so HEAD wins without losing the
`post_types_filter`.

Backup branch left behind for easy rollback:
```
backup-before-neo4j-cherrypick  →  cbcc765
```

### 2.2 Working-tree patch applied on top

`anja` had ~1.7k lines of uncommitted polish that the running
`humainism_api_test` container depended on at build time (Docker copies the
working tree, not `HEAD`). The patch was extracted with
`git diff HEAD` from the `anja` repo and applied cleanly on prod. Ten files
touched, no conflicts:

```
chatbot-api/app/ai/ai_services/graph.py
chatbot-api/app/ai/ai_services/graph_expansion.py
chatbot-api/app/ai/ai_services/graph_expansion_node.py
chatbot-api/app/ai/ai_services/neo4j_graph_client.py
chatbot-api/app/api/routes/debug_route.py
chatbot-api/app/core/neo4j_config.py          (added GRAPH_CONTEXT_FORMAT, GRAPH_BOOST_ALPHA, GRAPH_BOOST_POOL_K)
chatbot-api/app/schemas/chat_schema.py
chatbot-api/app/services/chat_service.py      (subgraph build + hop3 expansion)
chatbot-api/scripts/benchmark_graph_rag.py
docker-compose.test.yml
```

These are **not committed** yet on prod; the anja team is expected to commit
and push them upstream in their own time. Until then the prod container
relies on the working tree at build time — identical workflow to how
`humainism_api_test` was built.

### 2.3 Secrets moved out of `docker-compose.yml`

`docker-compose.yml` used to carry Neo4j URI, credentials and flags as
inline `environment:` entries. Those now live only in `.env` (see
`.env.example` for the shape). Compose reads `.env` via `env_file:` and
forwards every key to the container. Benefit: ops can flip
`GRAPH_EXPANSION_ENABLED` without editing compose and credentials stop
appearing in the tracked file.

### 2.4 `.env.example` templates

- **`.env.example`** at repo root — **new**. Sanitized copy of the real
  `.env` used by docker-compose; every secret replaced with
  `REPLACE_ME` and hostnames with `*.example.com`. This is the file to
  copy when provisioning a fresh host.
- **`chatbot-api/.env.example`** — **updated**. Added the three graph
  variables that were missing (`GRAPH_CONTEXT_FORMAT`, `GRAPH_BOOST_ALPHA`,
  `GRAPH_BOOST_POOL_K`) and a comment documenting the
  `GRAPH_EXPANSION_ENABLED` master kill-switch.

`.gitignore` already ignored `.env` (and every `.env.*`) with an explicit
`!.env.example` exception, so the examples are tracked while the real
values are not.

---

## 3. `.env` additions

All Neo4j-related variables (connection, tuning, kill-switch) live in
`.env` (git-ignored) and are forwarded to the container via
`docker-compose.yml`'s `env_file: .env`. Real production values:

```env
# Master kill-switch
GRAPH_EXPANSION_ENABLED = True

# Connection
NEO4J_URI = bolt://nimani.diplomacy.edu:7687
NEO4J_USER = neo4j
NEO4J_PASS = <redacted>
NEO4J_DATABASE_DIPLO = weaviatediplo
NEO4J_DATABASE_DW    = weaviatedw

# Tuning
GRAPH_EXPANSION_TIMEOUT       = 5.0
GRAPH_EXPANSION_MAX_RELATIONS = 30
GRAPH_EXPANSION_MAX_DEPTH     = 1
GRAPH_CONTEXT_FORMAT          = boost_inline
GRAPH_BOOST_ALPHA             = 0.15
GRAPH_BOOST_POOL_K            = 16
```

See the [Neo4j Graph Expansion section in `.env.example`](../.env.example)
for descriptions of each variable and the allowed values for
`GRAPH_CONTEXT_FORMAT`.

---

## 4. Kill-switch semantics

`GRAPH_EXPANSION_ENABLED=False` disables the entire subsystem without code
changes. Verified in production:

| Surface                              | `True` (current)                                          | `False`                        |
|--------------------------------------|-----------------------------------------------------------|--------------------------------|
| Startup log                          | `[STARTUP] Neo4j graph expansion: ENABLED (...)`          | `... DISABLED`                 |
| Neo4j driver / connection pool       | opened at startup                                         | never created                  |
| `GET /api/debug/neo4j-health`        | `{"status":"ok"}`                                         | `{"status":"disabled"}`        |
| `POST /api/debug/retrieve`           | `counts.graph_boost` and `counts.graph_context` reflect real expansion sizes | both keys present but `0`; debug payload shape unchanged |
| `POST /api/chat/{id}` source entries | `text/title/date/url` only (no `graph_*` on cards)          | same card shape; graph off when disabled |
| WebSocket subgraph payload           | cited-subgraph + hop3 entities pushed                     | no `status:"graph_data"` frame |
| Retrieval ranking                    | graph boost applied when `GRAPH_CONTEXT_FORMAT=boost_inline` | identical to pre-Neo4j prod |

Implementation entry points that gate on the flag:
- `app/core/singleton.py::init_graph_expansion()` — short-circuits startup.
- `app/core/singleton.py::get_graph_expansion_service()` — returns `None`
  when the service was never initialised.
- `app/ai/ai_services/graph_expansion_node.py` — reads the flag via
  `get_param('enable_graph_expansion', GRAPH_EXPANSION_ENABLED)` so
  per-request overrides still work.
- `app/api/routes/debug_route.py::debug_retrieve` — skips the
  expand/boost/enrich block when `service is None`.
- `app/api/routes/debug_route.py::neo4j_health` — returns
  `"disabled"` without touching the driver.
- `app/services/chat_service.py` — skips the `cited_subgraph` /
  `hop3` block when `get_graph_expansion_service() is None`.

---

## 5. Operations

### 5.1 Deploy the current change

```bash
cd /opt/dev/humainism_ai_chatbot_api
docker compose build api            # rebuilds the image (poetry installs neo4j + langgraph)
docker compose up -d --force-recreate api
```

### 5.2 Toggle the kill-switch at runtime

```bash
cd /opt/dev/humainism_ai_chatbot_api
# edit .env: flip GRAPH_EXPANSION_ENABLED=True/False
docker compose up -d --force-recreate api   # recreate picks up env_file
```

No image rebuild is needed when only `.env` changes.

### 5.3 Smoke test

```bash
# Neo4j health (via nginx)
curl -s https://chat-api.humainism.ai/api/debug/neo4j-health
# → {"status":"ok"} when enabled, {"status":"disabled"} when off

# Retrieval with graph counts
curl -s -X POST https://chat-api.humainism.ai/api/debug/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"What is digital diplomacy?","user_type":"general",
       "retrieval_config":{"retrieval_chunks":5}}' \
  | python3 -c "import json,sys; c=json.load(sys.stdin)['counts']; \
                print('graph_boost=',c.get('graph_boost'),'graph_context=',c.get('graph_context'))"

# End-to-end chat (sources have no graph_* keys; graph_data is separate on WS)
CID=$(curl -s -X POST https://chat-api.humainism.ai/api/conversation/get_id \
  -H "Content-Type: application/json" -d '{"conversationId": null}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['conversationId'])")
curl -s -X POST "https://chat-api.humainism.ai/api/chat/$CID" \
  -H "Content-Type: application/json" \
  -d '{"userIp":"127.0.0.1","message":"What is digital diplomacy?","userType":"general"}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); \
                print('graph_data nodes=', len((d.get('graph_data') or {}).get('nodes') or [])); \
                print('source keys=', list((d.get('sources') or [{}])[0].keys()))"
```

### 5.4 Rollback (remove Neo4j entirely)

Full rollback path — returns the repo to the state before the cherry-pick:

```bash
cd /opt/dev/humainism_ai_chatbot_api
git checkout backup-before-neo4j-cherrypick -- .
docker compose build api
docker compose up -d --force-recreate api
```

Soft rollback — keep the code but run with graph expansion off:

```bash
# edit .env: GRAPH_EXPANSION_ENABLED = False
docker compose up -d --force-recreate api
```

---

## 6. Post-deployment checklist (completed 2026-04-21)

- [x] Neo4j TCP reachability from `humainism_api` network (`nimani.diplomacy.edu:7687`)
- [x] `neo4j` / `langgraph` present in the built image (`poetry.lock`
      already included them after cherry-pick)
- [x] Container `healthy` after recreate
- [x] `/api/debug/neo4j-health` returns `ok` via public URL
- [x] `/api/debug/retrieve` returns non-zero `graph_boost`/`graph_context`
      counts
- [x] `/api/chat/{id}` returns `graph_data` when expansion is on; source cards omit `graph_*`
- [x] Credentials removed from `docker-compose.yml`
- [x] Root `.env.example` created
- [x] `chatbot-api/.env.example` updated with missing variables
- [x] Kill-switch verified in both states (`True` and `False`) against the
      public URL

---

## 7. Open follow-ups

- The `anja` upstream still has the ~1.7k-line working-tree polish
  uncommitted. Once that lands on `origin/master`, prod can drop the
  hand-applied patch and pull cleanly.
- If the root-level `docker-compose.yml` changes (credentials removed,
  comment added) get committed, make sure the same sanitization is
  replicated in `docker-compose.test.yml` before `anja` pushes.
- `chat-api.humainism.ai` nginx config (`/etc/nginx/sites-available/chat-api.humainism.ai.conf`)
  requires no changes; the upstream stays `127.0.0.1:8561` and the
  added `/api/debug/neo4j-health` route is picked up automatically.

---

## 8. Security cleanup — benchmark scripts & notebook (2026-04-21)

GitGuardian flagged plaintext credentials in scripts and a notebook that
originated on `anja` and landed on prod via the cherry-pick described in
§2.1. A follow-up commit moved all secrets to `.env` without rewriting
git history.

### 8.1 Files sanitised

| Repo / path                                                  | What was hardcoded                                                                                                                | Now reads from `.env`                                                                                                         |
|--------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `chatbot-api/scripts/benchmark_graph_rag.py`                 | `NEO4J_PASS`, `WV_KEY`, full Mongo URI (with password), `LLM_KEY`, `EMBEDDING_KEY`, `RERANKER_API_KEY`, `DB_PWD`, `DB_CONNECTION` | `NEO4J_PASS`, `WV_KEY`, `DB_CONNECTION`, `LOCAL_LLM_KEY`, `LOCAL_EMBEDDING_KEY`, `RERANKER_API_KEY`, …                         |
| `chatbot-api/scripts/test_format_abc.py`                     | same set as above (subset)                                                                                                        | same as above                                                                                                                 |
| `chatbot-api/notebooks/graph_rag_explorer.ipynb` *(anja only; not yet on prod)* | cells 2 / 27 / 28 hardcoded `NEO4J_PASS`, `WV_KEY`, `LLM_KEY`, `MONGO_URI`                                     | same `Config(env_file)` pattern, `LOCAL_LLM_*`, `DB_CONNECTION`                                                               |
| `chatbot-neo4j/app/core/neo4j_config.py` *(anja only)*       | `NEO4J_URI` default pointed to internal host                                                                                       | default changed to `bolt://localhost:7687`; real URI comes from `NEO4J_URI` in `.env`                                         |

The pattern matches the one already in use by
`scripts/benchmark_weaviate_collections.py`,
`scripts/export_random_chunk.py`, and
`scripts/ingest_topics.py` (ranije i `populate_diplo_heading.py`, uklonjen):

```python
env_file = os.getenv("ENV_FILE", "../.env")
_config = Config(env_file)
NEO4J_PASS = _config("NEO4J_PASS", cast=str)
WEAVIATE_API_KEY = _config("WV_KEY", cast=str)
# ... etc
```

One new optional env var — `BENCHMARK_MONGO_DB` (defaults to
`chatbot_humainism_ai`) — was added for the benchmark scripts so the
target DB can be overridden without editing code.

### 8.2 What this commit fixes vs what it does NOT fix

| Question                                                             | Status                                                                                                                                                          |
|----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Plaintext secrets in tracked files on `HEAD`                         | **FIXED** — scripts and notebook now read everything from `.env`                                                                                                |
| Plaintext secrets in *git history* on GitHub                         | **NOT fixed** — the tainted commits (`ecb11a9` on `anja/master`, `b84ec1b`/`a0ec5c8` on `origin/master`, plus old `9d6e059`/`b82f079` on `origin/main`) remain  |
| GitGuardian alerts for this repo                                     | Will **NOT** stop until the leaked credentials are rotated (or history is rewritten + force-pushed with rotation)                                                |

History rewrite (`git filter-repo` or BFG) + force-push was explicitly
deferred because it breaks every collaborator clone and every existing
PR reference. The realistic path forward is **credential rotation**, not
history rewrite — the secrets have been publicly cloneable ever since
they first landed on GitHub, so rotating them is mandatory regardless
of what we do with git history.

### 8.3 Required credential rotation (operations follow-up)

All of these have been exposed on GitHub and must be rotated on the
source systems, then updated in every `.env` that uses them:

| Secret                                                                 | System                                      | Where it is consumed                                                                                   |
|------------------------------------------------------------------------|---------------------------------------------|--------------------------------------------------------------------------------------------------------|
| `NEO4J_PASS`                                                           | Neo4j KG server                             | `.env` of `humainism_ai_chatbot_api` (prod + test), any KG pipeline, any ops tooling that connects     |
| `WV_KEY`                                                               | Weaviate vector DB                          | `.env` of `humainism_ai_chatbot_api`, any ingest/reindex scripts, `nested_chunking_api-dev`, etc.       |
| `LOCAL_LLM_KEY` / `LOCAL_EMBEDDING_KEY` / `RERANKER_API_KEY` (shared)  | TEI / vLLM gateway (embedder, reranker, gpt-oss) | same `.env` + every other service that calls those endpoints                                      |
| Mongo password inside `DB_CONNECTION`                                  | MongoDB (admin user)                        | same `.env`, all diplo apps that connect to this Mongo                                                 |
| `OPENAI_KEY`                                                           | OpenAI service account                      | `.env` of prod + any other app using the same key                                                      |

> The concrete old values are intentionally not written here — they are
> recorded in the internal GitGuardian/ops incident ticket and must be
> replaced everywhere they are currently stored (live `.env` files,
> vault, CI secrets, etc.).

After rotation, update the `.env` in both `/opt/dev/humainism_ai_chatbot_api/`
and `/opt/dev/humainism_ai_chatbot_api anja/` and recreate the
containers:

```bash
cd /opt/dev/humainism_ai_chatbot_api && docker compose up -d --force-recreate api
cd "/opt/dev/humainism_ai_chatbot_api anja" && docker compose -f docker-compose.test.yml up -d --force-recreate api-test
```

### 8.4 Optional future hardening

- Add a `pre-commit` hook running `gitleaks`/`detect-secrets` so similar
  leaks are caught locally before they reach GitHub.
- If/when the team is ready to accept the disruption, rewrite history
  with `git filter-repo --replace-text <patterns.txt>` and force-push
  all branches + tags (requires coordinated re-clone by every
  developer).
- Rotate secrets on a schedule (90 days), store them in a proper secret
  manager (Vault / 1Password / AWS SM) instead of `.env` on disk.
