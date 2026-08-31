# API Reference

Base URL: `http://{HOST}:{PORT}/api`

## Endpoints

### Create/Validate Conversation

```
POST /api/conversation/get_id
```

Creates a new conversation ID or validates an existing one.

**Request:**
```json
{
  "conversation_id": "optional-existing-id"
}
```

- Send `null` or omit `conversation_id` to create a new conversation
- Send an existing ID to validate it; if invalid, a new one is returned

**Response:**
```json
{
  "conversationId": "abc123-uuid"
}
```

---

### Chat

```
POST /api/chat/{conversation_id}
```

Send a message and receive a complete response with sources.

**Path Parameters:**
- `conversation_id` -- Valid conversation ID from `/conversation/get_id`

**Request:**
```json
{
  "user_ip": "192.168.1.1",
  "message": "What is digital diplomacy?",
  "user_type": "general",
  "retrieval_config": {
    "use_dynamic_label_weights": true,
    "dynamic_weight_alpha": 2.2,
    "dynamic_weight_min_sim": 0.3
  }
}
```

**`user_type` options:** `general`, `student`, `diplomat`, `researcher`, `historian`, `philosopher`, `contrarian`, `journalist`

**`retrieval_config`** (optional): Override any retrieval parameter for this request. All fields are optional; omitted fields use env defaults.

**Content filters** (Weaviate-level; combined with AND; always includes `visibility != private`):

| Field | Type | Description |
|-------|------|-------------|
| `site_filter_name` | string | Only chunks from this site (e.g. `"diplomacy.edu"`) |
| `parent_filter_name` | string | Only chunks under a parent document (resolved via `DiploDocument.name` → `parent_document_hash`) |
| `person_filter_name` | string | Only chunks whose `person` array contains this name (exact match) |
| `person_filter_names` | string[] | Same as above, but match **any** of the listed names (OR) |
| `city_filter_name` | string | Only chunks where `city` equals this value (exact match) |
| `city_filter_names` | string[] | Match **any** of the listed cities (OR) |
| `country_filter_name` | string | Only chunks where `country` equals this value (exact match) |
| `country_filter_names` | string[] | Match **any** of the listed countries (OR) |
| `organisation_filter_name` | string | Only chunks where `organisation` equals this value (exact match) |
| `organisation_filter_names` | string[] | Match **any** of the listed organisations (OR) |
| `post_types` | string[] | Whitelist WordPress `post_type` values (e.g. `["blog", "topic"]`) |

The `person` property on `DiploChunk_contextual` / `DiploParagraph_contextual` is a **text array** of associated people (authors, speakers, etc.). Filtering uses Weaviate `contains_any`: a chunk with `person: ["Jovan Kurbalija", "Aleksandar Stankovic"]` matches `person_filter_name: "Jovan Kurbalija"` or `person_filter_names: ["Aleksandar Stankovic", "Other Author"]` when any listed name appears in the array.

`city`, `country`, and `organisation` are **TEXT** fields (scalar, not arrays). Filtering also uses `contains_any`: the property value must equal one of the listed values. These fields are populated mainly on `post_type=people` profiles and propagated to chunks at ingest.

If both `*_filter_name` and `*_filter_names` are set for the same field (person, city, country, or organisation), values are merged (deduplicated) before filtering. Multiple **different** filter fields are combined with **AND** (e.g. city + country + organisation must all match).

Example with person filter (single name):

```json
{
  "user_ip": "192.168.1.1",
  "message": "What has he written about AI?",
  "user_type": "general",
  "retrieval_config": {
    "person_filter_name": "Jovan Kurbalija",
    "site_filter_name": "diplomacy.edu"
  }
}
```

Example with multiple authors (OR — chunks matching any listed person):

```json
{
  "retrieval_config": {
    "person_filter_names": ["Jovan Kurbalija", "Aleksandar Stankovic"]
  }
}
```

Example with city / country / organisation filters (multi-select, OR within each field; AND across fields):

```json
{
  "retrieval_config": {
    "city_filter_names": ["Geneva", "Zurich"],
    "country_filter_names": ["Switzerland"],
    "organisation_filter_name": "DiploFoundation"
  }
}
```

**Other `retrieval_config` fields** (tuning; see `RetrievalConfig` in `chat_schema.py`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_dynamic_label_weights` | bool | `true` | Enable/disable dynamic label weight boosting |
| `dynamic_weight_alpha` | float | `2.2` | Boost scale factor (similarity * alpha) |
| `dynamic_weight_min_sim` | float | `0.3` | Minimum similarity threshold for boost |

**Response:**
```json
{
  "message_id": "60f7b2...",
  "answer": "Digital diplomacy refers to... [1] ... [2]",
  "sources": [
    {
      "text": "Source content excerpt...",
      "title": "Digital Diplomacy Overview",
      "date": "2025-03-15",
      "url": "https://example.com/topics/digital-diplomacy/#section",
      "deep_link_url": "https://example.com/topics/digital-diplomacy/?diplo-deep-link-text=..."
    }
  ],
  "related_questions": [
    "How has digital diplomacy evolved?",
    "What tools are used in digital diplomacy?"
  ],
  "graph_data": {
    "nodes": [{ "id": "doc_abc", "label": "Digital Diplomacy", "type": "Source" }],
    "edges": [{ "from": "doc_abc", "to": "t_xyz", "label": "about" }],
    "subgraph_url": "/subgraphs/abc123_def456.html"
  }
}
```

**Source cards** (`sources[]`) expose only citation display fields: `text`, `title`, `date`, `url`, and optionally `link`, `name`, `deep_link_url`. Neo4j enrichment (`graph_topics`, `graph_people`, etc.) is **not** included on source objects — it is used internally for retrieval/LLM context and for the separate **`graph_data`** payload (Knowledge Graph widget). When `GRAPH_EXPANSION_ENABLED=false`, `graph_data` is omitted or null.

Graph metadata for debugging is still available on `POST /api/debug/retrieve` per-document (`_graph_topics`, `_graph_people`, …).

**Error Responses:**
- `404` -- Conversation ID not found
- `500` -- Internal server error

---

### Chat (WebSocket Streaming)

```
WS /api/chat/ws/{conversation_id}
```

Real-time streaming of the chat response via WebSocket.

**Connection Flow:**
1. Client connects to WebSocket
2. Server sends: `{"status": "info_message", "text": "Welcome! Connection established."}`
3. Client sends JSON message (same schema as POST chat request)
4. Server sends: `{"status": "info_message", "text": "Please wait..."}`
5. Server streams answer chunks: `{"status": "answer", "text": "..."}` (with live citation renumbering)
6. Optional per-citation URLs: `{"status": "citation_url", "num": 1, "url": "..."}`
7. Server sends sources: `{"status": "sources", "text": [...]}` — same shape as HTTP `sources` (no `graph_*` keys)
8. When graph expansion is enabled: `{"status": "graph_data", "text": {"nodes": [...], "edges": [...], "subgraph_url": "..."}}`
9. Server sends related questions: `{"status": "related_questions", "text": [...]}`
10. Optional: `renumber_map`, `citation_urls`, then `{"status": "message_id", "text": "..."}`
11. Connection may stay open for the next message

**Error message:**
```json
{"status": "error", "text": "Wrong conversation ID. Please try again"}
```

---

### Feedback

```
PUT /api/chat/feedback/{message_id}
```

Submit user feedback for a response.

**Path Parameters:**
- `message_id` -- The `message_id` returned from the chat endpoint

**Request:**
```json
{
  "feedback_type": 1,
  "feedback_message": "Very helpful answer!"
}
```

**`feedback_type` values:**
- `1` -- Positive (thumbs up)
- `-1` -- Negative (thumbs down)
- `0` -- Neutral / reset

**Response:**
```json
{"message": "Feedback updated successfully"}
```

**Error Responses:**
- `404` -- Message or response not found
- `500` -- Internal server error

---

## Ingestion Endpoints

### Ingest a Single Topic

```
POST /api/ingest/topic
```

Calls the chunking API to ingest a single topic from WordPress into contextual Weaviate collections. If the topic already exists, old data is replaced.

**Request:**
```json
{
  "topic_slug": "digital-diplomacy",
  "site": "diplomacy.edu"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `topic_slug` | string | *required* | Topic slug, e.g. `digital-diplomacy` |
| `site` | string | `diplomacy.edu` | Site domain |

**Response:**
```json
{
  "success": true,
  "message": "Ingested 'digital-diplomacy': 45 paragraphs, 130 chunks",
  "data": {
    "paragraphs_created": 45,
    "chunks_created": 130
  }
}
```

**Error Responses:**
- `502` -- Chunking API error
- `500` -- Internal server error

---

### Reingest ALL Topics

```
POST /api/ingest/topics/all
```

Full reingest: deletes all existing topic data from Weaviate, then calls the chunking API to bulk-ingest all topics from WordPress into contextual collections. This is a long-running operation (typically 5-15 minutes).

**Request:**
```json
{
  "site": "diplomacy.edu",
  "skip_existing": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `site` | string | `diplomacy.edu` | Site domain |
| `skip_existing` | bool | `false` | Skip topics already in Weaviate (incremental mode) |

**Response:**
```json
{
  "success": true,
  "message": "Ingested 85 topics: 3200 paragraphs, 9500 chunks (482s)",
  "data": {
    "topics_ingested": 85,
    "paragraphs_created": 3200,
    "chunks_created": 9500,
    "total_elapsed_s": 482
  }
}
```

> **Removed endpoint:** `POST /api/ingest/headings/refresh` — heading hierarchy is encoded in contextual embeddings; `DiploHeading` is no longer used.

---

### List Unique Persons (Weaviate catalog)

```
GET /api/person
```

Returns a sorted list of unique **author** names from the `person` TEXT[] field on `DiploDocument_contextual` only (~47k documents; authors are propagated to chunks at ingest). Non-private documents only (`visibility != "private"`).

Intended for the **Diplo Chatbot** WordPress plugin (`humainism-chatbot`): sync into table `{prefix}diplo_chatbot_persons` via **Person Filter Name** on each chatbot (Sync now + daily cron). Values must match `person_filter_name` / entries in `person_filter_names` exactly (case-sensitive).

**Access control:** Same as other API routes — `ALLOWED_IPS` (server-side calls from the WordPress host) and `ALLOWED_HOSTS` for CORS if called from the browser.

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `refresh` | bool | `false` | If `true`, bypass in-memory cache and rescan Weaviate |

**Response:**
```json
{
  "persons": ["Aleksandar Stankovic", "Jovan Kurbalija"],
  "count": 2,
  "sources": {
    "DiploDocument_contextual": 46754
  },
  "cached": false,
  "generated_at": "2026-06-04T12:00:00Z"
}
```

- `persons`: case-sensitive unique names (must match `person_filter_name` or `person_filter_names` in chat)
- `sources`: number of objects scanned per collection (diagnostics)
- `cached`: `true` when served from in-memory cache (`PERSON_LIST_CACHE_TTL_SECONDS`, default 3600)

**Error responses:**
- `403` — client IP not in `ALLOWED_IPS`
- `503` — Weaviate scan failed

---

### List Unique Cities, Countries, and Organisations (Weaviate catalog)

```
GET /api/metadata-filters
```

Returns sorted lists of unique **city**, **country**, and **organisation** values from `DiploDocument_contextual` only (non-private documents). Values are propagated to chunks/paragraphs at ingest.

Intended for the **Diplo Chatbot** WordPress plugin: populate multi-select dropdowns for `city_filter_names`, `country_filter_names`, and `organisation_filter_names` in `retrieval_config`. Values must match exactly (case-sensitive; British spelling `organisation`).

**Access control:** Same as other API routes — `ALLOWED_IPS` and `ALLOWED_HOSTS`.

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `refresh` | bool | `false` | If `true`, bypass in-memory cache and rescan Weaviate |

**Response:**
```json
{
  "cities": ["Geneva", "Zurich"],
  "countries": ["Malta", "Switzerland"],
  "organisations": ["DiploFoundation", "Geneva Academy of International Humanitarian Law and Human Rights"],
  "counts": {
    "city": 179,
    "country": 181,
    "organisation": 182
  },
  "sources": {
    "DiploDocument_contextual": 51595
  },
  "cached": false,
  "generated_at": "2026-07-02T12:00:00Z"
}
```

- `cities` / `countries` / `organisations`: case-sensitive unique values for use in `*_filter_name` / `*_filter_names`
- `counts`: unique value counts per field (diagnostics)
- `cached`: `true` when served from in-memory cache (`PERSON_LIST_CACHE_TTL_SECONDS`, default 3600)

**Error responses:**
- `403` — client IP not in `ALLOWED_IPS`
- `503` — Weaviate scan failed
