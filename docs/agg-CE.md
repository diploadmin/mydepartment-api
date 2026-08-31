# Plan: async za Agg i CE (Sentence-First)

**Izvor istine:** [NewworldProg/chatobt-api-dev](https://github.com/NewworldProg/chatobt-api-dev) branch `dev-luka`  
(ne lokalni dirty working tree — tamo ima necommitovanih izmena koje nisu na GitHubu).

**Pitanje:** da li mogu **agg** i **CE** da budu async / paralelni?

**Kratak odgovor (iz remote koda):**  
- **Agg ∥ sentence-CE — ne** (CE bira grupe po `aggregate_score`).  
- **Sentence-CE ∥ document-rerank — ne** (doc rerank koristi `_best_sentence` u payload-u).  
- **Async I/O da ne blokira event loop — da** (oba TEI `requests.post` + Weaviate sync).  
- **Agg async radi brzine — ne vredi** (`docs/RETRIEVAL_PIPELINE.md`: agg **&lt;5ms**).

---

## Šta je na GitHubu (`dev-luka`)

| Fajl | Uloga |
|------|--------|
| `chatbot-api/.../retrievers/sentence_first.py` | hybrid → group → **agg** → **sentence CE** → docs |
| `chatbot-api/.../reranker.py` | `rerank_sentences` + `rerank` — oba sync `requests` |
| `chatbot-api/.../graph.py` | `async def direct_retrieval_node` ali **`retriever.invoke` + `tei_reranker.rerank` sync** |
| `docs/RETRIEVAL_PIPELINE.md` | timing: embed ~100ms, hybrid ~1.5–2.4s, agg &lt;5ms, **doc rerank ~1.1–1.5s** |
| `docs/SENTENCE_RETRIEVER_OPTIMIZATIONS.md` | istorija optimizacija (top-N, blocklist, …) |

Config (`config.py` na remote): `SENTENCE_CE_*`, `SENTENCE_HIGHLIGHT_MODE`, `USE_RERANKER`, `SENTENCE_RERANKER_TOP_K`, `TOP_N_PER_SECTION`.

---

## Dva različita “CE” sloja (važno)

Na remote pipeline-u nisu isto:

```text
SentenceFirstRetriever
  Step 3   AGG          — CPU, lokalno
  Step 3b  sentence CE  — TEI rerank_sentences → _best_sentence  (highlight)
       │
       ▼
graph.direct_retrieval_node
  doc CE / rerank       — TEI rerank(docs) → _reranker_score
       │                  payload uključuje [BEST MATCH]: _best_sentence
       ▼
  label weights
```

| Sloj | Funkcija | Zavisi od |
|------|----------|-----------|
| **Agg** | rang sekcija | hybrid skorova |
| **Sentence CE** | najbolja rečenica po grupi | **agg top-N** (`SENTENCE_CE_MAX_GROUPS`) |
| **Doc rerank** | redosled sekcija za LLM | docs + **`_best_sentence`** |

Zato “async agg i CE” u praksi mora da kaže **koji** CE.

---

## Zašto agg ∥ sentence-CE ne ide

Iz `sentence_first.py` na remote:

1. Agg piše `aggregate_score`
2. `_select_best_sentence_single` sortira grupe po tom skoru
3. Tek onda TEI batch (`SENTENCE_CE_CANDIDATES` × top groups)

Bez agg nema smisla koje grupe ići u CE.  
Mode `SENTENCE_HIGHLIGHT_MODE=multi` → sentence CE **skip** (samo hybrid best).

---

## Šta *može* async (preporuka za ovaj repo)

### A) Ne blokirati asyncio loop (MVP, najveći ops win)

`direct_retrieval_node` je `async` ali zove sync I/O. Pod loadom to smrzava ostale WS.

```python
docs = await asyncio.to_thread(active_retriever.invoke, user_question)
if tei_reranker and _use_reranker:
    docs = await asyncio.to_thread(tei_reranker.rerank, user_question, docs, _reranker_top_k)
```

Ne skraćuje wall-clock jednog requesta; štiti konkurentnost.

### B) Async TEI (`httpx.AsyncClient`)

U `reranker.py`: `arerank_sentences` + `arerank`.  
Sentence CE i doc rerank i dalje **sekvencijalni** međusobno, ali bez `to_thread` overhead-a.

### C) Async Weaviate / embed (veći refactor)

Step 1 (hybrid + embed) je dominantan pored doc reranka — vredi tek posle merenja.

### D) Overlap koji *nije* OK bez redesign-a

| Ideja | Verdict |
|-------|---------|
| Agg ∥ sentence CE | **Ne** |
| Sentence CE ∥ doc rerank | **Ne** (doc format koristi `_best_sentence`) |
| Doc rerank bez BEST MATCH polja pa CE kasnije | Quality change — samo A/B |

---

## Timing (remote docs) — gde ima smisla async

| Korak | Tipično | Async win? |
|-------|---------|------------|
| Embed | ~100ms | da (I/O) |
| Hybrid | ~1.5–2.4s | da (I/O) |
| **Agg** | **&lt;5ms** | **ne** |
| Sentence CE | (deo retrieve; batch TEI) | da (I/O), posle agg |
| Doc rerank | ~1.1–1.5s | da (I/O), posle sentence CE |
| Label weights | &lt;1ms | ne |

---

## Faze

### Faza 0 — Meranja na prod/dev

U `TIMING` / debug dodati: `agg_ms`, `sentence_ce_ms`, `doc_rerank_ms` (pored postojećeg `retrieval=` / `reranker=`).

### Faza 1 — MVP

- [ ] `asyncio.to_thread` oko `invoke` + `rerank` u `direct_retrieval_node`  
Acceptance: paralelni chatovi ne čekaju jedan na drugog zbog sync TEI.

### Faza 2

- [ ] `httpx` async u `TEIReranker`  
- [ ] opciono `_aget_relevant_documents` u `SentenceFirstRetriever`

### Faza 3 — samo uz A/B

Eksperimenti sa ranijim / jeftinijim highlightom (bez sentence CE, ili CE samo top-5) — **ne** kao “async agg∥CE”.

---

## Odluka

| Pitanje | Odgovor sa GitHub koda |
|---------|------------------------|
| Async agg+CE jedan pored drugog? | **Ne** |
| Async da retrieval ne blokira FastAPI? | **Da** — Faza 1 |
| Gde je latencija? | Hybrid + **doc** rerank; agg zanemarljiv |
| Lokal vs remote | Plan i implementacija gledati **`chatobt/dev-luka`**, ne uncommitted lokal |

**Preporuka:** Faza 0 → Faza 1 `to_thread` u `graph.py`; ne paralelizovati agg i CE.
