# Diplo Chatbot API — Arhitektura sistema

> Detaljan pregled kako radi chatbot servis, od korisničkog upita do generisanog odgovora.
> Pripremljeno kao osnova za integraciju Neo4j graph expansion koraka.

---

## Sadržaj

1. [Pregled sistema](#1-pregled-sistema)
2. [Infrastruktura i servisi](#2-infrastruktura-i-servisi)
3. [Weaviate kolekcije i šema podataka](#3-weaviate-kolekcije-i-šema-podataka)
4. [Kompletni tok korisničkog upita](#4-kompletni-tok-korisničkog-upita)
5. [Retrieval pipeline — detaljan pregled](#5-retrieval-pipeline--detaljan-pregled)
6. [Reranking i label weights](#6-reranking-i-label-weights)
7. [LLM prompt konstrukcija](#7-llm-prompt-konstrukcija)
8. [Generisanje odgovora i sources](#8-generisanje-odgovora-i-sources)
9. [Ključni fajlovi i gde šta živi](#9-ključni-fajlovi-i-gde-šta-živi)
10. [document_hash i parent_document_hash](#10-document_hash-i-parent_document_hash)
11. [Tačka integracije za Neo4j Graph Expansion](#11-tačka-integracije-za-neo4j-graph-expansion)

---

## 1. Pregled sistema

Chatbot API je **FastAPI** aplikacija koja implementira **RAG (Retrieval-Augmented Generation)** pipeline koristeći:

- **Weaviate** — vektorska baza za skladištenje i pretragu čankova (sentence/paragraph level)
- **LangGraph** — orkestrator koji povezuje retrieval → reranking → LLM generisanje
- **LLM** (OpenAI / Anthropic / lokalni) — za generisanje krajnjeg odgovora
- **TEI** (Text Embeddings Inference) — lokalni embedding + cross-encoder reranker servisi
- **Redis** — keširanje retrieval rezultata + skraćivanje deep link URL-ova
- **MongoDB** — perzistencija chat istorije i feedbacka

```
┌─────────────┐     ┌──────────────────────────────────────────────────────────────┐
│   Korisnik   │────▶│  FastAPI (POST /api/chat/{id} ili WS /api/chat/ws/{id})     │
└─────────────┘     └────────────────────────┬─────────────────────────────────────┘
                                             │
                                             ▼
                    ┌────────────────────────────────────────────────────────────────┐
                    │                    LangGraph Pipeline                          │
                    │                                                                │
                    │  START ──▶ direct_retrieval_node ──▶ generate_final_node ──▶ END│
                    │              │                           │                      │
                    │              ▼                           ▼                      │
                    │     ┌─────────────────┐        ┌─────────────────┐             │
                    │     │  Weaviate Search │        │  LLM (bez tools)│             │
                    │     │  + TEI Reranker  │        │  + citiranje    │             │
                    │     │  + Label Weights │        │  [1], [2]...    │             │
                    │     └─────────────────┘        └─────────────────┘             │
                    └────────────────────────────────────────────────────────────────┘
```

---

## 2. Infrastruktura i servisi

| Servis | Port | Uloga |
|--------|------|-------|
| **Chatbot API** (ovaj repo) | 8561 | FastAPI — chat, ingest, debug |
| **Weaviate** | 8080 (HTTP), 50051 (gRPC) | Vektorska baza podataka |
| **Redis** | 6379 | Retrieval cache + deep link shortener |
| **MongoDB** | 27017 | Chat istorija, feedback |
| **Chunking API** (eksterni) | 8563 | WordPress scraping → chunking → Weaviate write |
| **TEI Embedding** | konfig. | Text Embeddings Inference server |
| **TEI Reranker** | konfig. | Cross-encoder reranking server |

### Docker Compose

```yaml
# docker-compose.yml
services:
  api:
    build: ./chatbot-api
    ports: ["8561:8561"]
    env: REDIS_HOST=redis, WV_CLIENT_URL=...
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

### Ključne env varijable (.env)

| Varijabla | Default | Opis |
|-----------|---------|------|
| `RETRIEVAL_MODE` | `sentence` | Režim pretrage: `sentence`, `paragraph`, `combined`, `twophase` |
| `RETRIEVAL_CHUNKS` | `8` | Broj chunk-ova koji se šalju LLM-u |
| `HYBRID_ALPHA` | `0.85` | Balans vektor (1.0) vs BM25 (0.0) |
| `CITE_SOURCES` | `True` | LLM citira izvore kao [1], [2]... |
| `SKIP_TOOL_DECISION` | `False` | `True` = preskoči LLM tool decision, direktno u retrieval |
| `USE_RERANKER` | `False` | Koristi TEI cross-encoder reranking |
| `SENTENCE_INDEX_NAME` | `DiploChunk` | Weaviate kolekcija za sentence search |
| `INDEX_NAME` | *(iz .env)* | Weaviate kolekcija za paragraph search (DiploParagraph) |
| `USE_CONTEXTUAL_COLLECTIONS` | `False` | Dodaje `_contextual` suffix na kolekcije |
| `USE_DYNAMIC_LABEL_WEIGHTS` | `True` | Dinamičko prilagođavanje label weights-a prema query intent-u |
| `RETRIEVAL_CACHE_TTL` | `120` | Redis cache TTL u sekundama (0 = isključen) |
| `LLM_MODEL_PROVIDER` | `openai` | Provider: `openai`, `anthropic`, `local`, `deepseek` |
| `WEBSITE_NAME` | *(iz .env)* | `diplomacy.edu` ili `humainism.ai` — diktira system prompt |

---

## 3. Weaviate kolekcije i šema podataka

### 3.1 DiploDocument — registar dokumenata

**Ne koristi se za pretragu** — služi kao lookup tabela za razrešavanje `document_hash` po imenu dokumenta.

```
DiploDocument {
  document_hash:        string   // MD5 hash — JEDINSTVEN IDENTIFIKATOR dokumenta
  name:                 string   // Naslov dokumenta (npr. "AI and Diplomacy")
  link:                 string   // URL dokumenta
  site:                 string   // "diplomacy.edu" ili "dig.watch"
  post_type:            string   // "blog", "event", "topic", "resource", "course"...
  content_type:         string   // "html", "pdf"
  wp_post_id:           string   // WordPress post ID
  date:                 datetime
  slug:                 string
  full_text:            string   // Kompletan tekst dokumenta
  pdf_filename:         string   // Ime PDF fajla (ako je PDF)
  parent_document_hash: string   // Hash parent dokumenta
  parent_wp_post_id:    string
  parent_post_type:     string
  language:             string
  summary:              string
  ingested_at:          datetime
}
```

Primer iz baze:
```json
{
  "document_hash": "c58ee7de1137f2454c33f4d9a79c4e22",
  "name": "Small Developing Economies in the World Trade Organization",
  "site": "diplomacy.edu",
  "post_type": "resource",
  "content_type": "pdf",
  "link": "https://caricom.org/documents/10160-small_developing_economies_in_the_wto.pdf"
}
```

### 3.2 DiploParagraph — paragraf-nivo čankovi

```
DiploParagraph {
  text:                    string   // Tekst paragrafa
  link:                    string   // URL stranice (FIELD tokenizacija za exact match)
  name:                    string   // Naslov stranice (h1)
  h1:                      string   // Glavni naslov
  h2:                      string   // Podnaslov
  h3:                      string
  section:                 string   // Sekcija unutar stranice
  last_h_title:            string   // Poslednji heading iznad paragrafa
  post_type:               string   // "blog", "event", "topic"...
  date:                    datetime
  visibility:              string   // null ili "private"
  site:                    string
  parent_document_hash:    string   // ← VEZA KA DiploDocument.document_hash
  parent_paragraph_id:     string
  paragraph_index:         int
}
```

### 3.3 DiploChunk — sentence-nivo čankovi

```
DiploChunk {
  sentence:                string   // Tekst jedne rečenice
  context:                 string   // Kontekst sekcije kojoj pripada
  chunk_level:             string   // "sentence" (filtrirano u pretragama)
  link:                    string   // URL stranice
  section:                 string   // Ime sekcije
  h1:                      string
  h2:                      string
  h3:                      string
  last_h_title:            string
  post_type:               string
  date:                    datetime
  visibility:              string
  site:                    string
  parent_document_hash:    string   // ← VEZA KA DiploDocument.document_hash
  sentence_index:          int
  parent_paragraph_id:     string
}
```

### 3.4 DiploHeading — uklonjeno (mart 2026)

Kolekcija `DiploHeading` (precomputed heading embeddings za URL-level boost) **više se ne koristi**. Heading hijerarhija je ugrađena u contextual vektore (`DiploChunk_contextual`, `DiploParagraph_contextual`). Kolekcija je obrisana iz Weaviate-a; ingest je više ne puni.

### 3.5 Veze između kolekcija (aktivne contextual kolekcije)

```
DiploDocument_contextual
    │
    │  document_hash  ←─────── parent_document_hash
    │
    ├──── DiploParagraph_contextual (N paragrafa po dokumentu)
    │         │
    │         │  parent_paragraph_id
    │         │
    │         └──── DiploChunk_contextual (M rečenica po paragrafu)
    │
    └──── DiploTurn_contextual (transkripti, po potrebi)
```

Legacy standard kolekcije (`DiploDocument`, `DiploChunk`, …) mogu još postojati u Weaviate-u ali se više ne pune pri ingest-u.

---

## 4. Kompletni tok korisničkog upita

### Korak po korak (WebSocket — primarni režim):

```
Korisnik šalje:
{
  "user_ip": "1.2.3.4",
  "message": "What is Diplo's position on AI governance?",
  "user_type": "general",
  "retrieval_config": { ... }  // opciono — override parametri
}
```

#### Faza 1: Prijem i priprema

```
1. WS /api/chat/ws/{conversation_id}
   │
   ├─ Validacija conversation_id (in-memory dict)
   │
   ├─ ChatService.handle_chat_request_WS()
   │     │
   │     ├─ set_retrieval_overrides()     ← contextvar, per-request params
   │     ├─ label_retriever.set_user_type("general")
   │     ├─ _resolve_system_prompt()      ← diplomacy_edu ili humainism_ai prompts
   │     ├─ repack_chat_history()         ← prethodni razgovori
   │     └─ diplomacy_bot.aupdate_state() ← seeduje SystemMessage
```

#### Faza 2: LangGraph izvršavanje

```
   │
   ├─ diplomacy_bot.astream_events() ────────────────────────────────┐
   │                                                                  │
   │    ┌──────────────────────────────────────────────────────┐     │
   │    │         LangGraph: SKIP_TOOL_DECISION=True           │     │
   │    │                                                      │     │
   │    │  START ──▶ direct_retrieval_node ──▶ generate_final ──▶ END│
   │    │              │                                       │     │
   │    │              ▼                                       │     │
   │    │   1. active_retriever.invoke(question)               │     │
   │    │      → Weaviate hybrid search                        │     │
   │    │      → Grupovanje po (url, section)                  │     │
   │    │      → Agregacija skorova                            │     │
   │    │                                                      │     │
   │    │   2. tei_reranker.rerank() [opciono]                 │     │
   │    │      → Cross-encoder rescoring                       │     │
   │    │                                                      │     │
   │    │   3. apply_post_reranker_label_weights()             │     │
   │    │      → Profil-bazirani label weights                 │     │
   │    │      → Dinamički weights po query intent-u           │     │
   │    │      → Truncate na RETRIEVAL_CHUNKS (default 8)      │     │
   │    │                                                      │     │
   │    │   4. Formatira kao ToolMessage:                       │     │
   │    │      "[1] Title\ncontent\n---\n[2] Title\n..."       │     │
   │    │                                                      │     │
   │    │              ▼                                       │     │
   │    │   generate_final_node:                               │     │
   │    │      LLM (bez tools) + "Cite [1], [2]..."           │     │
   │    │      → Streamuje odgovor                             │     │
   │    └──────────────────────────────────────────────────────┘     │
   │                                                                  │
```

#### Faza 3: Streaming i post-processing

```
   │    on_chat_model_stream:
   │    ├─ Live citation renumbering ([4] → [1], [7] → [2]...)
   │    ├─ WebSocket: { "status": "answer", "text": "chunk..." }
   │    └─ WebSocket: { "status": "citation_url", "num": 1, "url": "..." }
   │
   │    on_retriever_end:
   │    ├─ Hvata retrieved_docs (prefer _last_reranked_docs)
   │    ├─ Izvlači related_questions iz metadataobjekata
   │    └─ build_deep_link_urls_batch() → URL-ovi sa highlight parametrima
   │
   ├─ filter_related_sources()
   │     ├─ Parse citations iz odgovora ([1], [2]...)
   │     ├─ Mapira na originalne dokumente
   │     └─ Batch deep link kreiranje
   │
   ├─ __repack_source_documents() → ResponseSource objekti
   │     │
   │     └─ Za svaki izvor:
   │        {
   │          "text": "...",
   │          "title": "AI Governance at Diplo",
   │          "date": "2024-03-15",
   │          "url": "https://diplomacy.edu/...",
   │          "deep_link_url": "https://diplomacy.edu/...?diplo-deep-link-text=..."
   │        }
   │
   ├─ WebSocket: { "status": "sources", "text": [...] }
   │     (samo text/title/date/url/deep_link_url — bez graph_* polja)
   ├─ WebSocket: { "status": "graph_data", "text": { nodes, edges, subgraph_url } }
   │     (Knowledge Graph widget; odvojeno od source kartica)
   ├─ WebSocket: { "status": "related_questions", "text": [...] }
   ├─ WebSocket: { "status": "citation_urls", "text": {...} }
   │
   ├─ MongoDB: save chat + response
   └─ WebSocket: { "status": "message_id", "text": "..." }
```

---

## 5. Retrieval pipeline — detaljan pregled

### 5.1 Retrieval modovi

Postoje 4 moda, bira se putem `RETRIEVAL_MODE`:

| Mod | Retriever klasa | Kolekcije | Opis |
|-----|----------------|-----------|------|
| **`sentence`** (default) | `SentenceFirstRetriever` | `DiploChunk_contextual` | Sentence hybrid → group by section → aggregate |
| **`paragraph`** | `LabelRescoringRetriever` | `DiploParagraph_contextual` | Paragraph hybrid → label rescoring |
| **`combined`** | `CombinedRetriever` | `DiploChunk_contextual` + `DiploParagraph_contextual` | Oba, merge, dedup |
| **`twophase`** | `TwoPhaseRetriever` | `DiploParagraph_contextual` | Vector → section selection (embedding) |

### 5.2 Sentence-First Retriever (detalji)

Ovo je **primarni** i najčešće korišćeni mod.

**Fajl:** `app/ai/ai_services/retrievers/sentence_first.py`

```
Korak 1a: Hybrid Search na DiploChunk_contextual
──────────────────────────────────────────────
  collection.query.hybrid(
    query = "What is Diplo's position on AI governance?",
    vector = embed_query(question),          // TEI embedding
    alpha = 0.6,                             // HYBRID_ALPHA (podesivo)
    fusion_type = RELATIVE_SCORE,
    limit = 200,                             // SENTENCE_RETRIEVAL_K
    filters = chunk_level=="sentence" AND (visibility IS NULL OR visibility!="private")
              [AND parent_document_hash==X]   // opcioni parent filter
              [AND site=="diplomacy.edu"]      // opcioni site filter
    query_properties = ["sentence"]
  )
  → 200 sentence objekata sa score-ovima
  (contextual vektori već enkoduju h1, section_path, last_h_title)

Korak 2: Grupovanje po (url, section)
──────────────────────────────────────
  200+ rečenica → N grupa (sekcija)
  Svaka grupa sadrži:
    - section_context (spojen tekst konteksta)
    - lista rečenica sa skorovima
    - metadata (title, url, date, label, post_type, h1/h2/h3...)

Korak 3: Agregacija skorova
───────────────────────────
  Za svaku grupu:
    aggregate_score = 0.6 * max_score + 0.3 * avg_score + 0.1 * log(1 + count)

  Opcija TOP_N_PER_SECTION: samo top N rečenica doprinose skoru
  → Sprečava inflaciju skora za sekcije sa mnogo rečenica

Korak 3b: Selekcija Best Sentence
──────────────────────────────────
  - SENTENCE_HIGHLIGHT_MODE="single": Cross-encoder bira najbolju rečenicu
  - SENTENCE_HIGHLIGHT_MODE="multi": Sve matched rečenice, CE preskočen

Korak 4: Sortiranje i URL cap
──────────────────────────────
  - Sortiraj po aggregate_score DESC
  - Deduplikacija po sadržaju (hash)
  - MAX_SECTIONS_PER_URL cap (default: neograničeno)

Korak 5: Vraćanje dokumenata
─────────────────────────────
  → final_k Document objekata (default RERANKER_TOP_K ako USE_RERANKER, inače RETRIEVAL_CHUNKS)
  Svaki Document:
    page_content = section_context (ceo tekst sekcije)
    metadata = {
      title, url, date, label, post_type,
      h1, h2, h3, section,
      _aggregate_score, _heading_boost,
      _best_sentence, _best_sentence_score,
      _matched_sentences: ["sent1", "sent2", ...],
      _total_sentences, _scored_sentences,
      _final_score, _label_weight, _user_type,
      _index: "DiploChunk"
    }
```

### 5.3 Metadata koja svaki Retrieved Document ima

Ovo je ono što retriever vraća i što dalje ide kroz pipeline:

```python
Document(
    page_content = "Full section text...",   # kontekst sekcije/paragrafa
    metadata = {
        # Bazni metadaci (iz Weaviate objekta)
        "title":     "AI Governance and Diplomacy",
        "url":       "https://diplomacy.edu/blog/ai-governance/",
        "date":      "2024-03-15",
        "label":     "Blog",              # mapiran iz post_type
        "post_type": "blog",
        "h1":        "AI Governance and Diplomacy",
        "h2":        "Key Findings",
        "h3":        "",
        "section":   "key-findings",

        # Scoring metadaci (dodati tokom pipeline-a)
        "_aggregate_score":    0.8234,
        "_heading_boost":      0.1200,
        "_final_score":        0.8234,
        "_label_weight":       1.42,      # iz profila korisnika
        "_post_label_score":   1.1692,    # _final_score * _label_weight
        "_reranker_score":     0.9100,    # dodat od TEI rerankera

        # Sentence metadaci (za deep link highlighting)
        "_best_sentence":      "Diplo advocates for inclusive AI governance...",
        "_best_sentence_score": 0.91,
        "_matched_sentences":  ["sent1", "sent2", "sent3"],
        "_total_sentences":    12,
        "_scored_sentences":   5,

        # Pipeline metadaci
        "_user_type":  "general",
        "_index":      "DiploChunk",
    }
)
```

**VAŽNO: `document_hash` nije prisutan u metadata-u retrieved dokumenata.**
Retriever ga ne kopira iz Weaviate objekata. Za integraciju sa Neo4j grafom, treba ga ili:
- dodati u retriever output, ili
- razrešiti naknadno putem `url` → `DiploDocument` lookup-a.

---

## 6. Reranking i label weights

### 6.1 TEI Cross-Encoder Reranking

**Fajl:** `app/ai/ai_services/reranker.py`

Šalje retrieved dokumente na TEI reranker API koji koristi cross-encoder model za precizniji scoring:

```
Za svaki dokument priprema tekst:
  "[Blog] AI Governance and Diplomacy > Key Findings

  [BEST MATCH]: Diplo advocates for inclusive AI governance...

  [FULL CONTEXT]: Full section text here..."

→ POST /rerank sa query + texts
→ Vraća reranked listu sa _reranker_score
```

### 6.2 Post-Reranker Label Weights

**Fajl:** `app/ai/ai_services/label_weights.py`

Nakon rerankinga, skorovi se množe label weight-ima prema korisničkom profilu:

```python
# Primer za "general" profil:
{
    "Topic":        1.50,    # Najviši prioritet
    "Course":       1.42,
    "Blog":         1.33,
    "Event":        1.25,
    "Resources":    1.17,
    "Technologies": 1.17,
    "Actor":        1.08,
    "Updates":      1.04,
    "People":       1.00,    # Najniži
}

# Za "journalist" profil:
{
    "Updates":      1.50,    # Novinari preferiraju aktuelnosti
    "Blog":         1.42,
    ...
}
```

### 6.3 Dinamički Label Weights

**Fajl:** `app/ai/ai_services/query_intent.py`

Kada je `USE_DYNAMIC_LABEL_WEIGHTS=True`:
1. Prototipska pitanja (iz `config/prototype_questions.txt`) se embeduju pri startupu
2. Za svaki upit, izračunava se kosinus sličnost sa prototipovima
3. Label weights se dinamički prilagođavaju na osnovu query intent-a

```
Formula:
  post_label_score = base_score * label_weight

Sortiranje:
  Dokumenti se sortiraju po post_label_score DESC
  Truncate na RETRIEVAL_CHUNKS (default 8)
```

---

## 7. LLM prompt konstrukcija

### 7.1 Slojevi prompt-a

LLM prima poruke u sledećem redosledu:

```
┌─ SystemMessage (role prompt) ──────────────────────────────────────────┐
│  "Context: Your name is Diplo AI Assistant. You are an AI assistant    │
│   designed specifically for a platform that explores the intersection  │
│   of diplomacy, technology, and global governance..."                  │
│                                                                        │
│  + Specifičan za user_type (general/student/diplomat/researcher...)     │
│  + Fallback prompt (kontaktirajte ask@diplomacy.edu)                  │
└────────────────────────────────────────────────────────────────────────┘

┌─ SystemMessage (graph prompt) ─────────────────────────────────────────┐
│  "You are a helpful AI assistant, whose name is Diplorene.             │
│   You have access to diplo_tool, and you should ALWAYS use it to      │
│   retrieve information..."                                             │
└────────────────────────────────────────────────────────────────────────┘

┌─ Prethodni razgovori (ako postoje) ──────────────────────────────────┐
│  HumanMessage: "prethodno pitanje"                                    │
│  AIMessage: "prethodni odgovor"                                       │
└────────────────────────────────────────────────────────────────────────┘

┌─ AIMessage (sintetisan tool call) ─────────────────────────────────────┐
│  tool_calls: [{"name": "diplo_tool", "args": {"question": "..."}}]   │
└────────────────────────────────────────────────────────────────────────┘

┌─ ToolMessage (retrieved context) ──────────────────────────────────────┐
│  "[1] AI Governance and Diplomacy                                      │
│   Diplo advocates for inclusive AI governance frameworks...            │
│                                                                        │
│   ---                                                                  │
│                                                                        │
│   [2] Digital Policy Developments                                      │
│   Recent developments in AI regulation include..."                     │
│                                                                        │
│   ---                                                                  │
│                                                                        │
│   [3] Events on AI Governance                                          │
│   The 2024 AI Governance Forum..."                                     │
└────────────────────────────────────────────────────────────────────────┘

┌─ HumanMessage (pitanje korisnika) ─────────────────────────────────────┐
│  "What is Diplo's position on AI governance?"                          │
└────────────────────────────────────────────────────────────────────────┘

┌─ HumanMessage (citation instrukcija) ──────────────────────────────────┐
│  "Based on the information above, please answer the question.          │
│   IMPORTANT: Cite your sources using [1], [2], etc."                   │
└────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Dva režima grafa

| Režim | Tok | Kada koristiti |
|-------|-----|----------------|
| **SKIP_TOOL_DECISION=True** | `START → direct_retrieval → generate_final → END` | Jedan tool (diplo_tool) — uvek se koristi |
| **SKIP_TOOL_DECISION=False** | `START → chatbot → tools → chatbot → ... → generate_final → END` | Više alata, LLM bira koji |

U praksi, `SKIP_TOOL_DECISION=True` je standardna konfiguracija — preskoči se nepotreban LLM poziv za odluku o alatu i uštedi ~1s.

---

## 8. Generisanje odgovora i sources

### 8.1 Streaming odgovora

```python
# WebSocket streaming sa live citation renumberingom:
#
# LLM generiše:  "According to [1], AI governance is... [4] shows that..."
#
# Renumbering:   [1] → [1] (prvi put viđen, ostaje)
#                [4] → [2] (drugi po redu, renumerisan)
#
# Klijentu se šalje:
#   { "status": "answer", "text": "According to [1], AI governance is... " }
#   { "status": "citation_url", "num": 1, "url": "https://..." }
#   { "status": "answer", "text": "[2] shows that..." }
#   { "status": "citation_url", "num": 2, "url": "https://..." }
```

### 8.2 Source dokumenti

Svaki izvor koji se vraća klijentu ima strukturu:

```python
class ResponseSource(BaseModel):
    text:          str              # Tekst sekcije + deep link payload
    date:          str              # Datum objave
    title:         str              # Naslov dokumenta
    url:           str              # URL stranice
    link:          Optional[str]    # Alternativa za url (DiploParagraph)
    name:          Optional[str]    # Alternativa za title (DiploParagraph)
    deep_link_url: Optional[str]    # URL sa highlight parametrima
```

### 8.3 Deep Links

Deep link URL-ovi omogućavaju highlighting teksta na izvornoj stranici:

```
# Paragraph mode:
https://diplomacy.edu/blog/ai-governance/?diplo-deep-link-text=AI+Governance...

# Sentence mode (multi):
https://diplomacy.edu/blog/ai-governance/?diplo-hl-id=abc123&diplo-hl-api=...

# PDF mode (preko chatbot-via proxy):
https://chatbot-via.diplomacy.edu/https://example.com/doc.pdf#diplo-deep-link-text=...
```

### 8.4 Related Questions

Generišu se iz dva izvora:
1. **Metadata:** `questions_this_excerpt_can_answer` polje iz chunk-ova
2. **LLM fallback:** Ako metadata nema pitanja, LLM generiše 3-5 follow-up pitanja

### 8.5 Knowledge Graph (`graph_data`)

Neo4j graph expansion obogaćuje retrieval i LLM prompt, ali **ne šalje** `graph_topics` / `graph_people` / `graph_actors` na source karticama koje vidi WordPress plugin.

| Izlaz | Polja | Namena |
|-------|-------|--------|
| `sources[]` | `text`, `title`, `date`, `url`, `deep_link_url` | Citirani chunk-ovi (isti oblik kao pre Neo4j integracije na karticama) |
| `graph_data` (HTTP) ili WS `status: "graph_data"` | `nodes`, `edges`, `subgraph_url` | Knowledge Graph vizualizacija u chatbot widgetu |
| `Document.metadata` (interno) | `_graph_topics`, `_graph_people`, … | LLM kontekst, boost, debug (`/api/debug/retrieve`) |

`GRAPH_EXPANSION_ENABLED=False` gasi ceo graph subsystem (boost, `graph_data`, Neo4j health). Source kartice ostaju istog oblika u oba slučaja.

---

## 9. Ključni fajlovi i gde šta živi

```
chatbot-api/
├── main.py                                    # FastAPI app factory
├── run_chatbot.py                           # Uvicorn runner
├── app/
│   ├── api/routes/
│   │   ├── api.py                             # Agregacija svih routera
│   │   ├── chat_route.py                      # POST /chat/{id}, WS /chat/ws/{id}
│   │   ├── conversation_route.py              # POST /conversation/get_id
│   │   ├── ingest_route.py                    # POST /ingest/topic, /topics/all
│   │   └── debug_route.py                     # Debug endpoints
│   │
│   ├── services/
│   │   ├── chat_service.py                    # ⭐ GLAVNI ORKESTRATOR
│   │   │                                      #    handle_chat_request()
│   │   │                                      #    handle_chat_request_WS()
│   │   │                                      #    filter_related_sources()
│   │   │                                      #    build_deep_link_url()
│   │   │                                      #    build_deep_link_urls_batch()
│   │   ├── conversation_service.py            # Conversation ID management
│   │   ├── ingest_service.py                  # Topic ingestion + heading population
│   │   ├── deep_link_service.py               # Redis shortener za deep links
│   │   ├── user_type_service.py               # Mapira user_type → system prompt
│   │   └── chatService.py                     # Legacy: repack_chat_history
│   │
│   ├── ai/
│   │   ├── ai_services/
│   │   │   ├── graph.py                       # ⭐ LANGGRAPH DEFINICIJA
│   │   │   │                                  #    initialize_diplomacy_bot_graph()
│   │   │   │                                  #    direct_retrieval_node()
│   │   │   │                                  #    generate_final_node()
│   │   │   │                                  #    diplo_tool()
│   │   │   │                                  #    apply_post_reranker_label_weights()
│   │   │   ├── embeddings.py                  # TEI embedding klijent
│   │   │   ├── reranker.py                    # TEI cross-encoder reranker
│   │   │   ├── label_weights.py               # Profil-bazirani label weights
│   │   │   ├── query_intent.py                # Dinamički label weights
│   │   │   ├── retrieval_cache.py             # Redis cache za retrieval
│   │   │   └── retrievers/
│   │   │       ├── factory.py                 # create_retriever() — bira mod
│   │   │       ├── sentence_first.py          # ⭐ SentenceFirstRetriever
│   │   │       ├── label_rescoring.py         # LabelRescoringRetriever (paragraph)
│   │   │       ├── combined.py                # CombinedRetriever
│   │   │       ├── twophase.py                # TwoPhaseRetriever
│   │   │       └── utils.py                   # resolve_parent_document_hash()
│   │   │                                      # visibility_filter()
│   │   │                                      # resolve_collection_name()
│   │   │                                      # group_paragraphs_by_url()
│   │   └── prompts/
│   │       ├── diplomacy_edu_prompts.py       # System promptovi po user_type
│   │       └── humainisam_ai_prompts.py       # Za humainism.ai sajt
│   │
│   ├── core/
│   │   ├── config.py                          # Sve env varijable
│   │   ├── singleton.py                       # App singleton (bot, retriever, WS)
│   │   ├── retrieval_context.py               # Per-request contextvar override
│   │   ├── blocklist.py                       # Sentence blocklist filter
│   │   ├── events.py                          # Startup/shutdown (MongoDB, Weaviate)
│   │   ├── logging.py                         # Loguru konfiguracija
│   │   └── websocket_manager.py               # WS connection manager
│   │
│   ├── schemas/
│   │   └── chat_schema.py                     # ChatRouteRequest, ResponseSource,
│   │                                          # RetrievalConfig, UserType
│   ├── models/
│   │   └── chat_domain.py                     # MongoEngine modeli
│   │
│   ├── repositories/
│   │   ├── chat_repository.py                 # MongoDB CRUD za poruke
│   │   └── response_repository.py             # MongoDB CRUD za odgovore
│   │
│   └── utils/
│       └── redis_client.py                    # Redis connection singleton
│
├── config/
│   ├── sentence_blocklist.txt                 # Rečenice za filtriranje
│   └── prototype_questions.txt                # Prototipska pitanja za query intent
│
└── scripts/
    ├── ingest_topics.py                       # CLI: topic ingestion
    ├── delete_topics.py                       # CLI: brisanje topic podataka
    ├── benchmark_weaviate_collections.py      # Benchmark alat
    └── export_random_chunk.py                 # Izvoz primera chunk-a
```

---

## 10. document_hash i parent_document_hash

### Kako se razrešava

```python
# utils.py — resolve_parent_document_hash()
#
# Koristi se KAD klijent pošalje parent_filter_name u RetrievalConfig:
#   "retrieval_config": { "parent_filter_name": "AI and Diplomacy" }
#
# 1. Query DiploDocument kolekciju:
#    DiploDocument.fetch_objects(filters=name.equal("AI and Diplomacy"), limit=50)
#
# 2. Post-filter u Python-u za exact match (Weaviate radi token-level matching)
#
# 3. Vrati document_hash (npr. "c58ee7de1137f2454c33f4d9a79c4e22")
#
# 4. Svi dalji retrieval upiti dodaju filter:
#    Filter.by_property("parent_document_hash").equal("c58ee7de1137f2454c33f4d9a79c4e22")
```

### Gde se document_hash koristi

| Kontekst | Kako |
|----------|------|
| **DiploDocument** | `document_hash` je primarni identifikator dokumenta |
| **DiploParagraph** | `parent_document_hash` referencira `DiploDocument.document_hash` |
| **DiploChunk** | `parent_document_hash` referencira `DiploDocument.document_hash` |
| **DiploHeading** | uklonjeno (mart 2026) |
| **Retriever output** | **NE SADRŽI** `document_hash` u metadata-u (!) |
| **RetrievalConfig** | `parent_filter_name` → resolve → filter na `parent_document_hash` |
| **RetrievalConfig** | `person_filter_name` / `person_filter_names` → `person.contains_any([...])` na chunk/paragraph kolekcijama (OR) |
| **RetrievalConfig** | `city_filter_name` / `city_filter_names` → `city.contains_any([...])` (OR) |
| **RetrievalConfig** | `country_filter_name` / `country_filter_names` → `country.contains_any([...])` (OR) |
| **RetrievalConfig** | `organisation_filter_name` / `organisation_filter_names` → `organisation.contains_any([...])` (OR) |
| **RetrievalConfig** | `site_filter_name`, `post_types` → Weaviate filteri na `site` / `post_type` |

### Kako doći do document_hash iz retriever output-a

Retriever output sadrži `url` ali **ne** `document_hash`. Za dobijanje hash-a:

```python
# Opcija A: Weaviate lookup (već postoji u utils.py)
from app.ai.ai_services.retrievers.utils import resolve_parent_document_hash
hash = resolve_parent_document_hash(client, document_name, site_name="diplomacy.edu")

# Opcija B: Dodati u retriever da kopira parent_document_hash iz Weaviate objekata
# (treba modifikovati sentence_first.py ili utils.py)
```

---

## 11. Tačka integracije za Neo4j Graph Expansion

### Predloženi flow sa Neo4j integracijom

```
Korisnik: "What is Diplo's position on AI governance?"
    │
    ▼
1. Trenutni retrieval pipeline (nepromenjeni)
   → Vraća N dokumenata, svaki sa url, title, label, post_type
   → Metadata NEMA document_hash (treba dodati ili razrešiti)
    │
    ▼
2. ★ NOVO: Razrešavanje document_hash za svaki retrieved dokument
   → Za svaki doc: lookup u DiploDocument po url ili name
   → Rezultat: lista document_hash-ova (npr. "abc123", "def456")
    │
    ▼
3. ★ NOVO: Neo4j graph expansion
   → MATCH (d:Document {document_hash: "abc123"})-[r]->(related)
   → Saznaje:
      - ovaj blog je RELATED_BLOG_AND_PEOPLE → Jovan Kurbalija
      - ovaj blog je RELATED_BLOG_&_TOPICS → AI diplomacy
      - AI diplomacy je SUBTOPIC_OF → Types of diplomacy
      - Postoji resource "AI and Diplomacy" sa PDF-om
    │
    ▼
4. ★ NOVO: Obogaćivanje konteksta
   → Originalni chunkovi + strukturalne informacije iz grafa
   → Formatirati kao dodatni kontekst za LLM
    │
    ▼
5. LLM generiše odgovor sa bogatijim kontekstom
   → Originalni chunkovi + graf relacije
   → LLM sada zna ko je autor, koji su povezani resursi, koji eventi
    │
    ▼
6. Bolji odgovor sa citatima i kontekstom
```

### Gde u kodu se integriše

**Opcija A: U `direct_retrieval_node` (preporučeno za SKIP_TOOL_DECISION=True)**

```
Fajl: app/ai/ai_services/graph.py
Funkcija: direct_retrieval_node()

Linija ~435: docs = apply_post_reranker_label_weights(docs, _chunks, ...)
                                                                    ↓
★ OVDE UBACITI Neo4j graph expansion:                              ↓
  1. Za svaki doc u docs:                                          ↓
     - Razreši document_hash (iz url-a ili dodati u retriever)     ↓
  2. Za svaki document_hash:                                       ↓
     - Query Neo4j graf za relacije                                ↓
  3. Obogati tool_content sa graf informacijama                    ↓
                                                                    ↓
Linija ~452-457: formatira tool_content za LLM
```

**Opcija B: U `diplo_tool` (za SKIP_TOOL_DECISION=False)**

```
Fajl: app/ai/ai_services/graph.py
Funkcija: diplo_tool()

Isti princip — nakon reranking-a i label weights-a, pre vraćanja rezultata.
```

**Opcija C: Kao novi LangGraph node (najčistija arhitektura)**

```python
# Novi node između direct_retrieval i generate_final:
#
#   START → direct_retrieval → ★graph_expansion★ → generate_final → END
#
async def graph_expansion_node(state: State):
    # 1. Izvuci dokumente iz prethodnog koraka
    # 2. Razreši document_hash-ove
    # 3. Query Neo4j
    # 4. Obogati poruke sa graf kontekstom
    # 5. Vrati obogaćeni state
```

### Šta treba pripremiti na Neo4j serveru

Za svaki `document_hash` koji chatbot vrati, Neo4j treba moći da odgovori na:

```cypher
// Osnovni query za relacije dokumenta
MATCH (d:Document {document_hash: $hash})-[r]->(related)
RETURN type(r) AS relationship, related.name AS name,
       related.document_hash AS hash, labels(related) AS types

// Širi kontekst — 2 nivoa dubine
MATCH (d:Document {document_hash: $hash})-[r1]->(n1)-[r2]->(n2)
RETURN type(r1), n1.name, type(r2), n2.name
LIMIT 50
```

### Ključne informacije za integraciju

| Podatak | Gde ga naći | Kako ga dobiti |
|---------|-------------|----------------|
| `document_hash` | DiploDocument (Weaviate) | `resolve_parent_document_hash(client, name)` |
| `url` | Retriever output `metadata["url"]` | Dostupan odmah |
| `title` | Retriever output `metadata["title"]` | Dostupan odmah |
| `post_type` | Retriever output `metadata["post_type"]` | Dostupan odmah |
| `label` | Retriever output `metadata["label"]` | Dostupan odmah |
| `site` | Config `WEBSITE_NAME` ili filter | Dostupan odmah |

### Per-request override mehanizam

Frontend može slati konfiguraciju za retrieval putem `retrieval_config`:

```json
{
  "message": "What is AI governance?",
  "user_type": "general",
  "retrieval_config": {
    "retrieval_chunks": 10,
    "hybrid_alpha": 0.9,
    "parent_filter_name": "AI and Diplomacy",
    "site_filter_name": "diplomacy.edu",
    "person_filter_name": "Jovan Kurbalija"
  }
}
```

Ovo koristi `contextvars` mehanizam (`retrieval_context.py`) za thread-safe per-request override. Isti mehanizam se može koristiti za Neo4j parametre (npr. `max_graph_depth`, `enable_graph_expansion`).

---

---

## 12. Neo4j Integration Kit (`chatbot-neo4j/`)

Folder `chatbot-neo4j/` sadrži pripremljene module za integraciju. **Nije standalone servis** — moduli se kopiraju u `chatbot-api/app/` strukturu.

### 12.1 Struktura kit-a

```
chatbot-neo4j/
├── env.neo4j.example                           # Env varijable za .env
├── integration_patch.py                        # Dokumentacija — gde šta dodati
└── app/
    ├── core/
    │   └── neo4j_config.py                     # Config modul (NEO4J_URI, database mapping)
    └── ai/
        └── ai_services/
            ├── neo4j_graph_client.py           # Async Neo4j klijent sa Cypher upitima
            ├── graph_expansion.py              # Servis: hash resolution → expand → format
            └── graph_expansion_node.py         # LangGraph node (Option C)
```

### 12.2 Neo4j šema grafa

**Node tipovi:**

| Label | Opis | Ključni property |
|-------|------|------------------|
| `Document` | Svaki dokument iz Weaviate | `document_hash` (MD5 URL-a) |
| `Topic` | Tematska kategorija | `name` |
| `TopicBasket` | Korpa tema | `name` |
| `Tag` | Oznaka/ključna reč | `name` |
| `Person` / `Expert` | Osoba/ekspert | `name` |
| `Actor` | Organizacija | `name` |
| `Date` | Datum (nije surfejsovan u LLM) | — |
| `Blog`, `Event`, `Resource`, `Course`... | Content čvorovi | `document_hash`, `url` |

**Relationship tipovi (primeri):**

```
(Document)-[RELATED_BLOG_AND_PEOPLE]->(Person)
(Document)-[RELATED_BLOG_&_TOPICS]->(Topic)
(Topic)-[SUBTOPIC_OF]->(Topic)          # hijerarhija do 3 nivoa
(Document)-[r]->(Topic)                  # gde type(r) CONTAINS 'TOPICS'
(Document)-[r]->(Tag)
(Document)-[r]->(Actor)
(Document)-[r]->(Document)              # incoming relacije od drugih dokumenata
```

**Veza sa Weaviate:**

```
Neo4j Document.document_hash = MD5(url)
                              ≈ Weaviate DiploDocument.document_hash
                              ≈ Weaviate DiploChunk.parent_document_hash
```

### 12.3 Neo4j baze podataka

| Sajt | Neo4j baza |
|------|-----------|
| `diplomacy.edu` | `weaviatediplo` |
| `dig.watch` | `weaviatedw` |

### 12.4 Dataclass-ovi

```python
@dataclass
class GraphNode:
    node_id: str
    name: str
    labels: list[str]           # ["Document", "Blog"]
    document_hash: str = ""
    post_type: str = ""
    url: str = ""
    site: str = ""
    properties: dict = {}

@dataclass
class GraphRelation:
    source_hash: str            # document_hash izvora
    source_name: str
    relationship: str           # tip relacije (npr. "RELATED_BLOG_AND_PEOPLE")
    target_hash: str
    target_name: str
    target_labels: list[str]    # ["Person"], ["Topic"], ["Blog", "Document"]...
    target_url: str = ""
    target_post_type: str = ""

@dataclass
class GraphExpansionResult:
    document_hash: str
    document_name: str
    relations: list[GraphRelation]    # sve relacije (outgoing + incoming)
    topics: list[str]                 # ["AI diplomacy", "Digital governance"]
    related_documents: list[GraphNode] # povezani blogovi, eventi, resursi
    tags: list[str]                   # ["artificial intelligence", "policy"]
    people: list[str]                 # ["Jovan Kurbalija", "Katharina Höne"]
    actors: list[str]                 # ["UN", "ITU", "ICANN"]
    subtopic_of: list[str]            # ["AI diplomacy → Types of diplomacy → Diplomacy"]
```

### 12.5 Tok graph expansion-a

```
Retrieved Documents (post-reranker, post-label-weights)
    │
    ▼
1. Hash Resolution (GraphExpansionService.resolve_hashes_from_docs)
   ├─ Ako doc.metadata ima parent_document_hash → koristi ga
   └─ Inače → MD5(url) + MD5(www variant) + MD5(trailing slash variant)
    │
    ▼
2. Verifikacija hash-ova u Neo4j (batch UNWIND query)
   UNWIND $hashes AS h
   MATCH (d:Document {document_hash: h})
   RETURN d.document_hash
    │
    ▼
3. URL fallback za nerazrešene (get_document_by_url sa normalizacijom)
   MATCH (d:Document) WHERE d.url IN $variants
    │
    ▼
4. Batch Expansion (paralelno za sve dokumente)
   Za svaki document_hash, paralelno:
   ├─ get_direct_relations:    (d)-[r]->(target)       LIMIT 30
   ├─ get_incoming_relations:  (source)-[r]->(d)       LIMIT 15
   └─ get_topic_hierarchy:     (d)-[r]->(t:Topic)-[:SUBTOPIC_OF*1..3]->(parent)
    │
    ▼
5. Agregacija (_aggregate_relations)
   ├─ Topics:    target labels sadrže Topic/TopicBasket
   ├─ People:    target labels sadrže Person/Expert
   ├─ Actors:    target labels sadrže Actor
   ├─ Tags:      target labels sadrže Tag
   ├─ Related:   target labels sadrže Blog/Event/Resource/Course/...
   └─ Hierarchy: topic → parent_topic → grandparent
    │
    ▼
6. Formatiranje za LLM (format_graph_context)
   --- KNOWLEDGE GRAPH CONTEXT ---

   Graph context for "AI Governance and Diplomacy":
     Topics: AI diplomacy, Digital governance, Internet governance
     Topic hierarchy: AI diplomacy → Types of diplomacy → Diplomacy
     Related people: Jovan Kurbalija, Katharina Höne
     Related organizations: UN, ITU
     Related content: Geneva Internet Platform [Resource]; AI Summit 2024 [Event]
     Tags: artificial intelligence, policy, governance
    │
    ▼
7. Dodavanje u ToolMessage
   tool_content = original_retrieval_text + graph_context
```

### 12.6 Timeout i Failsafe mehanizam

```
GRAPH_EXPANSION_TIMEOUT = 3.0s (ukupno)
├─ Phase 1 (hash verify):   40% budžeta → 1.2s
├─ Phase 2 (URL fallback):  30% budžeta → 0.9s
└─ Phase 3 (batch expand):  100% budžeta → 3.0s

Ako bilo koji korak timeout-uje → vraća originalne rezultate BEZ graph konteksta
Ako Neo4j nije dostupan → GRAPH_EXPANSION_ENABLED=False, pipeline radi normalno
```

---

## 13. Korak-po-korak integracija

### 13.1 Kopiranje fajlova

```bash
# Iz chatbot-neo4j/ u chatbot-api/
cp chatbot-neo4j/app/core/neo4j_config.py          chatbot-api/app/core/
cp chatbot-neo4j/app/ai/ai_services/neo4j_graph_client.py  chatbot-api/app/ai/ai_services/
cp chatbot-neo4j/app/ai/ai_services/graph_expansion.py     chatbot-api/app/ai/ai_services/
cp chatbot-neo4j/app/ai/ai_services/graph_expansion_node.py chatbot-api/app/ai/ai_services/
```

### 13.2 Dodavanje env varijabli u `.env`

```env
# Neo4j Graph Expansion
NEO4J_URI=bolt://nimani.diplomacy.edu:7687
NEO4J_USER=neo4j
NEO4J_PASS=<lozinka>
NEO4J_DATABASE_DIPLO=weaviatediplo
NEO4J_DATABASE_DW=weaviatedw
GRAPH_EXPANSION_ENABLED=False           # Počni sa False, uključi kad testiraš
GRAPH_EXPANSION_TIMEOUT=3.0
GRAPH_EXPANSION_MAX_RELATIONS=30
GRAPH_EXPANSION_MAX_DEPTH=1
```

### 13.3 Dodavanje dependency-ja

```
# U pyproject.toml ili requirements.txt:
neo4j>=5.17.0
```

### 13.4 Modifikacija `singleton.py`

Dodati na vrh fajla i nove funkcije:

```python
# Imports
from typing import Optional
from app.core.neo4j_config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASS,
    NEO4J_DATABASE_DIPLO, NEO4J_DATABASE_DW,
    GRAPH_EXPANSION_ENABLED,
)

# Module-level
_neo4j_client = None
_graph_expansion_service = None

async def init_graph_expansion():
    global _neo4j_client, _graph_expansion_service
    if not GRAPH_EXPANSION_ENABLED:
        return
    from app.ai.ai_services.neo4j_graph_client import Neo4jGraphClient
    from app.ai.ai_services.graph_expansion import GraphExpansionService
    _neo4j_client = Neo4jGraphClient(
        NEO4J_URI, NEO4J_USER, NEO4J_PASS,
        NEO4J_DATABASE_DIPLO, NEO4J_DATABASE_DW,
    )
    await _neo4j_client.connect()
    healthy = await _neo4j_client.health_check()
    if healthy:
        _graph_expansion_service = GraphExpansionService(_neo4j_client)
        print(f"[STARTUP] Neo4j graph expansion: ENABLED ({NEO4J_URI})")
    else:
        print(f"[STARTUP] Neo4j graph expansion: FAILED health check")
        await _neo4j_client.close()
        _neo4j_client = None

def get_graph_expansion_service():
    return _graph_expansion_service

async def shutdown_graph_expansion():
    global _neo4j_client, _graph_expansion_service
    _graph_expansion_service = None
    if _neo4j_client:
        await _neo4j_client.close()
        _neo4j_client = None
```

### 13.5 Modifikacija `events.py`

```python
# U startup handler:
from app.core.singleton import init_graph_expansion
await init_graph_expansion()

# U shutdown handler:
from app.core.singleton import shutdown_graph_expansion
await shutdown_graph_expansion()
```

### 13.6 Modifikacija `graph.py` — Opcija A (inline, najjednostavnija)

U `direct_retrieval_node()`, **posle** `apply_post_reranker_label_weights` a **pre** formatiranja `tool_content`:

```python
# ---- posle linije: docs = apply_post_reranker_label_weights(docs, _chunks, ...) ----

# Graph Expansion
from app.core.singleton import get_graph_expansion_service
graph_context = ""
graph_service = get_graph_expansion_service()
if graph_service is not None:
    try:
        _site = get_param('site_filter_name', None) or "diplomacy.edu"
        expansions, graph_elapsed = await graph_service.expand_documents(docs, site=_site)
        if expansions:
            graph_service.enrich_document_metadata(docs, expansions)
            graph_context = graph_service.format_graph_context(docs, expansions)
            print(f"TIMING graph_expansion: {graph_elapsed:.3f}s "
                  f"({len(expansions)}/{len(docs)} docs expanded)")
    except Exception as e:
        print(f"Graph expansion failed (non-fatal): {e}")

# ---- zatim u formatiranju tool_content, dodaj graph_context: ----
# tool_content = "\n\n---\n\n".join(formatted_docs) + graph_context
```

### 13.7 Modifikacija `graph.py` — Opcija C (novi LangGraph node, najčistija)

```python
# U initialize_diplomacy_bot_graph():

if SKIP_TOOL_DECISION:
    from app.ai.ai_services.graph_expansion_node import graph_expansion_node

    g2.add_node('direct_retrieval', direct_retrieval_node)
    g2.add_node('graph_expansion', graph_expansion_node)  # NOVO
    g2.add_node('generate_final', generate_final_node)

    g2.add_edge(START, 'direct_retrieval')
    g2.add_edge('direct_retrieval', 'graph_expansion')    # PROMENJENO
    g2.add_edge('graph_expansion', 'generate_final')      # NOVO
    g2.add_edge('generate_final', END)

# direct_retrieval_node() treba da sačuva docs u state:
#   return {
#       'messages': [ai_msg, tool_msg],
#       'tool_call_count': 1,
#       '_retrieved_docs': docs,      # NOVO
#       '_site': site_name,           # NOVO
#   }
```

### 13.8 Opcioni: Debug endpoint

U `debug_route.py`:

```python
@router.get("/debug/neo4j-health")
async def neo4j_health():
    from app.core.singleton import get_graph_expansion_service
    service = get_graph_expansion_service()
    if service is None:
        return {"status": "disabled", "graph_expansion_enabled": False}
    healthy = await service._client.health_check()
    return {"status": "ok" if healthy else "error", "graph_expansion_enabled": True}
```

### 13.9 Opcioni: Frontend per-request control

U `chat_schema.py`, klasa `RetrievalConfig`:

```python
enable_graph_expansion: Optional[bool] = None
graph_max_relations: Optional[int] = None
```

---

## 14. Primer kompletnog toka sa Neo4j

```
Korisnik: "What is Diplo's position on AI governance?"
    │
    ▼
1. SentenceFirstRetriever
   → 200 rečenica iz DiploChunk (hybrid search)
   → Grupovanje u ~50 sekcija po (url, section)
   → Top 8 sekcija posle agregacije + reranking + label weights
   → Rezultat:
     Doc[1]: url=diplomacy.edu/blog/ai-governance, title="AI Governance at Diplo"
     Doc[2]: url=diplomacy.edu/topics/artificial-intelligence, title="Artificial Intelligence"
     Doc[3]: url=dig.watch/updates/ai-summit-2024, title="AI Summit 2024"
     ...
    │
    ▼
2. Hash Resolution
   Doc[1]: md5("https://diplomacy.edu/blog/ai-governance") = "abc123..."
   Doc[2]: md5("https://diplomacy.edu/topics/artificial-intelligence") = "def456..."
   Doc[3]: md5("https://dig.watch/updates/ai-summit-2024") = "789xyz..."
    │
    ▼
3. Neo4j: Verify hashes exist → "abc123", "def456" found, "789xyz" not found
   URL fallback for Doc[3] → found by URL match → hash="789xyz2..."
    │
    ▼
4. Neo4j: Batch Expand (paralelno)
   Doc[1] "abc123":
     ├─ Topics: ["AI governance", "Digital policy"]
     ├─ People: ["Jovan Kurbalija"]
     ├─ Actors: ["UN", "EU"]
     ├─ Related: ["Geneva Internet Platform" [Resource]]
     └─ Hierarchy: "AI governance → Internet governance → Governance"

   Doc[2] "def456":
     ├─ Topics: ["Artificial intelligence", "Machine learning"]
     ├─ Tags: ["AI", "technology", "ethics"]
     └─ Hierarchy: "Artificial intelligence → Technologies → Digital"
    │
    ▼
5. LLM prima:
   ┌──────────────────────────────────────────────────────────────┐
   │ [1] AI Governance at Diplo                                   │
   │ Diplo advocates for inclusive AI governance frameworks...     │
   │                                                              │
   │ ---                                                          │
   │                                                              │
   │ [2] Artificial Intelligence                                  │
   │ AI is transforming diplomacy through...                      │
   │                                                              │
   │ --- KNOWLEDGE GRAPH CONTEXT ---                              │
   │                                                              │
   │ Graph context for "AI Governance at Diplo":                  │
   │   Topics: AI governance, Digital policy                      │
   │   Topic hierarchy: AI governance → Internet governance       │
   │   Related people: Jovan Kurbalija                            │
   │   Related organizations: UN, EU                              │
   │   Related content: Geneva Internet Platform [Resource]       │
   │                                                              │
   │ Graph context for "Artificial Intelligence":                 │
   │   Topics: Artificial intelligence, Machine learning          │
   │   Topic hierarchy: AI → Technologies → Digital               │
   │   Tags: AI, technology, ethics                               │
   └──────────────────────────────────────────────────────────────┘
    │
    ▼
6. LLM odgovor:
   "Diplo's position on AI governance emphasizes inclusive frameworks [1].
    Their work on artificial intelligence [2] connects to broader digital
    policy through experts like Jovan Kurbalija, and organizations such as
    the UN and EU are key partners. The Geneva Internet Platform provides
    additional resources on this topic."
```

---

## Napomene za implementaciju

1. **Redis cache**: Retrieval cache (`RETRIEVAL_CACHE_TTL`) kešira post-reranker rezultate. Graph expansion dolazi POSLE keša — keširana pretraga + svež graph kontekst. Ako želiš keširati i graph rezultate, dodaj zasebni cache.

2. **Performance budžet**: `GRAPH_EXPANSION_TIMEOUT=3.0s` je default. U praksi, sa 3 paralelna Cypher upita, očekuj ~0.5-1.5s za 8 dokumenata. Monitoring: `print(f"TIMING graph_expansion: {elapsed:.3f}s")`.

3. **Failsafe**: Ako Neo4j ne odgovori u okviru timeout-a, pipeline nastavlja sa originalnim rezultatima. Ovo je ugrađeno u `GraphExpansionService.expand_documents()` — svaka faza ima `asyncio.wait_for` sa delom ukupnog budžeta.

4. **document_hash podudaranje**: Neo4j koristi `MD5(url)`, Weaviate `DiploDocument` koristi svoj `document_hash`. Ova dva hash-a se **ne moraju poklapati** — `GraphExpansionService` prvo probava `parent_document_hash` iz metadata-e, pa `MD5(url)` sa URL varijantama, pa URL lookup u Neo4j-u kao fallback.

5. **GRAPH_EXPANSION_ENABLED=False** je default — uključi tek kad potvrdiš da Neo4j server radi i da je baza populisana.

6. **Opcija A vs C**: Opcija A (inline u `direct_retrieval_node`) je brža za implementaciju. Opcija C (novi LangGraph node) je čistija arhitekturalno ali zahteva promenu State TypedDict-a da podrži `_retrieved_docs` polje.
