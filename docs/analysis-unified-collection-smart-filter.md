# Analiza: Unified Collection + Smart Filter

**Datum:** 2026-03-18  
**Kontekst:** Jocin zahtev za spajanje kolekcija u jednu i implementaciju "pametnog filtera"

---

## 1. Trenutno stanje (Dev Weaviate — `127.0.0.1:8591`)

| Kolekcija | Objekata | Status |
|---|---|---|
| `DiploChunk_contextual` | 3,066,729 | **AKTIVNA** — sentence-level chunkovi |
| `DiploParagraph_contextual` | 1,122,062 | **AKTIVNA** — paragraph-level chunkovi |
| `DiploDocument_contextual` | 46,520 | **AKTIVNA** — metapodaci dokumenata |
| `DiploTurn_contextual` | 4,936 | **AKTIVNA** — turn-level podaci iz transkripata |
| `DiploHeading` | — | **OBRISANO** (mart 2026) — heading hijerarhija u contextual vektorima |
| `DiploChunk` | 3,243,623 | Stara non-contextual kopija |
| `DiploParagraph` | 1,202,284 | Stara non-contextual kopija |
| `DiploDocument` | 48,564 | Stara non-contextual kopija |
| `DiploTurn` | 4,936 | Stara non-contextual kopija |

**Ukupno: 9 kolekcija, od kojih su 4 aktivne, 1 obsolete, 4 stare kopije.**

### DiploHeading — uklonjeno

Sa `USE_CONTEXTUAL_COLLECTIONS = true`, heading hijerarhija je u contextual vektorima. Kolekcija `DiploHeading` je obrisana iz Weaviate-a; ingest i retrieval više je ne koriste (Steps 1b/1c uklonjeni iz retrievera).

---

## 2. Jocin zahtev (sumarizacija)

1. **Spajanje 3 kolekcije u jednu** — svi zapisi imaju iste propertije, prazna polja su OK.
2. **"Pametan" filter** — korisnik ukuca npr. "Jovan Kurbalija" u jedno polje i dobije:
   - Blog postove gde je on autor
   - Transkripte gde je bio govornik
   - Sve chunkove gde se pominje
3. Filter treba da radi za **beskonačno mnogo kombinacija ulaznog teksta** — imena, teme, pojmovi.

---

## 3. Dizajn od nule: 2 kolekcije umesto 9

### Kolekcija 1: `Document` (~46K objekata)

Jedan zapis po dokumentu. Izvor istine za metapodatke. Bez vektorskog pretraživanja.

```
Document
├── document_hash        (TEXT, PK)
├── name                 (TEXT)
├── link                 (TEXT)
├── slug                 (TEXT)
├── author               (TEXT)          ← NOVO
├── post_type            (TEXT)
├── site                 (TEXT)
├── content_type         (TEXT: html/pdf)
├── language             (TEXT)
├── date                 (DATE)
├── updated_at           (DATE)
├── ingested_at          (DATE)
├── summary              (TEXT)
├── gist                 (TEXT)
├── full_text            (TEXT)
├── is_empty             (BOOL)
├── video_id             (TEXT)
├── pdf_filename         (TEXT)
└── wp_post_id           (TEXT)
```

### Kolekcija 2: `Chunk` (~4.2M objekata)

Spojena DiploChunk_contextual + DiploParagraph_contextual + DiploTurn_contextual.

```
Chunk
│
│── Vektor: jedan contextual vektor (precomputed, Vectorizer.none)
│           heading hijerarhija već enkodovana u vektor
│
├── ── Identifikacija ──
├── chunk_type           (TEXT: "sentence" | "paragraph" | "turn")
├── text                 (TEXT)          ← unificirano polje za sav sadržaj
├── context              (TEXT)
├── chunk_id             (TEXT)
│
├── ── Hijerarhija ──
├── document_hash        (TEXT, FK → Document)
├── parent_chunk_id      (TEXT)          ← sentence → paragraph veza
├── chunk_index          (INT)
│
├── ── Heading info ──
├── h1 .. h6             (TEXT)
├── section              (TEXT)
├── last_h_title         (TEXT)
├── strong_heading       (TEXT)
│
├── ── Denormalizovano za filtriranje ──
├── link                 (TEXT)
├── post_type            (TEXT)
├── site                 (TEXT)
├── date                 (DATE)
├── content_type         (TEXT)
│
├── ── Entiteti (za metadata filter) ──
├── author               (TEXT)          ← denormalizovano sa Document-a
├── speaker              (TEXT)          ← za transkripte
├── persons              (TEXT[])        ← sva pomenuta imena (author + speaker + NER)
├── organizations        (TEXT[])        ← sve organizacije
│
├── ── Turn-specifično ──
├── turn_id              (TEXT)
├── turn_index           (INT)
├── event_type           (TEXT)
├── location             (TEXT)
│
├── ── Sentence-specifično ──
├── sentence_index       (INT)
├── sentence_in_paragraph (INT)
│
├── ── Tehničko ──
├── provision_id         (TEXT)
├── chunker_version      (TEXT)
└── chunker_timestamp    (DATE)
```

### Šta ovo daje?

- **9 kolekcija → 2** (Document + Chunk)
- DiploHeading — obrisano iz Weaviate-a (mart 2026)
- Non-contextual kopije — obrisane
- Chunk + Paragraph + Turn — spojeni, razlikuju se po `chunk_type`
- Retriever logika se pojednostavljuje: jedna kolekcija, filtriraš po `chunk_type`

---

## 4. Analiza "pametnog filtera"

### Problem

Joca želi jedno polje gde korisnik ukuca bilo šta i sistem "pametno" filtrira.
Ali "Jovan Kurbalija" i "covid" su **fundamentalno različite operacije**:

| Tip inputa | Primer | Operacija |
|---|---|---|
| **Metadata filter** | "Jovan Kurbalija", "blog", "2024", "diplomacy.edu" | Weaviate property filter — egzaktan match na indeksiranom polju |
| **Content search** | "covid", "digital governance", "trade agreements" | Hybrid/vector search na tekstu — ono što sistem već radi |

- "Jovan Kurbalija" kao filter → korisnik želi sadržaj **OD** Kurbalije (author/speaker)
- "covid" kao filter → korisnik želi sadržaj **O** covidu (tema)

Prvo je metadata filtriranje, drugo je pretraga sadržaja. Ovo su različite operacije.

### Opcija A: LLM parsira input (najfleksibilnije, najskuplje)

Korisnik ukuca slobodan tekst, LLM interpretira intent i generiše Weaviate filtere.

```
Input: "covid Kurbalija 2020"
         ↓
LLM parsira:
{
  "search_query": "covid",
  "filters": {
    "persons": ["Jovan Kurbalija"],
    "date_year": 2020
  }
}
         ↓
Weaviate hybrid search za "covid"
+ Filter.by_property("persons").contains_any(["Jovan Kurbalija"])
+ Filter.by_property("date").greater_than("2020-01-01")
```

**Pros:** radi sa bilo čim, razume kontekst, "magično" polje  
**Cons:** dodatan LLM poziv (~500ms-1s), košta, može pogrešiti

### Opcija B: Structured UI (najrobustnije)

Frontend ima odvojena polja — search box za slobodni tekst + strukturirani filteri.

```
[Search: covid          ]    ← hybrid search na tekstu
[Person: Kurbalija      ]    ← filter po persons[]
[Type:   blog ▼         ]    ← filter po post_type
[Site:   diplomacy.edu ▼]    ← filter po site
[Date:   2020-01 → 2020-12]  ← filter po date range
```

**Pros:** korisnik eksplicitno kaže šta želi, nema magije koja može pogrešiti  
**Cons:** nije "jedno polje za sve"

### Opcija C: Hybrid pristup — search + faceted filtering (preporučeno)

Sve što korisnik ukuca ide kao hybrid search query. Dodatno, korisnik može opciono dodati strukturirane filtere (post_type, site, date range).

```
[Search: covid kurbalija ]    ← hybrid search (BM25 + vector)
[+ Filters ▼]                 ← opcioni strukturirani filteri
   Type: blog
   Site: diplomacy.edu
   Date: 2020
```

"Kurbalija" u hybrid search-u **će** pronaći chunkove gde se pominje — BM25 komponenta hvata keyword match u tekstu, heading naslovima, speaker polju.

**Pros:**
- "covid" u hybrid search već radi
- "Kurbalija" u hybrid search već radi (BM25 ga nađe)
- Strukturirani filteri za metadata su laki za implementaciju
- Nema LLM overhead-a
- Standard pattern (Elasticsearch, Algolia, Solr svi rade ovako)

**Cons:**
- Hybrid search za "Kurbalija" nađe sve gde se pominje, ne samo gde je autor
  (ali `persons[]` array se može dodati kao poboljšanje)

---

## 5. Preporuka — fazni pristup

### Faza 1: Brzi win (bez merge-a kolekcija)

Dodati strukturirane filtere na postojeću arhitekturu:
- Dodati `author` property na chunkove u chunking API-ju
- Implementirati API endpoint koji prima structured filters (post_type, site, date range, author)
- Frontend dobija filter UI

**Effort:** Mali  
**Vrednost:** Pokriva 80% Jocinog zahteva

### Faza 2: Merge kolekcija (opcionalno, kada bude vremena)

- Kreirati unified `Chunk` kolekciju sa svim chunk tipovima
- Migrirati podatke
- Prebaciti retrievere na jednu kolekciju
- Dodati `persons[]` array sa entity extraction

**Effort:** Veliki (chunking API + svi retrieveri + migracija)  
**Vrednost:** Čistija arhitektura, bolji entity-based filtering

### Faza 3: LLM filter parsing (opcionalno)

- Dodati LLM sloj koji parsira free-text filter input u strukturirane Weaviate filtere
- Ovo je "magično jedno polje" ali sa cenom LLM poziva

**Effort:** Srednji  
**Vrednost:** UX poboljšanje, ali Opcija C bez LLM-a pokriva većinu slučajeva

---

## 6. Ključni fajlovi u codebase-u

| Fajl | Relevantnost |
|---|---|
| `chatbot-api/app/core/config.py` | Konfiguracija kolekcija, retrieval parametri |
| `chatbot-api/app/ai/ai_services/retrievers/sentence_first.py` | Glavni retriever (sentence mode) |
| `chatbot-api/app/ai/ai_services/retrievers/twophase.py` | TwoPhase retriever |
| `chatbot-api/app/ai/ai_services/retrievers/combined.py` | Combined retriever |
| `chatbot-api/app/ai/ai_services/retrievers/utils.py` | Shared utils (`fetch_heading_boost` uklonjen) |
| `chatbot-api/app/services/ingest_service.py` | DiploHeading populacija uklonjena |
| `chatbot-api/app/schemas/chat_schema.py` | RetrievalConfig schema |
| `.env` | Dev konfiguracija (`USE_CONTEXTUAL_COLLECTIONS = true`) |

---

## 7. Zaključak

Jocin zahtev za spajanje kolekcija **ima smisla** kao arhitekturno čišćenje — od 9 kolekcija na 4–5 aktivnih contextual. DiploHeading je uklonjen.

"Pametan filter" za **metadata** (autor, govornik, tip sadržaja, datum, sajt) je izvodljiv kroz structured filtering + opcioni `persons[]` array.

"Pametan filter" za **sadržaj** (covid, digital governance, trade agreements) je bukvalno ono što hybrid search već radi — ne treba ništa novo, samo hybrid search query.

Kombinacija ova dva pristupa (search + faceted filters, **Opcija C**) je standardan, robustan pattern koji ne zahteva LLM overhead i pokriva realnu većinu use case-ova.
