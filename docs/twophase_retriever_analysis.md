 # TwoPhase Retriever — Analiza Pipeline-a

## Pregled

TwoPhase retriever je dvofazni retrieval sistem koji kombinuje brzo otkrivanje URL-ova (Phase 1) sa preciznom selekcijom sekcija (Phase 2), nakon čega sledi cross-encoder reranking i label-based boosting.

**Fajl**: `app/ai/ai_services/chatbot.py`, klasa `TwoPhaseRetriever` (linija ~1145)

**Konfiguracija** (`.env`):
- `RETRIEVAL_MODE = twophase`
- `PARAGRAPH_RETRIEVAL_K = 25` (broj URL-ova koji se prosleđuju Phase 2)
- `RERANKER_TOP_K = 25`
- `RETRIEVAL_CHUNKS = 10` (finalni broj dokumenata za LLM)
- `USE_RERANKER = True`

---

## Pipeline dijagram

```
Korisnikov upit
       │
       ▼
┌──────────────────┐
│ 1. Embed query   │  TEI embedder → vektor 1024 dim
└──────────────────┘
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
┌──────────────────────┐   ┌──────────────────────────┐
│ Phase 1a:            │   │ Phase 1b:                │
│ Hybrid search        │   │ Title filter             │
│ alpha=0.85           │   │ h1.like("*term*")        │
│ limit=200 paragrafa  │   │ 50 per term, max 3 terma │
│ 85% vector + 15% BM25│   │ score = 0.5 (neutral)    │
└──────────────────────┘   └──────────────────────────┘
       │                                 │
       └────────────┬────────────────────┘
                    ▼
       ┌──────────────────────┐
       │ Phase 1c:            │
       │ Merge u url_groups   │
       │ Aggregate scores     │
       │ Sort, top 25 URL-ova │
       └──────────────────────┘
                    │
                    ▼
       ┌──────────────────────┐
       │ Phase 2:             │
       │ Za svaki URL:        │
       │  - Fetch ALL paras   │
       │  - Group by section  │
       │  - Embed sec titles  │
       │  - Pick best section │
       │  - Join → dokument   │
       │  max 3500 chars      │
       └──────────────────────┘
                    │
                    ▼
       ┌──────────────────────┐
       │ Reranker:            │
       │ TEI cross-encoder    │
       │ prefix: [Label]      │
       │ H1 > Section Title   │
       │ 25 kandidata         │
       └──────────────────────┘
                    │
                    ▼
       ┌──────────────────────┐
       │ Label Weights:       │
       │ Topic=2.2x           │
       │ Course=2.0x          │
       │ Blog=1.8x            │
       │ Event=1.6x           │
       │ Updates=1.1x         │
       │ Top 10 → LLM        │
       └──────────────────────┘
```

---

## Detaljan opis svake faze

### 1. Embedding querija (linija 1177)

Query se embedduje putem TEI embeddera u vektor od 1024 dimenzija. Ovaj vektor se koristi i za pretragu i za kasniju selekciju sekcija.

### 2. Phase 1a — Hybrid Search (linije 1187-1234)

```python
collection.query.hybrid(query=query, vector=query_vector, alpha=0.85, limit=200)
```

- **Šta radi**: Traži 200 najrelevantnijih paragrafa iz `DiploParagraph` kolekcije
- **alpha=0.85**: 85% vektorska sličnost + 15% BM25 keyword matching
- **Zašto hybrid**: Vektor hvata semantičko značenje ("digital diplomacy" ≈ "online diplomatic practice"), a BM25 hvata retke termine (akronimi, proper nouns) koje embedding može da promaši
- **Rezultat**: Svaki paragraf se grupiše po URL-u u `url_groups` dict. Score se čuva po paragrafu.
- Tipičan output: **200 paragrafa → ~98 jedinstvenih URL-ova**

### 3. Phase 1b — Title Filter (linije 1240-1288)

```python
significant_terms = [t for t in q_terms if len(t) >= 4][:3]
# Primer: "what is digital diplomacy?" → ["what", "digital", "diplomacy"]

for term in significant_terms:
    collection.query.fetch_objects(
        filters=Filter.by_property("h1").like(f"*{term}*"),
        limit=50
    )
```

- **Šta radi**: Traži stranice čiji H1 naslov sadrži query termine
- **Zašto**: Čist hybrid search može da promaši stranice koje su nominalno o temi ali nemaju dovoljno jak vektorski signal (npr. kratki topici)
- **Score**: Fiksni 0.5 (neutralni) — dovoljno da uđu u razmatranje, ne previše da dominiraju
- **Poznati problem**: `fetch_objects` vraća u proizvoljnom redosledu. Za česte termine ("diplomacy") postoji 500+ paragrafa koji matchuju, a vraća se samo 50. Topici sa malo paragrafa (npr. Digital Diplomacy sa 5 paragrafa) se lako izgube.
- Tipičan output: **+150 paragrafa → ukupno ~194 URL-ova**

### 4. Phase 1c — Agregacija i selekcija URL-ova (linije 1293-1310)

Za svaki URL izračunava se **aggregate score**:

```python
scores = sorted([p["score"] for p in paragraphs], reverse=True)
weights = [1.0, 0.5, 0.25, 0.15, 0.1]
aggregate_score = sum(score * weight for score, weight in zip(scores, weights))
```

- Uzima top 5 paragraf-skorova po URL-u i pomnoži sa opadajućim težinama
- URL sa jednim odličnim paragrafom (score 0.9) dobija: `0.9 × 1.0 = 0.9`
- URL sa 5 dobrih paragrafa (svaki 0.5) dobija: `0.5×1.0 + 0.5×0.5 + 0.5×0.25 + 0.5×0.15 + 0.5×0.1 = 1.0`
- Ovo favorizuje stranice sa **više relevantnih paragrafa** (širi pokrivanje teme)
- Sortira URL-ove po agregatu, uzima **top 25** (`k_urls = PARAGRAPH_RETRIEVAL_K`)

### 5. Phase 2 — Section Selection (linije 1318-1438)

Za svaki od 25 URL-ova:

**Korak 2a — Fetch svih paragrafa:**
```python
collection.query.fetch_objects(
    filters=Filter.by_property("link").equal(url),
    limit=100
)
```
Dovlači SVE paragrafe sa tog URL-a (ne samo one koji su pronađeni u Phase 1).

**Korak 2b — Grupisanje po sekciji:**
Paragrafi se grupišu po `section` propertiju i sortiraju po `paragraph_index` (originalni redosled na stranici).

**Korak 2c — Embedding similarity za izbor sekcije:**
```python
# Za svaku sekciju sa >= 200 chars teksta:
title = section_paragraphs[0].last_h_title  # ili prvih 150 chars teksta ako nema naslova
# Embedduj sve section titles odjednom
title_vectors = self.embeddings.embed_documents(titles)
# Izračunaj cosine similarity svakog title vektora sa query vektorom
# Izaberi sekciju sa najvećom sličnošću
```

Primer za DD topic — sekcije i njihova sličnost sa "what is digital diplomacy?":
- `"Digital diplomacy"` → visoka sličnost (direktan match)
- `"Cyber diplomacy"` → srednja sličnost
- `"Zoom diplomacy"` → niža sličnost

**Korak 2d — Spajanje paragrafa u dokument:**
```python
section_text = "\n\n".join(paragraph.text for paragraph in best_section_paragraphs)
if len(section_text) > 3500:
    section_text = section_text[:3500] + "..."
```

Rezultat: 25 dokumenata, svaki sadrži **celu sekciju** (spojeni paragrafi), ne samo jedan izolovani paragraf.

### 6. Reranker — TEI Cross-Encoder (linije 123-230)

Za svaki dokument pravi se structured text:
```
[Course] Digital Public Diplomacy online course > Digital diplomacy

[BEST MATCH]: Digital is considered to have a broader scope...

[FULL CONTEXT]: (cela sekcija do 3500 chars)
```

- **Cross-encoder**: Za razliku od bi-encodera (koji embedduju query i dokument nezavisno), cross-encoder procesira query+dokument zajedno. Preciznije ali sporije.
- Šalje svih 25 dokumenata, vraća score za svaki
- Prefiks sa label-om, H1 naslovom i section title-om daje rerankeru kontekst o vrsti i lokaciji sadržaja

### 7. Label Weights — Post-Reranker (linije 1661-1707)

```python
final_score = reranker_score × label_weight

# Primer za "general" profil:
# Topic    → 2.2x    (prioritet za definicije/objašnjenja)
# Course   → 2.0x    (edukativni sadržaj)
# Blog     → 1.8x    (analitički sadržaj)
# Event    → 1.6x    (konferencije, webinari)
# Updates  → 1.1x    (kratke vesti)
```

- Sortira po `final_score`
- Vraća **top 10** (`RETRIEVAL_CHUNKS`) LLM-u za generisanje odgovora

---

## Brojke za test query "what is digital diplomacy?"

| Faza | Ulaz | Izlaz |
|------|------|-------|
| Phase 1a (hybrid) | query + vector | 200 paragrafa → ~98 URL-ova |
| Phase 1b (title filter) | 3 termina × 50 | ~150 paragrafa → ~194 URL-ova ukupno |
| Phase 1c (aggregate) | 194 URL-ova | Top 25 URL-ova |
| Phase 2 (section selection) | 25 URL-ova × fetch all | 25 dokumenata (cele sekcije) |
| Reranker | 25 dokumenata | 25 sa cross-encoder scorom |
| Label weights | 25 scored | Top 10 → LLM |
| **Latencija** | | **~10-11 sekundi** |

---

## Poznati problemi

### 1. Digital Diplomacy topic se ne pojavljuje u rezultatima
- **Phase 1a**: DD topic je na poziciji ~61 u 1000 paragrafa. Sa limit=200, ne ulazi.
- **Phase 1b**: Termini "digital" i "diplomacy" imaju 500+ matcheva u h1. DD-ovih 5 paragrafa se izgube u uzorku od 50.
- **Potencijalno rešenje**: Dedicated topic query sa `post_type="topic"` filterom.

### 2. Phase 2 latencija
- Embedding section titles za svaki od 25 URL-ova pojedinačno (25 API poziva)
- **Potencijalno rešenje**: Batch embedding svih sekcija odjednom.

### 3. Title filter proizvoljni redosled
- `fetch_objects` ne garantuje redosled rezultata
- Za česte termine vraća proizvoljan podskup od 50 iz 500+ matcheva
- **Potencijalno rešenje**: Kombinovani filter (`h1.like + post_type = topic`) ili veći limit sa Python dedup.

---

## Poređenje sa Paragraph Retrieverom

| Aspekt | Paragraph | TwoPhase |
|--------|-----------|----------|
| Pretraga | Hybrid (alpha=0.6), 50+50=100 | Hybrid (alpha=0.85), 200 + title filter 150 |
| Kolekcije | 2 (oba DiploParagraph) | 1 (DiploParagraph) |
| Jedinica za reranker | Pojedinačni paragraf (kratak) | Cela sekcija (spojeni paragrafi, do 3500 chars) |
| Section selection | Nema | Da (embedding similarity sekcija) |
| Secondary fetch | Nema | Da (svi paragrafi po URL-u) |
| URL dedup | Nema (može isti URL višestruko) | Da (po URL-u od starta) |
| Kandidati za reranker | 8 | 25 |
| BM25 udeo | 40% | 15% |
| Label weights | Ne rade (nema label property) | Rade (post_type → label mapping) |
| Latencija | ~9s | ~10-11s |

---

*Dokument generisan: 2026-02-03*
*Poslednji test: "what is digital diplomacy?" — twophase mode*
