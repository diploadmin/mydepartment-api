# SentenceFirstRetriever — Optimizacije

**Status:** Istorijski dokument (2026-02). Optimizacije 1, 4, 6 su i dalje aktivne u `SentenceFirstRetriever`.

> **Mart 2026:** Optimizacije **2, 3, 5** (DiploHeading Step 1b/1c i heading boost) su **uklonjene**. Kolekcija `DiploHeading` je obrisana; retrieval koristi samo `DiploChunk_contextual` sa ugrađenom heading hijerarhijom u vektorima.

**Benchmark rezultat (pre uklanjanja DiploHeading):** DD na #1, #2, #3 u top 8 za "What is digital diplomacy?"

---

## 1. Section-level grouping ✅

**Status:** Integrisano

**Problem:** Stari kod grupiše rečenice po `parent_paragraph_id`. Topic stranice imaju mnogo kratkih paragrafa, pa se relevantne rečenice rasipaju po mnogo sitnih grupa sa niskim agregatnim skorom.

**Rešenje:** Grupisano po `(url, section)` umesto `parent_paragraph_id`. Sve rečenice iz iste sekcije iste stranice se kombinuju, dajući topic stranicama fer šansu.

---

## 2. DiploHeading near_vector search (Step 1b) — uklonjeno

**Status:** Uklonjeno mart 2026. Zamenjeno contextual embedding-ima (`build_chunk_contextual` u chunking API).

---

## 3. Injection (Step 1c) — uklonjeno

**Status:** Uklonjeno mart 2026 (zavisilo od DiploHeading boost mape).

---

## 4. Top-N per section cap ✅

**Status:** Integrisano

**Problem:** Sekcija sa 40 rečenica dobija `count_bonus = log(1+40) = 3.74`, dok sekcija sa 5 odličnih rečenica dobija samo `log(1+5) = 1.79`. Kvantitet dominira nad kvalitetom.

**Rešenje:** Pre agregacije, sortiraju se rečenice po skoru i koriste samo top N (5) za formulu. Sve sekcije sa 5+ rečenica se takmice na ravnoj nozi.

**Konfig:** `TOP_N_PER_SECTION=5`

---

## 5. Heading boost na aggregate score (Step 3) — uklonjeno

**Status:** Uklonjeno mart 2026 (`HEADING_BOOST_WEIGHT` više nema efekta).

---

## 6. Weaviate-level blocklist filter ✅

**Status:** Integrisano (2026-02-14)

**Problem:** Boilerplate rečenica "Would you like to learn more about AI, tech and digital diplomacy?" postoji u ~643 kopija u bazi. Sa Python post-filtriranjem, morao se koristiti over-fetch (limit=1000) da bi se posle filtriranja dobilo 200 čistih rečenica. Over-fetch usporavao hybrid search na ~2.4s.

**Rešenje:** `build_weaviate_blocklist_filter()` u `app/core/blocklist.py` generiše Weaviate `not_equal` filter koji isključuje boilerplate na DB nivou. Limit ostaje na 200, Weaviate vraća samo čiste rečenice.

**Ušteda:** Hybrid search sa ~2.4s na ~1.5s (warm).

---

## Konfigurabilni parametri (svi u .env)

| Parametar | Vrednost | Opis |
|---|---|---|
| `HYBRID_ALPHA` | 0.6 | Vector/BM25 balans (0=BM25, 1=vector) |
| `HEADING_BOOST_WEIGHT` | 0.5 | *Neaktivno* (legacy) |
| `MAX_SECTIONS_PER_URL` | 3 | Max sekcija po URL-u |
| `SENTENCE_RERANKER_TOP_K` | 50 | Kandidata za reranker |
| `TOP_N_PER_SECTION` | 5 | Max rečenica za scoring po sekciji |
| `HEADING_INJECT_*` | — | *Neaktivno* (legacy) |
| `USE_SENTENCE_BLOCKLIST` | True | Weaviate-level blocklist filter |

---

## Benchmark (produkcija, query: "What is digital diplomacy?")

| Metrika | Warm vrednost |
|---|---|
| DD u top 8 | **#1, #2, #3** |
| Unique URLs u top 8 | **6** |
| #1 rezultat | **Topic: Digital diplomacy** |
| SentenceRetriever | ~2.3s |
| Reranker | ~1.1s |
| LLM | ~5s |
| **End-to-end** | **~9-10s** |

---

## Relevantni fajlovi

1. `app/ai/ai_services/chatbot.py` — `SentenceFirstRetriever._get_relevant_documents()`
2. `app/core/config.py` — svi parametri
3. `app/core/blocklist.py` — blocklist loading, Weaviate filter builder
4. `config/sentence_blocklist.txt` — blocklist pattern-i
5. `.env` — vrednosti parametara
