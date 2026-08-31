"""
SentenceFirstRetriever — sentence-level hybrid search with section grouping.

Pipeline (oneweaviate):
1a. Hybrid search on Sentences.text
2.  Group sentences by (parent_document_hash, heading_path)
3.  Aggregate top-N scores
4.  Select best sentence per group (cross-encoder or fallback)
5.  Cap + dedup → return top K sections (page_content from Paragraphs)
"""

import math
import time
from collections import defaultdict
from typing import List

import weaviate
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from weaviate.classes.query import HybridFusion

from app.core.collections import CHUNK
from app.core.config import (
    HYBRID_ALPHA, USE_SENTENCE_BLOCKLIST,
    SENTENCE_CE_CANDIDATES, SENTENCE_CE_MAX_GROUPS, SENTENCE_HIGHLIGHT_MODE,
    TOP_N_PER_SECTION, MAX_SECTIONS_PER_URL,
    USE_RERANKER, RERANKER_URL, RERANKER_API_KEY, RERANKER_TOP_K,
)
from app.core.retrieval_context import get_param
from app.core.blocklist import build_weaviate_blocklist_filter
from app.core.weaviate_props import (
    TEXT,
    DOCUMENT_TITLE,
    HEADING_PATH,
    PARENT_DOCUMENT_HASH,
    PARENT_PARAGRAPH_ID,
    doc_url,
    doc_title,
    doc_date,
    section_key,
    source_type,
    label_from_props,
    sentence_text,
    normalize_props,
    heading_leaf,
)
from app.ai.ai_services.label_weights import get_label_weights
from app.ai.ai_services.reranker import TEIReranker
from app.ai.ai_services.retrievers.utils import (
    clean_sentence_text,
    resolve_collection_name,
    combine_filters,
    build_base_retrieval_filter,
    fetch_paragraph_texts_by_ids,
)

_tei_reranker = None


def _get_sentence_reranker() -> TEIReranker | None:
    global _tei_reranker
    if _tei_reranker is None and USE_RERANKER and RERANKER_URL and RERANKER_API_KEY:
        _tei_reranker = TEIReranker(url=RERANKER_URL, api_key=RERANKER_API_KEY, top_k=RERANKER_TOP_K)
    return _tei_reranker


class SentenceFirstRetriever(BaseRetriever):
    """
    Sentence-First Retriever for oneweaviate Sentences collection.
    """
    client: weaviate.WeaviateClient
    embeddings: Embeddings
    index_name: str = CHUNK
    k_sentences: int = 200
    final_k: int = 8
    alpha: float = HYBRID_ALPHA
    use_title_hybrid: bool = False
    k_titles: int = 20
    title_alpha: float = 0.3
    title_fetch_limit: int = 60
    user_type: str = "general"
    max_weight: float = 0.6
    avg_weight: float = 0.3
    count_weight: float = 0.1
    _debug_data: dict | None = None
    last_query_vector: list[float] | None = None

    model_config = {"arbitrary_types_allowed": True}

    def set_user_type(self, user_type: str):
        self.user_type = user_type.lower() if user_type else "general"
        print(f"DEBUG: SentenceFirstRetriever user_type set to: {self.user_type}", flush=True)

    def run_debug(self, query: str) -> tuple:
        """Run retrieval with debug data collection. Returns (documents, debug_data)."""
        self._debug_data = {"timing": {}}
        try:
            docs = self.invoke(query)
            debug_data = self._debug_data
            return docs, debug_data
        finally:
            self._debug_data = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        t_pipeline = time.time()
        t0 = time.time()
        query_vector = self.embeddings.embed_query(query)
        self.last_query_vector = query_vector
        t_embed = time.time() - t0

        get_label_weights(self.user_type)
        print(f"DEBUG: SentenceFirstRetriever using profile '{self.user_type}'", flush=True)

        effective_index = resolve_collection_name(self.index_name)
        collection = self.client.collections.get(effective_index)

        base_filter = build_base_retrieval_filter(self.client, effective_index)
        if USE_SENTENCE_BLOCKLIST:
            bl_filter = build_weaviate_blocklist_filter(text_property=TEXT)
            if bl_filter is not None:
                base_filter = combine_filters(base_filter, bl_filter)

        _alpha = get_param("hybrid_alpha", self.alpha)
        _k_sents = get_param("sentence_retrieval_k", self.k_sentences)
        t0 = time.time()

        hybrid_kwargs = dict(
            query=query,
            vector=query_vector,
            alpha=_alpha,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            limit=_k_sents,
            query_properties=[TEXT],
            return_metadata=["score"],
        )
        if base_filter is not None:
            hybrid_kwargs["filters"] = base_filter

        results = collection.query.hybrid(**hybrid_kwargs)
        sentence_objects = list(results.objects)
        t_hybrid = time.time() - t0

        title_objects: list = []
        n_title_unique = 0
        n_overlap = 0
        t_title = 0.0
        if self.use_title_hybrid:
            _title_alpha = get_param("sentence_title_alpha", self.title_alpha)
            _title_k = get_param("sentence_title_k", self.k_titles)
            _title_fetch = get_param("sentence_title_fetch_limit", self.title_fetch_limit)
            t_t0 = time.time()
            title_kwargs = dict(
                query=query,
                vector=query_vector,
                alpha=_title_alpha,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                limit=_title_fetch,
                query_properties=[DOCUMENT_TITLE],
                return_metadata=["score"],
            )
            if base_filter is not None:
                title_kwargs["filters"] = base_filter
            title_results = collection.query.hybrid(**title_kwargs)
            t_title = time.time() - t_t0

            # Dedupe by document hash / url
            best_by_doc: dict = {}
            for obj in title_results.objects:
                props = obj.properties or {}
                key = (
                    props.get(PARENT_DOCUMENT_HASH)
                    or doc_url(props)
                    or doc_title(props)
                    or str(getattr(obj, "uuid", id(obj)))
                )
                score = obj.metadata.score if obj.metadata else 0.0
                prev = best_by_doc.get(key)
                if prev is None or score > (prev.metadata.score if prev.metadata else 0.0):
                    best_by_doc[key] = obj
            ranked = sorted(
                best_by_doc.items(),
                key=lambda kv: (kv[1].metadata.score if kv[1].metadata else 0.0),
                reverse=True,
            )[: int(_title_k)]
            title_objects = [obj for _, obj in ranked]
            n_title_unique = len(title_objects)

            seen_keys = set()
            for obj in sentence_objects:
                seen_keys.add(str(getattr(obj, "uuid", None) or id(obj)))
            merged = list(sentence_objects)
            for obj in title_objects:
                key = str(getattr(obj, "uuid", None) or id(obj))
                if key in seen_keys:
                    n_overlap += 1
                    continue
                seen_keys.add(key)
                merged.append(obj)
            sentence_objects = merged

        print(
            f"DEBUG: Retrieved {len(results.objects)} sentences from {effective_index} "
            f"(alpha={_alpha}, k={_k_sents}, blocklist={'WEAVIATE' if USE_SENTENCE_BLOCKLIST else 'OFF'})",
            flush=True,
        )
        if self.use_title_hybrid:
            print(
                f"DEBUG: Title hybrid unique_docs={n_title_unique} fetch_hits={len(title_objects)} "
                f"overlap={n_overlap} merged={len(sentence_objects)} "
                f"(title_alpha={get_param('sentence_title_alpha', self.title_alpha)}, "
                f"title_k={get_param('sentence_title_k', self.k_titles)})",
                flush=True,
            )

        if self._debug_data is not None:
            self._debug_data["timing"]["embed"] = round(t_embed, 4)
            self._debug_data["timing"]["hybrid"] = round(t_hybrid, 4)
            if self.use_title_hybrid:
                self._debug_data["timing"]["hybrid_title"] = round(t_title, 4)
                self._debug_data["counts_title"] = {
                    "title_unique": n_title_unique,
                    "sentence_raw": len(results.objects),
                    "overlap": n_overlap,
                    "merged": len(sentence_objects),
                }
            self._debug_data["step_1a_hybrid_sentences"] = [
                {
                    "sentence": sentence_text(obj.properties or {}),
                    "url": doc_url(obj.properties or {}),
                    "section": section_key(obj.properties or {}),
                    "score": round(obj.metadata.score, 6) if obj.metadata else 0.0,
                    "h1": doc_title(obj.properties or {}),
                    "post_type": source_type(obj.properties or {}),
                    "date": doc_date(obj.properties or {}),
                }
                for obj in sentence_objects
            ]

        if not sentence_objects:
            return []

        # STEP 2: Group by (parent_document_hash, heading_path)
        section_groups = {}
        for obj in sentence_objects:
            props = normalize_props(obj.properties or {})
            score = obj.metadata.score if obj.metadata else 0.0

            doc_hash = (props.get(PARENT_DOCUMENT_HASH) or "").strip()
            section = section_key(props)
            url = doc_url(props)
            if not doc_hash and not url:
                # Still allow title-only grouping as last resort
                title = doc_title(props)
                if not title:
                    continue
                group_key = (f"title:{title}", section)
            else:
                group_key = (doc_hash or f"url:{url}", section)

            if group_key not in section_groups:
                st = source_type(props)
                label = label_from_props(props)
                section_groups[group_key] = {
                    "section_context": "",
                    "context_parts": set(),
                    "paragraph_ids": set(),
                    "sentences": [],
                    "metadata": {
                        "title": doc_title(props),
                        "url": url,
                        "date": doc_date(props),
                        "label": label,
                        "post_type": st,
                        "h1": doc_title(props),
                        "h2": heading_leaf(section),
                        "h3": "",
                        "section": section,
                        "parent_document_hash": doc_hash,
                        "_index": effective_index,
                    },
                }

            para_id = (props.get(PARENT_PARAGRAPH_ID) or "").strip()
            if para_id:
                section_groups[group_key]["paragraph_ids"].add(para_id)

            sent = sentence_text(props)
            section_groups[group_key]["sentences"].append({
                "text": sent,
                "score": score,
                "sentence_index": props.get("sentence_index", 0),
                "parent_paragraph_id": para_id,
            })

        # Resolve paragraph context for page_content
        all_para_ids = []
        for group in section_groups.values():
            all_para_ids.extend(group["paragraph_ids"])
        para_texts = fetch_paragraph_texts_by_ids(self.client, all_para_ids)

        for key, group in section_groups.items():
            contexts = []
            for pid in group["paragraph_ids"]:
                text = para_texts.get(pid)
                if text:
                    contexts.append(text)
            if contexts:
                # Preserve order of first appearance of paragraph ids
                group["section_context"] = "\n\n".join(dict.fromkeys(contexts))
            else:
                # Fallback: joined unique sentence texts
                joined = "\n".join(
                    dict.fromkeys(s["text"] for s in group["sentences"] if s.get("text"))
                )
                group["section_context"] = joined
            del group["context_parts"]
            del group["paragraph_ids"]

        # STEP 3: Aggregate scores
        for group_key, group in section_groups.items():
            all_scores = sorted([s["score"] for s in group["sentences"]], reverse=True)
            total_sentences = len(all_scores)

            if not all_scores:
                group["aggregate_score"] = 0.0
                continue

            _tnps = get_param("top_n_per_section", TOP_N_PER_SECTION)
            scores = all_scores[:_tnps] if _tnps > 0 else all_scores
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            count_bonus = math.log(1 + len(scores))
            group["aggregate_score"] = (
                self.max_weight * max_score
                + self.avg_weight * avg_score
                + self.count_weight * count_bonus
            )

            group["metadata"]["_total_sentences"] = total_sentences
            group["metadata"]["_scored_sentences"] = len(scores)
            group["metadata"]["_heading_boost"] = 0.0
            group["metadata"]["_aggregate_score"] = group["aggregate_score"]
            group["metadata"]["_matched_sentences"] = [
                clean_sentence_text(s["text"]) for s in group["sentences"]
            ]

        if self._debug_data is not None:
            self._debug_data["step_2_3_section_groups"] = [
                {
                    "url": g["metadata"].get("url", ""),
                    "section": g["metadata"].get("section", ""),
                    "title": g["metadata"].get("title", ""),
                    "label": g["metadata"].get("label", ""),
                    "post_type": g["metadata"].get("post_type", ""),
                    "total_sentences": g["metadata"].get("_total_sentences", 0),
                    "scored_sentences": g["metadata"].get("_scored_sentences", 0),
                    "aggregate_score": round(g.get("aggregate_score", 0), 6),
                    "heading_boost": round(g["metadata"].get("_heading_boost", 0), 6),
                    "sentences": [
                        {"text": s["text"][:200], "score": round(s["score"], 6)}
                        for s in sorted(g["sentences"], key=lambda x: -x["score"])[:10]
                    ],
                }
                for gk, g in sorted(
                    section_groups.items(), key=lambda x: -x[1].get("aggregate_score", 0)
                )
            ]

        top_groups_debug = sorted(
            section_groups.items(), key=lambda x: -x[1]["aggregate_score"]
        )[:5]
        print("DEBUG: Top 5 groups by aggregate score:")
        for gk, g in top_groups_debug:
            url_short = (g["metadata"].get("url") or g["metadata"].get("title") or "N/A")[-60:]
            n_matched = len(g["metadata"]["_matched_sentences"])
            n_total = g["metadata"]["_total_sentences"]
            print(
                f"  [{g['aggregate_score']:.3f}] {url_short} | "
                f"{n_matched} matched (of {n_total} total)",
                flush=True,
            )

        _sh_mode = get_param("sentence_highlight_mode", SENTENCE_HIGHLIGHT_MODE)
        if _sh_mode == "single":
            self._select_best_sentence_single(query, section_groups)
        else:
            self._select_best_sentence_multi(section_groups)

        for group_key, group in section_groups.items():
            group["final_score"] = group["aggregate_score"]
            group["metadata"]["_label_weight"] = 1.0
            group["metadata"]["_final_score"] = group["final_score"]
            group["metadata"]["_user_type"] = self.user_type

        sorted_sections = sorted(
            section_groups.values(),
            key=lambda g: g["final_score"],
            reverse=True,
        )
        seen_content = set()
        url_section_count = defaultdict(int)
        unique_sections = []
        skipped_by_url_cap = 0
        _mspu = get_param("max_sections_per_url", MAX_SECTIONS_PER_URL)

        for section in sorted_sections:
            content = section["section_context"]
            content_hash = hash(content)
            if content_hash in seen_content or not content:
                continue
            # Cap by document hash when URL missing
            cap_key = (
                section["metadata"].get("url")
                or section["metadata"].get("parent_document_hash")
                or ""
            )
            if _mspu > 0 and cap_key and url_section_count[cap_key] >= _mspu:
                skipped_by_url_cap += 1
                continue
            seen_content.add(content_hash)
            if cap_key:
                url_section_count[cap_key] += 1
            unique_sections.append(section)

        if skipped_by_url_cap > 0:
            print(
                f"DEBUG: URL/doc cap ({_mspu}/doc) skipped {skipped_by_url_cap} sections, "
                f"{len(url_section_count)} unique docs in pool",
                flush=True,
            )

        documents = []
        for section in unique_sections[: self.final_k]:
            documents.append(
                Document(
                    page_content=section["section_context"],
                    metadata=section["metadata"],
                )
            )

        top_n_info = (
            f", top-{get_param('top_n_per_section', TOP_N_PER_SECTION)}/section cap"
            if get_param("top_n_per_section", TOP_N_PER_SECTION) > 0
            else ""
        )
        t_total_sentence = time.time() - t_pipeline
        print(
            f"DEBUG: {len(section_groups)} section groups → {len(unique_sections)} after dedup/URL-cap → "
            f"returning {len(documents)} sections "
            f"({len(set(d.metadata.get('url', '') or d.metadata.get('parent_document_hash', '') for d in documents))} unique docs"
            f"{top_n_info})",
            flush=True,
        )
        print(
            f"TIMING SentenceRetriever: embed={t_embed:.3f}s hybrid={t_hybrid:.3f}s "
            f"TOTAL={t_total_sentence:.3f}s",
            flush=True,
        )

        if self._debug_data is not None:
            self._debug_data["timing"]["total"] = round(t_total_sentence, 4)
            self._debug_data["step_5_pre_reranker"] = [
                {
                    "url": d.metadata.get("url", ""),
                    "title": d.metadata.get("title", ""),
                    "section": d.metadata.get("section", ""),
                    "label": d.metadata.get("label", ""),
                    "post_type": d.metadata.get("post_type", ""),
                    "final_score": round(d.metadata.get("_final_score", 0), 6),
                    "aggregate_score": round(d.metadata.get("_aggregate_score", 0), 6),
                    "heading_boost": round(d.metadata.get("_heading_boost", 0), 6),
                    "best_sentence": d.metadata.get("_best_sentence", ""),
                    "total_sentences": d.metadata.get("_total_sentences", 0),
                    "content_preview": d.page_content[:300],
                }
                for d in documents
            ]
            self._debug_data["counts"] = {
                "hybrid_sentences": len(self._debug_data.get("step_1a_hybrid_sentences", [])),
                "section_groups": len(section_groups),
                "after_dedup": len(unique_sections),
                "pre_reranker": len(documents),
            }

        return documents

    def _select_best_sentence_single(self, query: str, section_groups: dict) -> None:
        """Cross-encoder picks the best sentence per group."""
        CANDIDATES_PER_GROUP = get_param("sentence_ce_candidates", SENTENCE_CE_CANDIDATES)
        MAX_GROUPS_TO_RERANK = get_param("sentence_ce_max_groups", SENTENCE_CE_MAX_GROUPS)
        ranked_groups = sorted(
            [(gk, g) for gk, g in section_groups.items() if g.get("sentences")],
            key=lambda x: x[1].get("aggregate_score", 0),
            reverse=True,
        )
        top_group_keys = set(gk for gk, _ in ranked_groups[:MAX_GROUPS_TO_RERANK])

        rerank_batch = []
        for group_key, group in section_groups.items():
            if group_key not in top_group_keys:
                continue
            sorted_sents = sorted(group["sentences"], key=lambda s: s["score"], reverse=True)
            for sent in sorted_sents[:CANDIDATES_PER_GROUP]:
                rerank_batch.append({
                    "group_key": group_key,
                    "sentence": sent,
                    "batch_idx": len(rerank_batch),
                })

        reranker_used = False
        tei_reranker = _get_sentence_reranker()
        if tei_reranker and rerank_batch:
            texts = [item["sentence"]["text"] for item in rerank_batch]
            t_sr = time.time()
            rerank_results = tei_reranker.rerank_sentences(query, texts)
            t_sr_elapsed = time.time() - t_sr

            if rerank_results:
                reranker_used = True
                ce_scores = {r["index"]: r["score"] for r in rerank_results}

                group_best = {}
                for item in rerank_batch:
                    gk = item["group_key"]
                    ce = ce_scores.get(item["batch_idx"], -999)
                    if gk not in group_best or ce > group_best[gk][1]:
                        group_best[gk] = (item["sentence"], ce)

                for group_key, (best_sent, ce_score) in group_best.items():
                    group = section_groups[group_key]
                    group["metadata"]["_best_sentence"] = clean_sentence_text(best_sent["text"])
                    group["metadata"]["_best_sentence_score"] = best_sent["score"]
                    group["metadata"]["_best_sentence_ce_score"] = round(ce_score, 4)

                print(
                    f"DEBUG: Step 3b [single] — CE reranked {len(rerank_batch)} candidates "
                    f"across {len(top_group_keys)} groups in {t_sr_elapsed:.3f}s",
                    flush=True,
                )

        for group_key, group in section_groups.items():
            if "_best_sentence" in group["metadata"]:
                continue
            if not group["sentences"]:
                continue
            best_sentence = max(group["sentences"], key=lambda s: s["score"])
            group["metadata"]["_best_sentence"] = clean_sentence_text(best_sentence["text"])
            group["metadata"]["_best_sentence_score"] = best_sentence["score"]

        if not reranker_used and rerank_batch:
            print("DEBUG: Step 3b [single] — CE unavailable, hybrid fallback", flush=True)

    def _select_best_sentence_multi(self, section_groups: dict) -> None:
        """Multi mode: all sentences via _matched_sentences, set _best_sentence as fallback."""
        for group_key, group in section_groups.items():
            if not group["sentences"]:
                continue
            best_sentence = max(group["sentences"], key=lambda s: s["score"])
            group["metadata"]["_best_sentence"] = clean_sentence_text(best_sentence["text"])
            group["metadata"]["_best_sentence_score"] = best_sentence["score"]
        print(
            "DEBUG: Step 3b [multi] — All matched sentences collected, CE skipped",
            flush=True,
        )
