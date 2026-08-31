"""
CombinedRetriever — merges sentence-first + paragraph-level search results.

Pipeline (oneweaviate):
1. Run SentenceFirstRetriever (Sentences hybrid + aggregation)
2. Run paragraph-level hybrid search on Paragraphs
3. Select best section per document via embedding similarity
4. Merge & deduplicate by URL / document hash
"""

import re
import time
from collections import defaultdict
from typing import List

import weaviate
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from weaviate.classes.query import HybridFusion

from app.core.collections import PARAGRAPH
from app.core.config import (
    HYBRID_ALPHA,
    PARAGRAPH_FETCH_LIMIT,
    FORCE_INCLUDE_H1, FORCE_INCLUDE_SECTION,
    FORCE_INCLUDE_OVERLAP, FORCE_INCLUDE_MAX,
)
from app.core.weaviate_props import TEXT, doc_title
from app.ai.ai_services.retrievers.sentence_first import SentenceFirstRetriever
from app.ai.ai_services.retrievers.utils import (
    group_paragraphs_by_url,
    compute_url_aggregate_scores,
    select_best_section,
    build_section_content,
    build_paragraph_document,
    resolve_collection_name,
    build_base_retrieval_filter,
    fetch_paragraphs_for_group,
)


class CombinedRetriever(BaseRetriever):
    """
    Merges SentenceFirstRetriever results with parallel paragraph-level
    hybrid search for broader coverage. Deduplicates by URL / doc hash.
    """
    client: weaviate.WeaviateClient
    embeddings: Embeddings
    sentence_retriever: SentenceFirstRetriever
    paragraph_index: str = PARAGRAPH
    k_paragraphs: int = 20
    alpha: float = HYBRID_ALPHA
    user_type: str = "general"

    model_config = {"arbitrary_types_allowed": True}

    def set_user_type(self, user_type: str):
        self.user_type = user_type
        self.sentence_retriever.set_user_type(user_type)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        t_combined_start = time.time()

        t0 = time.time()
        sentence_docs = self.sentence_retriever._get_relevant_documents(
            query, run_manager=run_manager
        )
        t_sentence = time.time() - t0
        print(
            f"DEBUG: CombinedRetriever - sentence search returned {len(sentence_docs)} docs",
            flush=True,
        )

        t0 = time.time()
        query_vector = self.embeddings.embed_query(query)
        t_para_embed = time.time() - t0

        effective_para_index = resolve_collection_name(self.paragraph_index)
        collection = self.client.collections.get(effective_para_index)
        fetch_limit = max(self.k_paragraphs * 10, PARAGRAPH_FETCH_LIMIT)

        _combined_filter = build_base_retrieval_filter(self.client, effective_para_index)

        t0 = time.time()
        hybrid_kwargs = dict(
            query=query,
            vector=query_vector,
            alpha=self.alpha,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            limit=fetch_limit,
            query_properties=[TEXT],
            return_metadata=["score"],
        )
        if _combined_filter is not None:
            hybrid_kwargs["filters"] = _combined_filter
        para_results = collection.query.hybrid(**hybrid_kwargs)
        t_para_hybrid = time.time() - t0

        url_groups = group_paragraphs_by_url(para_results.objects)
        compute_url_aggregate_scores(url_groups)

        for group in url_groups.values():
            group["_index"] = effective_para_index

        # FORCE-INCLUDE on document_title / heading_path
        q_terms = set(re.sub(r"[^a-z0-9\s]", "", query.lower()).split())
        q_terms = {t for t in q_terms if len(t) >= 3}

        force_matched = {}
        for gkey, group in url_groups.items():
            best_overlap = 0.0
            match_reason = ""

            if FORCE_INCLUDE_H1 and q_terms:
                h1 = (doc_title(group["props"]) or "").lower()
                h1_terms = set(re.sub(r"[^a-z0-9\s]", "", h1).split())
                h1_terms = {t for t in h1_terms if len(t) >= 3}
                if h1_terms:
                    overlap = len(h1_terms & q_terms) / len(q_terms)
                    if overlap >= FORCE_INCLUDE_OVERLAP:
                        best_overlap = overlap
                        match_reason = "document_title"

            if FORCE_INCLUDE_SECTION and q_terms:
                for para in group["paragraphs"]:
                    lht = (para.get("last_h_title") or para.get("section") or "").lower()
                    lht_terms = set(re.sub(r"[^a-z0-9\s]", "", lht).split())
                    lht_terms = {t for t in lht_terms if len(t) >= 3}
                    if lht_terms:
                        overlap = len(lht_terms & q_terms) / len(q_terms)
                        if overlap >= FORCE_INCLUDE_OVERLAP and overlap > best_overlap:
                            best_overlap = overlap
                            match_reason = "heading_path"

            if best_overlap > 0:
                boost = 1.0 + best_overlap * 0.5
                group["aggregate_score"] *= boost
                group["_force_boost"] = round(boost, 2)
                group["_force_overlap"] = best_overlap
                group["_force_reason"] = match_reason
                force_matched[gkey] = {"reason": match_reason, "overlap": best_overlap}

        sorted_groups = sorted(
            url_groups.items(), key=lambda x: x[1]["aggregate_score"], reverse=True
        )

        top_k_keys = {k for k, _ in sorted_groups[: self.k_paragraphs]}
        forced = []
        for gkey in force_matched:
            if gkey not in top_k_keys:
                forced.append((gkey, url_groups[gkey]))
        forced.sort(
            key=lambda x: (
                force_matched[x[0]]["overlap"],
                x[1].get("aggregate_score", 0),
            ),
            reverse=True,
        )
        forced = forced[:FORCE_INCLUDE_MAX]

        if forced:
            trim_count = min(len(forced), len(sorted_groups))
            sorted_groups = sorted_groups[: self.k_paragraphs - trim_count]
            sorted_groups.extend(forced)
            sorted_groups = sorted(
                sorted_groups, key=lambda x: x[1]["aggregate_score"], reverse=True
            )
            print(
                f"DEBUG: force-included {len(forced)} docs: "
                + ", ".join(
                    f"{k[:40]}({force_matched[k]['reason']}="
                    f"{force_matched[k]['overlap']:.0%})"
                    for k, _ in forced
                ),
                flush=True,
            )

        t_secondary_start = time.time()
        paragraph_docs = []
        for gkey, group in sorted_groups[: self.k_paragraphs]:
            best_section = group["best_section"]
            display_url = group.get("url") or ""

            fetched_count = 0
            try:
                all_paras_for_url = fetch_paragraphs_for_group(
                    self.client, group, limit=100
                )
                fetched_count = len(all_paras_for_url)
            except Exception as e:
                print(f"DEBUG: Secondary fetch failed for {gkey[:60]}: {e}", flush=True)
                all_paras_for_url = [
                    {
                        "text": p["text"],
                        "h2": p.get("h2", ""),
                        "section": p.get("section", "general"),
                        "paragraph_index": p.get("paragraph_index", 0),
                        "last_h_title": p.get("last_h_title", ""),
                    }
                    for p in group["paragraphs"]
                ]
                fetched_count = len(all_paras_for_url)

            section_paras = defaultdict(list)
            for p in all_paras_for_url:
                section_paras[p.get("section") or "general"].append(p)

            target_section = select_best_section(
                section_paras, query_vector, self.embeddings, default_section=best_section
            )
            section_text, target_paras = build_section_content(section_paras, target_section)

            doc = build_paragraph_document(
                display_url,
                group,
                section_text,
                target_section,
                target_paras,
                fetched_count,
                source="paragraph_search",
            )
            paragraph_docs.append(doc)

        print(
            f"DEBUG: CombinedRetriever - paragraph search: {len(para_results.objects)} raw → "
            f"{len(url_groups)} unique docs "
            f"({sum(len(g['sections']) for g in url_groups.values())} sections) → "
            f"top {len(paragraph_docs)} aggregated docs",
            flush=True,
        )

        for doc in paragraph_docs[:5]:
            m = doc.metadata
            force_str = (
                f" force:{m.get('_force_reason', '')}x{m.get('_force_boost', 1.0):.1f}"
                if m.get("_force_boost", 1.0) > 1.0
                else ""
            )
            print(
                f"DEBUG:   [{m.get('_aggregate_score', 0):.4f}] "
                f"({m.get('_matched_paragraphs', 1)} matched, {m.get('_num_sections', '?')} secs, "
                f"content={len(doc.page_content)}ch){force_str} "
                f"{m.get('label', '')} | {m.get('title', '')[:40]} | "
                f"{(m.get('url') or m.get('parent_document_hash') or '')[:45]} | "
                f"sec={m.get('_section', '')[:25]}",
                flush=True,
            )

        seen_keys = set()
        merged = []

        def _doc_key(doc: Document):
            url = doc.metadata.get("url", "")
            if url:
                return f"url:{url}"
            dh = doc.metadata.get("parent_document_hash", "")
            if dh:
                return f"hash:{dh}"
            return f"content:{hash(doc.page_content[:200])}"

        for doc in sentence_docs:
            key = _doc_key(doc)
            if key not in seen_keys:
                seen_keys.add(key)
                doc.metadata["_source"] = doc.metadata.get("_source", "sentence_search")
                merged.append(doc)

        para_added = 0
        for doc in paragraph_docs:
            key = _doc_key(doc)
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(doc)
                para_added += 1

        t_combined_total = time.time() - t_combined_start
        print(
            f"DEBUG: CombinedRetriever - merged {len(merged)} unique docs "
            f"({len(sentence_docs)} sentence + {para_added} new from paragraph)",
            flush=True,
        )
        print(
            f"TIMING CombinedRetriever: sentence={t_sentence:.3f}s "
            f"para_embed={t_para_embed:.3f}s para_hybrid={t_para_hybrid:.3f}s "
            f"secondary_fetch={time.time() - t_secondary_start:.3f}s "
            f"TOTAL={t_combined_total:.3f}s",
            flush=True,
        )

        return merged
