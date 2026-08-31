"""
TwoPhaseRetriever — document discovery + parallel section selection.

Phase 1 (fast):
  a) Hybrid search on Paragraphs
  c) Aggregate scores per document → top k_urls

Phase 2 (parallel + batched):
  For each top document: parallel fetch paragraphs, group by heading_path,
  batch-embed titles, select best section, build documents.
"""

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

import weaviate
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from weaviate.classes.query import HybridFusion

from app.core.collections import PARAGRAPH
from app.core.config import HYBRID_ALPHA
from app.core.weaviate_props import (
    TEXT,
    doc_url,
    doc_title,
    doc_date,
    section_key,
    source_type,
    label_from_props,
    normalize_props,
    heading_leaf,
    group_key,
    PARENT_DOCUMENT_HASH,
)
from app.ai.ai_services.retrievers.utils import (
    clean_sentence_text,
    aggregate_scores,
    resolve_collection_name,
    build_base_retrieval_filter,
    fetch_paragraphs_for_group,
    MAX_SECTION_LEN,
)


class TwoPhaseRetriever(BaseRetriever):
    """
    Two-phase retriever for oneweaviate Paragraphs.
    """
    client: weaviate.WeaviateClient
    embeddings: Embeddings
    paragraph_index: str = PARAGRAPH
    k_urls: int = 30
    vector_limit: int = 200
    user_type: str = "general"

    model_config = {"arbitrary_types_allowed": True}

    def set_user_type(self, user_type: str):
        self.user_type = user_type.lower() if user_type else "general"
        print(f"DEBUG: TwoPhaseRetriever user_type set to: {self.user_type}", flush=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_vector = self.embeddings.embed_query(query)
        effective_para_index = resolve_collection_name(self.paragraph_index)
        collection = self.client.collections.get(effective_para_index)

        _combined_filter = build_base_retrieval_filter(self.client, effective_para_index)

        try:
            hybrid_kwargs = dict(
                query=query,
                vector=query_vector,
                alpha=HYBRID_ALPHA,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                limit=self.vector_limit,
                query_properties=[TEXT],
                return_metadata=["score"],
            )
            if _combined_filter is not None:
                hybrid_kwargs["filters"] = _combined_filter
            vector_results = collection.query.hybrid(**hybrid_kwargs)
        except Exception as e:
            print(f"WARNING: TwoPhase hybrid search failed: {e}", flush=True)
            vector_results = type("R", (), {"objects": []})()

        url_groups: Dict[str, dict] = {}
        for obj in vector_results.objects:
            props = normalize_props(obj.properties or {})
            text = props.get(TEXT) or ""
            if not text:
                continue
            score = obj.metadata.score if obj.metadata else 0.0
            gkey = group_key(props)
            url = doc_url(props)
            section = section_key(props)
            post_type = source_type(props)
            label = label_from_props(props)

            if gkey not in url_groups:
                url_groups[gkey] = {
                    "paragraphs": [],
                    "sections": {},
                    "props": props,
                    "post_type": post_type,
                    "label": label,
                    "source": "vector",
                    "url": url,
                    "group_key": gkey,
                }

            para_entry = {
                "text": text,
                "score": score,
                "h2": heading_leaf(section),
                "section": section,
                "last_h_title": heading_leaf(section) or section,
                "paragraph_index": props.get("paragraph_index") or 0,
            }
            url_groups[gkey]["paragraphs"].append(para_entry)
            if section not in url_groups[gkey]["sections"]:
                url_groups[gkey]["sections"][section] = []
            url_groups[gkey]["sections"][section].append(para_entry)

        print(
            f"DEBUG: TwoPhase P1a — vector search: {len(vector_results.objects)} paragraphs → "
            f"{len(url_groups)} unique docs",
            flush=True,
        )

        for gkey, group in url_groups.items():
            paras = group["paragraphs"]
            if paras:
                scores = sorted([p["score"] for p in paras], reverse=True)
                agg = aggregate_scores(scores)
                group["best_para"] = max(paras, key=lambda p: p["score"])
            else:
                agg = 0.0
                group["best_para"] = {
                    "text": "",
                    "score": 0,
                    "h2": "",
                    "section": "general",
                    "last_h_title": "",
                }

            group["aggregate_score"] = agg
            group["_agg_raw"] = agg
            group["_heading_boost"] = 0.0
            group["match_count"] = len(paras)

            best_section_name = "general"
            best_section_score = 0
            for sec_name, sec_paras in group["sections"].items():
                sec_score = sum(p["score"] for p in sec_paras)
                if sec_score > best_section_score:
                    best_section_score = sec_score
                    best_section_name = sec_name
            group["best_section"] = best_section_name

        sorted_groups = sorted(
            url_groups.items(), key=lambda x: x[1]["aggregate_score"], reverse=True
        )

        print("DEBUG: TwoPhase P1c — top docs:", flush=True)
        for i, (gkey, group) in enumerate(sorted_groups[:15], 1):
            raw = group.get("_agg_raw", group["aggregate_score"])
            hb = group.get("_heading_boost", 0.0)
            total = group["aggregate_score"]
            pt = group.get("post_type", "")
            label = group.get("url") or gkey
            print(
                f"DEBUG:   #{i:2d}: total={total:.4f} (agg={raw:.4f} + h_boost={hb:.4f}) "
                f"hits={group['match_count']:2d} {pt:12s} {label[:60]}",
                flush=True,
            )

        selected_groups = sorted_groups[: self.k_urls]
        t_p2_start = time.time()

        def _fetch_group_paras(item):
            gkey, group = item
            try:
                return gkey, fetch_paragraphs_for_group(self.client, group, limit=200)
            except Exception as e:
                print(
                    f"DEBUG: TwoPhase secondary fetch failed for {gkey[:60]}: {e}",
                    flush=True,
                )
                return gkey, None

        url_fetched = {}
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(_fetch_group_paras, item) for item in selected_groups
            ]
            for future in as_completed(futures):
                try:
                    gkey, paras = future.result()
                    url_fetched[gkey] = paras
                except Exception:
                    pass

        t_p2_fetch = time.time()
        print(
            f"DEBUG: TwoPhase P2 fetch: {t_p2_fetch - t_p2_start:.2f}s "
            f"({len(url_fetched)} docs)",
            flush=True,
        )

        url_data = {}
        all_titles = []
        title_index = []
        embed_needed_urls = 0

        for gkey, group in selected_groups:
            fetched_paras = url_fetched.get(gkey)
            if fetched_paras is None:
                fetched_paras = [
                    {
                        "text": p["text"],
                        "h2": p.get("h2", ""),
                        "section": p.get("section", "general"),
                        "paragraph_index": p.get("paragraph_index", 0),
                        "last_h_title": p.get("last_h_title", ""),
                    }
                    for p in group["paragraphs"]
                ]

            section_paras = defaultdict(list)
            for p in fetched_paras:
                section_paras[p.get("section") or "general"].append(p)
            for sec_name in section_paras:
                section_paras[sec_name].sort(key=lambda p: p.get("paragraph_index", 0))

            eligible = []
            for sec_name, sec_ps in section_paras.items():
                sec_text = "\n\n".join(p["text"] for p in sec_ps if p["text"])
                if len(sec_text) < 200:
                    continue
                title = (sec_ps[0].get("last_h_title") or heading_leaf(sec_name) or "").strip()
                if not title:
                    title = sec_text[:150]
                eligible.append((sec_name, title))

            if len(eligible) > 1:
                embed_needed_urls += 1
                for sec_name, title in eligible:
                    all_titles.append(title)
                    title_index.append((gkey, sec_name))

            url_data[gkey] = {
                "section_paras": dict(section_paras),
                "eligible": eligible,
                "fetched_count": len(fetched_paras),
                "group": group,
            }

        all_vectors = []
        if all_titles:
            try:
                EMBED_BATCH = 30
                for start in range(0, len(all_titles), EMBED_BATCH):
                    batch = all_titles[start : start + EMBED_BATCH]
                    batch_vecs = self.embeddings.embed_documents(batch)
                    all_vectors.extend(batch_vecs)
            except Exception as e:
                print(f"DEBUG: TwoPhase batch embed failed: {e}", flush=True)
                all_vectors = []

        t_p2_embed = time.time()
        n_batches = (len(all_titles) + 29) // 30 if all_titles else 0
        print(
            f"DEBUG: TwoPhase P2 section select: "
            f"{embed_needed_urls} docs need embedding "
            f"({len(all_titles)} titles in {n_batches} batches, "
            f"{t_p2_embed - t_p2_fetch:.2f}s)",
            flush=True,
        )

        url_sec_vectors = defaultdict(dict)
        for i, (gkey, sec_name) in enumerate(title_index):
            if i < len(all_vectors):
                url_sec_vectors[gkey][sec_name] = all_vectors[i]

        q_norm = sum(x * x for x in query_vector) ** 0.5

        documents = []
        for gkey, group in selected_groups:
            data = url_data.get(gkey)
            if not data:
                continue
            section_paras = data["section_paras"]
            eligible = data["eligible"]
            fetched_count = data["fetched_count"]
            props = group["props"]
            best_section = group["best_section"]
            url = group.get("url") or doc_url(props)

            target_section = best_section
            target_paras = section_paras.get(target_section, [])
            section_text = "\n\n".join(p["text"] for p in target_paras if p["text"])

            sec_vectors = url_sec_vectors.get(gkey, {})
            if len(sec_vectors) > 1:
                best_sim = -1.0
                best_sec_name = target_section
                for sec_name, tv in sec_vectors.items():
                    dot = sum(a * b for a, b in zip(query_vector, tv))
                    t_norm = sum(x * x for x in tv) ** 0.5
                    sim = dot / (q_norm * t_norm) if q_norm and t_norm else 0.0
                    if sim > best_sim:
                        best_sim = sim
                        best_sec_name = sec_name
                if best_sec_name != target_section:
                    target_section = best_sec_name
                    target_paras = section_paras.get(target_section, [])
                    section_text = "\n\n".join(
                        p["text"] for p in target_paras if p["text"]
                    )
            elif len(eligible) == 1:
                only_sec = eligible[0][0]
                if only_sec != target_section:
                    target_section = only_sec
                    target_paras = section_paras.get(target_section, [])
                    section_text = "\n\n".join(
                        p["text"] for p in target_paras if p["text"]
                    )

            if len(section_text) > MAX_SECTION_LEN:
                section_text = section_text[:MAX_SECTION_LEN] + "..."

            best_para_text = group["best_para"].get("text", "") if group["best_para"] else ""
            content = section_text if section_text else best_para_text
            if not content:
                continue
            best = target_paras[0] if target_paras else group["best_para"]

            target_section_title = ""
            for tp in target_paras:
                lht = tp.get("last_h_title", "") or heading_leaf(tp.get("section", ""))
                if lht:
                    target_section_title = lht
                    break

            from app.services.doc_title_service import display_title

            title = display_title(props)
            metadata = {
                "title": title,
                "url": url,
                "date": doc_date(props),
                "label": group["label"],
                "post_type": group["post_type"],
                "h1": title,
                "h2": best.get("h2", ""),
                "h3": "",
                "section": target_section,
                "strong_heading": "",
                "parent_document_hash": props.get(PARENT_DOCUMENT_HASH) or "",
                "_index": effective_para_index,
                "_best_sentence": clean_sentence_text(
                    (group["best_para"].get("text", "") or "")[:200]
                ),
                "_best_sentence_score": (
                    group["best_para"].get("score", 0) if group["best_para"] else 0
                ),
                "_aggregate_score": group["aggregate_score"],
                "_matched_paragraphs": group["match_count"],
                "_fetched_paragraphs": fetched_count,
                "_section": target_section,
                "_section_title": target_section_title,
                "_num_sections": len(section_paras),
                "_source": f"twophase_{group.get('source', 'vector')}",
                "_label_weight": 1.0,
                "_final_score": group["aggregate_score"],
            }

            documents.append(Document(page_content=content, metadata=metadata))

        t_p2_end = time.time()
        print(
            f"DEBUG: TwoPhase P2 build: {t_p2_end - t_p2_embed:.2f}s ({len(documents)} docs)",
            flush=True,
        )
        print(f"DEBUG: TwoPhase P2 total: {t_p2_end - t_p2_start:.2f}s", flush=True)
        print(
            f"DEBUG: TwoPhase — {len(url_groups)} docs → top {len(documents)} documents",
            flush=True,
        )

        return documents
