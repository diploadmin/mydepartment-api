"""
Debug retrieval endpoint — runs the full retrieval + reranking pipeline
and returns structured intermediate data from every step.
"""

import asyncio
import time
import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import (
    HYBRID_ALPHA, SENTENCE_RETRIEVAL_K, HEADING_BOOST_WEIGHT,
    HEADING_INJECT_MIN_SIM, HEADING_INJECT_MAX, HEADING_INJECT_MIN_SENTS,
    TOP_N_PER_SECTION, MAX_SECTIONS_PER_URL,
    SENTENCE_CE_CANDIDATES, SENTENCE_CE_MAX_GROUPS, SENTENCE_HIGHLIGHT_MODE,
    USE_RERANKER, RERANKER_TOP_K, SENTENCE_RERANKER_TOP_K,
    RETRIEVAL_CHUNKS, EMBEDDING_PROVIDER, EMBEDDING_MODEL_NAME,
    LOCAL_EMBEDDING_URL, LOCAL_EMBEDDING_KEY, OPENAI_KEY,
    USE_DYNAMIC_LABEL_WEIGHTS, DYNAMIC_WEIGHT_ALPHA, DYNAMIC_WEIGHT_MIN_SIM,
    USE_RECENCY_BOOST, RECENCY_BOOST_MAX_AGE_DAYS, RECENCY_HALF_LIFE_DAYS, RECENCY_MAX_BOOST,
)
from app.core.neo4j_config import (
    GRAPH_EXPANSION_ENABLED,
    GRAPH_CONTEXT_FORMAT,
    GRAPH_BOOST_ALPHA,
)
from app.core.retrieval_context import set_retrieval_overrides, clear_retrieval_overrides, get_param
from app.schemas.chat_schema import RetrievalConfig
from app.ai.ai_services.label_weights import DEFAULT_LABEL_WEIGHT, get_label_weights
from app.ai.ai_services.query_intent import get_dynamic_label_weights
from app.ai.ai_services.retrievers.recency import get_recency_multiplier_for_metadata
from app.ai.ai_services.retrievers.utils import (
    resolve_collection_name,
    resolve_parent_document_hash,
    get_content_filter_debug_info,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_cached_embeddings = None

# description: get the embeddings - this is used to get the embeddings for the query - either cached or new (local or openai)
def _get_embeddings():
    global _cached_embeddings
    if _cached_embeddings is None:
        if EMBEDDING_PROVIDER == 'local':
            from app.ai.ai_services.embeddings import TEIEmbeddings
            _cached_embeddings = TEIEmbeddings(url=LOCAL_EMBEDDING_URL, api_key=LOCAL_EMBEDDING_KEY)
        else:
            from langchain_openai import OpenAIEmbeddings
            _cached_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=OPENAI_KEY)
    return _cached_embeddings

# description: the request for the debug retrieve route - abstracted from the frontend
# retrieval_config is set up my admin panel - it regulates the retrieval process - available customization is in chat_schema.py
class DebugRetrieveRequest(BaseModel):
    query: str
    user_type: str = "general"
    retrieval_config: Optional[RetrievalConfig] = None

# description: the debug retrieve route - this is used to get the debug retrieve information
# parameters:
# - data: DebugRetrieveRequest - the debug retrieve request - this is the request from the frontend
# returns:
# - The debug retrieve information - this is the debug retrieve information
@router.post("/retrieve")
async def debug_retrieve(data: DebugRetrieveRequest):
    from app.ai.ai_services.graph import weaviate_client, tei_reranker
    from app.ai.ai_services.retrievers.factory import create_retriever, resolve_retrieval_mode, _get_reranker_top_k
    from app.ai.ai_services.retrievers.sentence_first import SentenceFirstRetriever
    # start timer
    t_total_start = time.time()
    # if the retrieval config is set, set the retrieval overrides - log it
    if data.retrieval_config:
        overrides = {k: v for k, v in data.retrieval_config.model_dump().items() if v is not None}
        set_retrieval_overrides(overrides)
    else:
        overrides = {}
        clear_retrieval_overrides()
    # try to get the embeddings, mode, retriever, and parent filter info
    try:
    # embedings used
        embeddings = _get_embeddings()
    # how to retrieve the documents
        mode = get_param('retrieval_mode', None) or resolve_retrieval_mode()
    # retrive and mode_reranker_top_k is the top k for the reranker
        retriever, mode_reranker_top_k = create_retriever(weaviate_client, embeddings, mode=mode)

        # Per-request override: shrink candidate pool into the reranker (not only output top_k).
        _override_k = get_param('reranker_top_k', None)
        if _override_k is None:
            _override_k = get_param('sentence_reranker_top_k', None)
        if _override_k is not None:
            mode_reranker_top_k = int(_override_k)
            if hasattr(retriever, 'final_k'):
                retriever.final_k = mode_reranker_top_k

        if hasattr(retriever, 'set_user_type'):
            retriever.set_user_type(data.user_type)
        # try to get the parent filter info
        parent_filter_info = None
        # set up parent_filter_name for current request
        _parent_name = get_param('parent_filter_name', None)
        if _parent_name:
            _parent_hash = resolve_parent_document_hash(weaviate_client, _parent_name)
            if _parent_hash:
                parent_filter_info = {"name": _parent_name, "resolved_hash": _parent_hash, "status": "active"}
            else:
                _doc_coll = resolve_collection_name("Documents")
                parent_filter_info = {
                    "name": _parent_name,
                    "resolved_hash": None,
                    "status": "not_found",
                    "warning": (
                        f"No document found in {_doc_coll} with "
                        f"document_title '{_parent_name}'. "
                        "Filter ignored — retrieving from all documents."
                    ),
                }
        # try to get the site filter info
        site_filter_info = None
        _site_name = get_param('site_filter_name', None)
        if _site_name:
            site_filter_info = {"name": _site_name, "status": "active"}
        # try to get the content filters info
        content_filters = get_content_filter_debug_info()
        person_filter_info = content_filters.get("person_filter")
        city_filter_info = content_filters.get("city_filter")
        country_filter_info = content_filters.get("country_filter")
        organisation_filter_info = content_filters.get("organisation_filter")
        access_groups_filter_info = content_filters.get("access_groups_filter")
        copyright_filter_info = content_filters.get("copyright_filter")

        # create the config used for the debug retrieve
        config_used = {
            "retrieval_mode": mode,
            "hybrid_alpha": get_param('hybrid_alpha', HYBRID_ALPHA),
            "sentence_retrieval_k": get_param('sentence_retrieval_k', SENTENCE_RETRIEVAL_K),
            "heading_boost_weight": get_param('heading_boost_weight', HEADING_BOOST_WEIGHT),
            "heading_inject_min_sim": get_param('heading_inject_min_sim', HEADING_INJECT_MIN_SIM),
            "heading_inject_max": get_param('heading_inject_max', HEADING_INJECT_MAX),
            "heading_inject_min_sents": get_param('heading_inject_min_sents', HEADING_INJECT_MIN_SENTS),
            "top_n_per_section": get_param('top_n_per_section', TOP_N_PER_SECTION),
            "max_sections_per_url": get_param('max_sections_per_url', MAX_SECTIONS_PER_URL),
            "sentence_highlight_mode": get_param('sentence_highlight_mode', SENTENCE_HIGHLIGHT_MODE),
            "sentence_ce_candidates": get_param('sentence_ce_candidates', SENTENCE_CE_CANDIDATES),
            "sentence_ce_max_groups": get_param('sentence_ce_max_groups', SENTENCE_CE_MAX_GROUPS),
            "use_reranker": get_param('use_reranker', USE_RERANKER),
            "reranker_top_k": get_param('reranker_top_k', mode_reranker_top_k),
            "sentence_reranker_top_k": get_param('sentence_reranker_top_k', SENTENCE_RERANKER_TOP_K),
            "retrieval_chunks": get_param('retrieval_chunks', RETRIEVAL_CHUNKS),
            "use_dynamic_label_weights": get_param('use_dynamic_label_weights', USE_DYNAMIC_LABEL_WEIGHTS),
            "dynamic_weight_alpha": get_param('dynamic_weight_alpha', DYNAMIC_WEIGHT_ALPHA),
            "dynamic_weight_min_sim": get_param('dynamic_weight_min_sim', DYNAMIC_WEIGHT_MIN_SIM),
            "use_recency_boost": get_param('use_recency_boost', USE_RECENCY_BOOST),
            "recency_boost_max_age_days": get_param('recency_boost_max_age_days', RECENCY_BOOST_MAX_AGE_DAYS),
            "recency_half_life_days": get_param('recency_half_life_days', RECENCY_HALF_LIFE_DAYS),
            "recency_max_boost": get_param('recency_max_boost', RECENCY_MAX_BOOST),
            "user_type": data.user_type,
            "overrides_applied": overrides,
            "parent_filter": parent_filter_info,
            "site_filter": site_filter_info,
            "person_filter": person_filter_info,
            "city_filter": city_filter_info,
            "country_filter": country_filter_info,
            "organisation_filter": organisation_filter_info,
            "access_groups_filter": access_groups_filter_info,
            "copyright_filter": copyright_filter_info,
        }
        # try to get the docs and debug data
        debug_data = {}
        # if the retriever is a sentence first retriever, run the debug function
        if isinstance(retriever, SentenceFirstRetriever):
        # inside docs, debug_data = store the debug data for the sentence first retriever
            docs, debug_data = retriever.run_debug(data.query)
        else:
        # this gets the documents with CallbackManagerForRetrieverRun -> get_relevant_documents - > its only for debugging
            docs = retriever.invoke(data.query)
        # start timer for reranker
        t_reranker_start = time.time()
        _use_reranker = get_param('use_reranker', USE_RERANKER)
        _reranker_top_k = get_param('reranker_top_k', mode_reranker_top_k)
        # post reranker docs are the documents after the reranker
        post_reranker_docs = []
        # call the reranker to rerank the documents
        if tei_reranker is not None and _use_reranker:
        # in rerank var = rerank for query - docs that are retrieved from Weaviate database - and reranker top k is 
            reranked = tei_reranker.rerank(data.query, docs, top_k=_reranker_top_k)
        # for each document in reranked, add the metadata to the post reranker docs
            for d in reranked:
                post_reranker_docs.append({
                    "url": d.metadata.get("url", ""),
                    "title": d.metadata.get("title", ""),
                    "section": d.metadata.get("section", ""),
                    "label": d.metadata.get("label", ""),
                    "post_type": d.metadata.get("post_type", ""),
                    "reranker_score": round(d.metadata.get("_reranker_score", 0), 6),
                    "best_sentence": d.metadata.get("_best_sentence", ""),
                    "content_preview": d.page_content[:300],
                })
            docs = reranked
        else:
            post_reranker_docs = [
                {
                    "url": d.metadata.get("url", ""),
                    "title": d.metadata.get("title", ""),
                    "label": d.metadata.get("label", ""),
                    "reranker_score": d.metadata.get("_final_score", 0),
                    "content_preview": d.page_content[:300],
                }
                for d in docs
            ]
        # end timer for reranker
        t_reranker_elapsed = time.time() - t_reranker_start

        # Compute label weights (dynamic or static)
        query_vector = getattr(retriever, 'last_query_vector', None)
        _use_dynamic = get_param('use_dynamic_label_weights', USE_DYNAMIC_LABEL_WEIGHTS)
        intent_similarities = {}
        # if the dynamic label weights are used, get the dynamic label weights
        if _use_dynamic and query_vector:
            _alpha = get_param('dynamic_weight_alpha', DYNAMIC_WEIGHT_ALPHA)
            _min_sim = get_param('dynamic_weight_min_sim', DYNAMIC_WEIGHT_MIN_SIM)
            label_weights, intent_similarities = get_dynamic_label_weights(
                query_vector, data.user_type, alpha=_alpha, min_similarity=_min_sim
            )
        else:
            label_weights = get_label_weights(data.user_type)

        static_weights = get_label_weights(data.user_type)
        _chunks = get_param('retrieval_chunks', RETRIEVAL_CHUNKS)

        post_label_docs = []
        # for each document in docs, add the metadata to the post label docs
        for d in docs:
            label = d.metadata.get('label', '')
            lw = label_weights.get(label, DEFAULT_LABEL_WEIGHT)
            base_score = d.metadata.get('_reranker_score',
                         d.metadata.get('_final_score',
                         d.metadata.get('_rescored', 0)))
            rm = get_recency_multiplier_for_metadata(d.metadata)
            post_score = base_score * lw * rm
            d.metadata['_label_weight'] = lw
            d.metadata['_recency_multiplier'] = rm
            d.metadata['_post_label_score'] = post_score

            post_label_docs.append({
                "url": d.metadata.get("url", ""),
                "title": d.metadata.get("title", ""),
                "section": d.metadata.get("section", ""),
                "label": label,
                "post_type": d.metadata.get("post_type", ""),
                "date": str(d.metadata.get("date", "")),
                "reranker_score": round(base_score, 6),
                "label_weight": round(lw, 2),
                "recency_multiplier": round(rm, 4),
                "final_score": round(post_score, 6),
                "best_sentence": d.metadata.get("_best_sentence", ""),
                "content_preview": d.page_content[:300],
            })

        post_label_docs.sort(key=lambda x: x["final_score"], reverse=True)

        final_selection = post_label_docs[:_chunks]

        # ── Graph Boost + Graph Context (steps 7b, 7c) ──────────────
        step_7b_graph_boost = []
        step_7c_graph_context = []
        subgraph_url = None
        graph_timing = 0.0
        subgraph_url = None

        graph_enabled = get_param('enable_graph_expansion', GRAPH_EXPANSION_ENABLED)
        if graph_enabled:
            from app.core.singleton import get_graph_expansion_service
            from app.ai.ai_services.graph_expansion import GraphExpansionService
            from app.ai.ai_services.graph_expansion_node import _save_static_subgraph

            service = get_graph_expansion_service()
            if service is not None:
                final_docs = [d for d in docs if any(
                    fs["url"] == d.metadata.get("url", "") for fs in final_selection
                )]

                try:
                    expansions, graph_elapsed = await service.expand_documents(
                        final_docs, site="diplomacy.edu",
                    )
                    graph_timing = graph_elapsed

                    if expansions:
                        boosts = service.compute_graph_boost_scores(
                            final_docs, expansions, data.query, alpha=1.0,
                        )
                        alpha = GRAPH_BOOST_ALPHA

                        boost_entries = []
                        for i, doc in enumerate(final_docs):
                            url = doc.metadata.get("url", "")
                            base = doc.metadata.get(
                                "_reranker_score",
                                doc.metadata.get("_final_score",
                                doc.metadata.get("_post_label_score", 0)),
                            )
                            boosted = base + alpha * boosts[i]
                            boost_entries.append({
                                "url": url,
                                "title": doc.metadata.get("title", ""),
                                "section": doc.metadata.get("section", ""),
                                "label": doc.metadata.get("label", ""),
                                "original_score": round(base, 6),
                                "graph_boost_score": round(boosts[i], 6),
                                "boosted_score": round(boosted, 6),
                                "alpha": alpha,
                            })
                        boost_entries.sort(key=lambda x: x["boosted_score"], reverse=True)
                        step_7b_graph_boost = boost_entries

                        # Render the static subgraph HTML so the dashboard can
                        # preview it as an iframe (same renderer the chat path uses).
                        try:
                            from app.ai.ai_services.graph_expansion_node import _save_static_subgraph
                            subgraph_url = _save_static_subgraph(
                                data.query, final_docs, expansions, service,
                            )
                        except Exception as e:
                            logger.warning("Debug subgraph render failed: %s", e, exc_info=True)

                        service.enrich_document_metadata(final_docs, expansions)
                        for doc in final_docs:
                            url = doc.metadata.get("url", "")
                            inline = service.format_inline_source_context(doc, expansions)
                            step_7c_graph_context.append({
                                "url": url,
                                "title": doc.metadata.get("title", ""),
                                "label": doc.metadata.get("label", ""),
                                "graph_topics": doc.metadata.get("_graph_topics", []),
                                "graph_people": doc.metadata.get("_graph_people", []),
                                "graph_actors": doc.metadata.get("_graph_actors", []),
                                "graph_expanded": doc.metadata.get("_graph_expanded", False),
                                "inline_context": inline,
                            })

                        # Render and persist the static subgraph HTML so the
                        # Debug Retrieve view can show the same vis-network
                        # visualisation that the chat flow exposes.
                        try:
                            subgraph_url = _save_static_subgraph(
                                data.query, final_docs, expansions, service,
                            )
                        except Exception as se:
                            logger.warning("Debug subgraph render failed: %s", se, exc_info=True)

                except Exception as e:
                    logger.warning("Debug graph expansion failed: %s", e, exc_info=True)

        t_total_elapsed = time.time() - t_total_start

        timing = debug_data.get("timing", {})
        timing["reranker"] = round(t_reranker_elapsed, 4)
        timing["graph_expansion"] = round(graph_timing, 4)
        timing["total_endpoint"] = round(t_total_elapsed, 4)

        steps = {
            "step_1a_hybrid_sentences": debug_data.get("step_1a_hybrid_sentences", []),
            "step_1b_heading_boost": debug_data.get("step_1b_heading_boost", []),
            "step_1b_url_boost_map": debug_data.get("step_1b_url_boost_map", {}),
            "step_1c_injected": debug_data.get("step_1c_injected", []),
            "step_2_3_section_groups": debug_data.get("step_2_3_section_groups", []),
            "step_5_pre_reranker": debug_data.get("step_5_pre_reranker", []),
            "step_6_post_reranker": post_reranker_docs,
            "step_7_post_label_weights": post_label_docs,
            "step_7b_graph_boost": step_7b_graph_boost,
            "step_7c_graph_context": step_7c_graph_context,
            "step_8_final_selection": final_selection,
        }

        counts = debug_data.get("counts", {})
        counts["post_reranker"] = len(post_reranker_docs)
        counts["post_label_weights"] = len(post_label_docs)
        counts["graph_boost"] = len(step_7b_graph_boost)
        counts["graph_context"] = len(step_7c_graph_context)
        counts["final_selection"] = len(final_selection)

        # Sort similarities descending for readability
        sorted_sims = dict(sorted(intent_similarities.items(), key=lambda x: x[1], reverse=True))

        return {
            "query": data.query,
            "config_used": config_used,
            "timing": timing,
            "counts": counts,
            "steps": steps,
            "label_weights_applied": label_weights,
            "label_weights_static": static_weights,
            "intent_similarities": {k: round(v, 4) for k, v in sorted_sims.items()},
            "subgraph_url": subgraph_url,
        }

    finally:
        clear_retrieval_overrides()


@router.get("/neo4j-health")
async def neo4j_health():
    """Check if the Neo4j graph expansion service is running and healthy."""
    from app.core.singleton import get_graph_expansion_service
    service = get_graph_expansion_service()
    if service is None:
        return {"status": "disabled"}
    try:
        healthy = await service._client.health_check()
        return {"status": "ok" if healthy else "error"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
