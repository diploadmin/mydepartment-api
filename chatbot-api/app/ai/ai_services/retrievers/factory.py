"""
Retriever factory — creates the appropriate retriever based on config.

Replaces the if/elif chain that was inside initialize_diplomacy_bot_graph().
"""

from typing import Tuple

import weaviate
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from app.core.config import (
    INDEX_NAME, INDEX_NAME2, # collection names in Weaviate
    HYBRID_ALPHA, # search by embeddings or BM25 (word find algorithm) or a combination of both
    SENTENCE_RETRIEVAL_K, SENTENCE_INDEX_NAME, # collection names in Weaviate
    SENTENCE_HEADER_RETRIEVAL_K, SENTENCE_TITLE_K,
    SENTENCE_TITLE_FETCH_LIMIT, SENTENCE_TITLE_ALPHA,
    PARAGRAPH_RETRIEVAL_K, # how many paragraphs to retrieve from the collection
    RETRIEVAL_MODE, USE_SENTENCE_RETRIEVAL, # use sentence retrieval or paragraph retrieval
    USE_RERANKER, RERANKER_TOP_K, # use reranker or not
    TWOPHASE_RERANKER_TOP_K, SENTENCE_RERANKER_TOP_K, # how many documents to rerank
    RETRIEVAL_CHUNKS, TWOPHASE_K_URLS, # how many documents to retrieve from the collection
    USE_CONTEXTUAL_COLLECTIONS, # use contextual collections or not
)
# factory creates one of the retrievers specified bellow ⬇️
from app.ai.ai_services.retrievers.sentence_first import SentenceFirstRetriever
from app.ai.ai_services.retrievers.combined import CombinedRetriever
from app.ai.ai_services.retrievers.twophase import TwoPhaseRetriever
from app.ai.ai_services.retrievers.label_rescoring import LabelRescoringRetriever

# retrieval modes - sentence, paragraph, combined, twophase, sentence_header
RETRIEVAL_MODES = ('sentence', 'paragraph', 'combined', 'twophase', 'sentence_header')


def _get_reranker_top_k(mode: str) -> int:
    if mode in ('sentence', 'combined', 'sentence_header') and SENTENCE_RERANKER_TOP_K > 0:
        return SENTENCE_RERANKER_TOP_K
    elif mode == 'twophase' and TWOPHASE_RERANKER_TOP_K > 0:
        return TWOPHASE_RERANKER_TOP_K
    return RERANKER_TOP_K

# description: resolve the retrieval mode from the config
# if no mode is sentence if USE_SENTENCE_RETRIEVAL is true, otherwise paragraph
def resolve_retrieval_mode() -> str:
    """Resolve effective retrieval mode from config (with legacy fallback)."""
    mode = RETRIEVAL_MODE.lower()
    if mode not in RETRIEVAL_MODES:
        mode = 'sentence' if USE_SENTENCE_RETRIEVAL else 'paragraph'
    return mode
# context: create_retriever is called by the graph.py file - this is the function that creates the retriever based on the mode
# - it fils up active_retriever, mode_reranker_top_k variables with its outputs
# description: function that calls the appropriate retriever from services/retrievers
# parameters:
# - client: weaviate.WeaviateClient - the weaviate client - to enter weaviate database
# - embeddings: Embeddings - the embeddings - to embed the query - TEI or openai
# - mode: str | None - the mode - if None, resolve the mode from the config - sentence, paragraph, combined, twophase
# returns:
# - Tuple[BaseRetriever, int] - the retriever and the mode reranker top k
def create_retriever(
    client: weaviate.WeaviateClient,
    embeddings: Embeddings,
    mode: str | None = None,
) -> Tuple[BaseRetriever, int]:
    """Create a retriever based on the specified mode.

    Returns (retriever, mode_reranker_top_k).
    """
    from app.core.logging import logger # to log the retrieval mode and reranker top k

    # if no mode is provided, resolve the mode from the config
    if mode is None:
        mode = resolve_retrieval_mode()
    mode_reranker_top_k = _get_reranker_top_k(mode)
    retriever_final_k = mode_reranker_top_k if USE_RERANKER else RETRIEVAL_CHUNKS

    # if the mode is combined, use the combined retriever - and log what is set up
    if mode == 'combined':
        logger.info(f"Using CombinedRetriever: {SENTENCE_RETRIEVAL_K} sentences from "
                    f"{SENTENCE_INDEX_NAME} + {PARAGRAPH_RETRIEVAL_K} paragraphs from {INDEX_NAME}")
        # create the sentence retriever - using the SentenceFirstRetriever class
        sentence_retriever = SentenceFirstRetriever(
            client=client, # weaviate client
            embeddings=embeddings, # embeddings for the query - TEI or openai - this is pulled from - langchain_core.embeddings -  
            index_name=SENTENCE_INDEX_NAME, # from the config - SENTENCE_INDEX_NAME - this is collection name in Weaviate
            k_sentences=SENTENCE_RETRIEVAL_K, # from the config - SENTENCE_RETRIEVAL_K - how many sentences to retrieve from the collection
            final_k=retriever_final_k, # from the config - retriever_final_k - how many chunks to return to the LLM
            alpha=HYBRID_ALPHA # from the config - HYBRID_ALPHA - search by embeddings or BM25 (word find algorithm) or a combination of both
        )
        retriever = CombinedRetriever(
            client=client,
            embeddings=embeddings,
            sentence_retriever=sentence_retriever,
            paragraph_index=INDEX_NAME,
            k_paragraphs=PARAGRAPH_RETRIEVAL_K,
            alpha=HYBRID_ALPHA
        )

    # if the mode is twophase, use the twophase retriever
    # 
    elif mode == 'twophase':
        logger.info(f"Using TwoPhaseRetriever: vector(200) + title filter → "
                    f"{TWOPHASE_K_URLS} URLs → reranker top {mode_reranker_top_k} from {INDEX_NAME}")
        retriever = TwoPhaseRetriever(
            client=client,
            embeddings=embeddings,
            paragraph_index=INDEX_NAME,
            k_urls=TWOPHASE_K_URLS,
        )
    # if the mode is sentence, use the sentence retriever
    elif mode == 'sentence':
        logger.info(f"Using SentenceFirstRetriever with {SENTENCE_RETRIEVAL_K} "
                    f"sentences from {SENTENCE_INDEX_NAME}")
        retriever = SentenceFirstRetriever(
            client=client,
            embeddings=embeddings,
            index_name=SENTENCE_INDEX_NAME,
            k_sentences=SENTENCE_RETRIEVAL_K,
            final_k=retriever_final_k,
            alpha=HYBRID_ALPHA
        )

    elif mode == 'sentence_header':
        logger.info(
            f"Using SentenceFirstRetriever+title hybrid: "
            f"{SENTENCE_HEADER_RETRIEVAL_K} sentences + {SENTENCE_TITLE_K} h1 URLs "
            f"from {SENTENCE_INDEX_NAME}"
        )
        retriever = SentenceFirstRetriever(
            client=client,
            embeddings=embeddings,
            index_name=SENTENCE_INDEX_NAME,
            k_sentences=SENTENCE_HEADER_RETRIEVAL_K,
            final_k=retriever_final_k,
            alpha=HYBRID_ALPHA,
            use_title_hybrid=True,
            k_titles=SENTENCE_TITLE_K,
            title_alpha=SENTENCE_TITLE_ALPHA,
            title_fetch_limit=SENTENCE_TITLE_FETCH_LIMIT,
        )

    # if no mode is provided, use the label rescoring retriever
    else:  # paragraph
        logger.info(f"Using LabelRescoringRetriever with {INDEX_NAME}, {INDEX_NAME2}")
        retriever = LabelRescoringRetriever(
            client=client,
            embeddings=embeddings,
            index_names=[INDEX_NAME, INDEX_NAME2],
            k_per_index=50,
            final_k=retriever_final_k,
            alpha=0.6
        )

    logger.info(f"Contextual collections: {'ENABLED' if USE_CONTEXTUAL_COLLECTIONS else 'DISABLED'} (default)")

    # if USE_RERANKER is true, log the reranker is enabled
    if USE_RERANKER:
        # get the reranker URL from the config
        from app.core.config import RERANKER_URL
        # log the reranker is enabled
        logger.info(f"Reranker ENABLED: {RERANKER_URL} (mode={mode}, "
                    f"reranker_top_k={mode_reranker_top_k}, returning {RETRIEVAL_CHUNKS})")
    else:
        logger.info(f"Reranker DISABLED: returning {RETRIEVAL_CHUNKS} documents directly")

    return retriever, mode_reranker_top_k
