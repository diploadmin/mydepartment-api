"""
Backward-compatibility shim.

All retrieval logic has been refactored into separate modules:
- app.ai.ai_services.embeddings         — TEIEmbeddings
- app.ai.ai_services.reranker           — TEIReranker
- app.ai.ai_services.label_weights      — profile weights, POST_TYPE_TO_LABEL
- app.ai.ai_services.retrieval_cache    — Redis cache helpers
- app.ai.ai_services.retrievers/        — retriever strategies + factory
- app.ai.ai_services.graph              — LangGraph agent, Weaviate client

This file re-exports public symbols so existing imports continue to work.
"""

# Graph + Weaviate client
from app.ai.ai_services.graph import (
    initialize_diplomacy_bot_graph,
    initialize_resource_filter,
    get_weaviate_client,
    weaviate_client,
    tei_reranker,
)

# Retrievers
from app.ai.ai_services.retrievers import (
    SentenceFirstRetriever,
    CombinedRetriever,
    TwoPhaseRetriever,
    LabelRescoringRetriever,
    create_retriever,
)

# Supporting modules
from app.ai.ai_services.embeddings import TEIEmbeddings
from app.ai.ai_services.reranker import TEIReranker
from app.ai.ai_services.label_weights import (
    PROFILE_LABEL_WEIGHTS,
    POST_TYPE_TO_LABEL,
    DEFAULT_LABEL_WEIGHT,
    get_label_weights,
)
from app.ai.ai_services.retrieval_cache import (
    _cache_key,
    _serialize_docs,
    _deserialize_docs,
    _is_json_serializable,
)
