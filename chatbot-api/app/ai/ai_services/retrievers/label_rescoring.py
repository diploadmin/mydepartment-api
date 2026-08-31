"""
LabelRescoringRetriever — paragraph-level hybrid search with label scoring.

Simple retriever that fetches from Weaviate Paragraphs,
deduplicates by content hash, and returns top N results.
Label re-scoring is applied post-reranker in the graph pipeline.
"""

from typing import List

import weaviate
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from weaviate.classes.query import HybridFusion

from app.core.config import HYBRID_ALPHA
from app.core.weaviate_props import (
    TEXT,
    PARENT_DOCUMENT_HASH,
    doc_url,
    doc_title,
    doc_date,
    source_type,
    label_from_props,
    normalize_props,
)
from app.ai.ai_services.label_weights import get_label_weights
from app.ai.ai_services.retrievers.utils import (
    resolve_collection_name,
    build_base_retrieval_filter,
)


class LabelRescoringRetriever(BaseRetriever):
    """
    Paragraph-level retriever for oneweaviate. Label weights applied post-reranker.
    """
    client: weaviate.WeaviateClient
    embeddings: Embeddings
    index_names: List[str]
    k_per_index: int = 50
    final_k: int = 8
    alpha: float = HYBRID_ALPHA
    user_type: str = "general"

    model_config = {"arbitrary_types_allowed": True}

    def set_user_type(self, user_type: str):
        self.user_type = user_type.lower() if user_type else "general"
        print(
            f"DEBUG: LabelRescoringRetriever user_type set to: {self.user_type}",
            flush=True,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_vector = self.embeddings.embed_query(query)

        label_weights = get_label_weights(self.user_type)
        print(
            f"DEBUG: Using profile '{self.user_type}' with weights: "
            f"Topic={label_weights.get('Topic', 1.0)}, "
            f"Courses={label_weights.get('Courses', 1.0)}, "
            f"Blog={label_weights.get('Blog', 1.0)}",
            flush=True,
        )

        all_docs_with_scores = []

        for index_name in self.index_names:
            effective_name = resolve_collection_name(index_name)
            _query_filter = build_base_retrieval_filter(self.client, effective_name)
            collection = self.client.collections.get(effective_name)
            hybrid_kwargs = dict(
                query=query,
                vector=query_vector,
                alpha=self.alpha,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                limit=self.k_per_index,
                query_properties=[TEXT],
                return_metadata=["score"],
            )
            if _query_filter is not None:
                hybrid_kwargs["filters"] = _query_filter
            results = collection.query.hybrid(**hybrid_kwargs)

            for obj in results.objects:
                props = normalize_props(obj.properties or {})
                original_score = obj.metadata.score if obj.metadata else 0.0
                label = label_from_props(props)
                title = doc_title(props)

                metadata = {
                    "title": title,
                    "date": doc_date(props),
                    "url": doc_url(props),
                    "label": label,
                    "post_type": source_type(props),
                    "parent_document_hash": props.get(PARENT_DOCUMENT_HASH) or "",
                    "_original_score": original_score,
                    "_label_weight": 1.0,
                    "_rescored": original_score,
                    "_index": effective_name,
                    "_user_type": self.user_type,
                }

                doc = Document(
                    page_content=props.get(TEXT) or "",
                    metadata=metadata,
                )
                all_docs_with_scores.append((doc, original_score))

        all_docs_with_scores.sort(key=lambda x: x[1], reverse=True)

        seen_content = set()
        unique_docs_with_scores = []
        for doc, score in all_docs_with_scores:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content and doc.page_content:
                seen_content.add(content_hash)
                unique_docs_with_scores.append((doc, score))

        return [doc for doc, score in unique_docs_with_scores[: self.final_k]]
