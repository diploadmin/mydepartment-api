"""
Retriever strategies for the Diplomacy chatbot.

Available strategies:
- SentenceFirstRetriever: sentence-level hybrid search with section grouping
- CombinedRetriever: merges sentence-first + paragraph-level results
- TwoPhaseRetriever: URL discovery + parallel section selection
- LabelRescoringRetriever: paragraph-level hybrid search with label scoring
"""

from app.ai.ai_services.retrievers.sentence_first import SentenceFirstRetriever
from app.ai.ai_services.retrievers.combined import CombinedRetriever
from app.ai.ai_services.retrievers.twophase import TwoPhaseRetriever
from app.ai.ai_services.retrievers.label_rescoring import LabelRescoringRetriever
from app.ai.ai_services.retrievers.factory import create_retriever, RETRIEVAL_MODES

__all__ = [
    "SentenceFirstRetriever",
    "CombinedRetriever",
    "TwoPhaseRetriever",
    "LabelRescoringRetriever",
    "create_retriever",
    "RETRIEVAL_MODES",
]
