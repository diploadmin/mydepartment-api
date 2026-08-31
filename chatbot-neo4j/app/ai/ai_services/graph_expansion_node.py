"""
LangGraph node for Neo4j graph expansion.

This is Option C from the architecture document — a standalone node
inserted between direct_retrieval and generate_final:

    START → direct_retrieval_node → ★ graph_expansion_node ★ → generate_final_node → END

The node reads retrieved documents from the state, enriches them with
graph context, and rewrites the ToolMessage with the additional information.
"""

import logging
from typing import Any

from langchain_core.messages import ToolMessage

from app.core.neo4j_config import GRAPH_EXPANSION_ENABLED
from app.ai.ai_services.graph_expansion import GraphExpansionService

logger = logging.getLogger(__name__)


async def graph_expansion_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node that enriches the retrieval ToolMessage with graph context.

    Prerequisites in state:
        - state["messages"]: must contain a ToolMessage with retrieved chunks
        - state["_retrieved_docs"]: list[Document] from retriever (set by direct_retrieval_node)

    The node modifies the last ToolMessage in-place by appending graph context,
    and enriches document metadata for downstream use (deep links, sources).

    If graph expansion is disabled or fails, the state passes through unchanged.
    """
    if not GRAPH_EXPANSION_ENABLED:
        return state

    from app.core.singleton import get_graph_expansion_service
    service: GraphExpansionService = get_graph_expansion_service()
    if service is None:
        return state

    docs = state.get("_retrieved_docs", [])
    if not docs:
        logger.debug("graph_expansion_node: no docs in state, skipping")
        return state

    site = state.get("_site", "diplomacy.edu")

    try:
        expansions, elapsed = await service.expand_documents(docs, site=site)
    except Exception as e:
        logger.error("graph_expansion_node failed: %s", e, exc_info=True)
        return state

    if not expansions:
        return state

    service.enrich_document_metadata(docs, expansions)

    graph_context = service.format_graph_context(docs, expansions)
    if graph_context:
        messages = state.get("messages", [])
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], ToolMessage):
                messages[i] = ToolMessage(
                    content=messages[i].content + graph_context,
                    tool_call_id=messages[i].tool_call_id,
                    name=getattr(messages[i], "name", None),
                )
                break

    state["_retrieved_docs"] = docs
    return state


# ── Graph builder integration ─────────────────────────────────────────
#
# To add this node to the existing LangGraph in graph.py:
#
#   from app.ai.ai_services.graph_expansion_node import graph_expansion_node
#
#   # In initialize_diplomacy_bot_graph():
#   builder.add_node("graph_expansion", graph_expansion_node)
#
#   # Replace the edge:
#   #   builder.add_edge("direct_retrieval", "generate_final")
#   # With:
#   #   builder.add_edge("direct_retrieval", "graph_expansion")
#   #   builder.add_edge("graph_expansion", "generate_final")


# ── Singleton setup ───────────────────────────────────────────────────
#
# Add to app/core/singleton.py:
#
#   _graph_expansion_service: Optional[GraphExpansionService] = None
#   _neo4j_client: Optional[Neo4jGraphClient] = None
#
#   async def init_graph_expansion():
#       global _neo4j_client, _graph_expansion_service
#       from app.core.neo4j_config import (
#           NEO4J_URI, NEO4J_USER, NEO4J_PASS,
#           NEO4J_DATABASE_DIPLO, NEO4J_DATABASE_DW,
#           GRAPH_EXPANSION_ENABLED,
#       )
#       if not GRAPH_EXPANSION_ENABLED:
#           return
#       from app.ai.ai_services.neo4j_graph_client import Neo4jGraphClient
#       from app.ai.ai_services.graph_expansion import GraphExpansionService
#       _neo4j_client = Neo4jGraphClient(
#           NEO4J_URI, NEO4J_USER, NEO4J_PASS,
#           NEO4J_DATABASE_DIPLO, NEO4J_DATABASE_DW,
#       )
#       await _neo4j_client.connect()
#       _graph_expansion_service = GraphExpansionService(_neo4j_client)
#
#   def get_graph_expansion_service() -> Optional[GraphExpansionService]:
#       return _graph_expansion_service
#
#   async def shutdown_graph_expansion():
#       global _neo4j_client
#       if _neo4j_client:
#           await _neo4j_client.close()
#
# Then call init_graph_expansion() in events.py startup,
# and shutdown_graph_expansion() in events.py shutdown.
