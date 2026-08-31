"""
Integration Patch — code snippets showing exactly where and how to wire
the graph expansion into the existing chatbot API.

This is NOT meant to be executed directly. It documents the minimal changes
needed in each existing file on the chatbot server.

================================================================
FILE 1:  app/core/singleton.py
ACTION:  Add Neo4j client + graph expansion service singletons
================================================================
"""

# ---- ADD these imports at top of singleton.py ----

from typing import Optional
from app.core.neo4j_config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASS,
    NEO4J_DATABASE_DIPLO, NEO4J_DATABASE_DW,
    GRAPH_EXPANSION_ENABLED,
)

# ---- ADD these module-level variables ----

_neo4j_client = None          # type: Optional[Neo4jGraphClient]
_graph_expansion_service = None  # type: Optional[GraphExpansionService]


# ---- ADD these functions ----

async def init_graph_expansion():
    """Initialize Neo4j connection and graph expansion service (call on startup)."""
    global _neo4j_client, _graph_expansion_service
    if not GRAPH_EXPANSION_ENABLED:
        return
    from app.ai.ai_services.neo4j_graph_client import Neo4jGraphClient
    from app.ai.ai_services.graph_expansion import GraphExpansionService

    _neo4j_client = Neo4jGraphClient(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASS,
        database_diplo=NEO4J_DATABASE_DIPLO,
        database_dw=NEO4J_DATABASE_DW,
    )
    await _neo4j_client.connect()
    healthy = await _neo4j_client.health_check()
    if healthy:
        _graph_expansion_service = GraphExpansionService(_neo4j_client)
        print(f"[STARTUP] Neo4j graph expansion: ENABLED ({NEO4J_URI})")
    else:
        print(f"[STARTUP] Neo4j graph expansion: FAILED health check ({NEO4J_URI})")
        await _neo4j_client.close()
        _neo4j_client = None


def get_graph_expansion_service():
    """Get the graph expansion service singleton (None if disabled)."""
    return _graph_expansion_service


async def shutdown_graph_expansion():
    """Close Neo4j connection (call on shutdown)."""
    global _neo4j_client, _graph_expansion_service
    _graph_expansion_service = None
    if _neo4j_client:
        await _neo4j_client.close()
        _neo4j_client = None


"""
================================================================
FILE 2:  app/core/events.py
ACTION:  Call init/shutdown in app lifecycle hooks
================================================================
"""

# ---- IN startup handler, ADD: ----

# async def startup():
#     ...existing code...
#     from app.core.singleton import init_graph_expansion
#     await init_graph_expansion()


# ---- IN shutdown handler, ADD: ----

# async def shutdown():
#     ...existing code...
#     from app.core.singleton import shutdown_graph_expansion
#     await shutdown_graph_expansion()


"""
================================================================
FILE 3:  app/ai/ai_services/graph.py — OPTION A (inline)
ACTION:  Add graph expansion call inside direct_retrieval_node()
         This is the simplest approach — no new LangGraph nodes.
================================================================
"""


# ---- LOCATE this block in direct_retrieval_node() (around line ~435-460): ----
#
#   docs = apply_post_reranker_label_weights(docs, _chunks, ...)
#   ...
#   tool_content = ...  # format "[1] Title\ncontent\n---\n..."
#
# ---- AFTER apply_post_reranker_label_weights, BEFORE formatting, ADD: ----

async def _example_inline_integration(docs, tool_content, site_name):
    """
    Shows the exact code to insert in direct_retrieval_node().
    This is pseudocode — adapt variable names to your actual code.
    """
    import time
    from app.core.singleton import get_graph_expansion_service

    # --- Graph Expansion (insert after label weights, before tool_content formatting) ---
    graph_service = get_graph_expansion_service()
    if graph_service is not None:
        try:
            expansions, graph_elapsed = await graph_service.expand_documents(
                docs, site=site_name
            )
            if expansions:
                graph_service.enrich_document_metadata(docs, expansions)
                graph_context = graph_service.format_graph_context(docs, expansions)
                tool_content = tool_content + graph_context
                print(f"TIMING graph_expansion: {graph_elapsed:.3f}s "
                      f"({len(expansions)}/{len(docs)} docs expanded)")
        except Exception as e:
            print(f"Graph expansion failed (non-fatal): {e}")
    # --- End Graph Expansion ---

    return tool_content


"""
================================================================
FILE 3 ALT:  app/ai/ai_services/graph.py — OPTION C (new node)
ACTION:  Add graph_expansion as a separate LangGraph node.
         Cleanest architecture — separate concern.
================================================================
"""

# ---- IN initialize_diplomacy_bot_graph(), CHANGE: ----

# BEFORE:
#   builder.add_edge("direct_retrieval", "generate_final")
#
# AFTER:
#   from app.ai.ai_services.graph_expansion_node import graph_expansion_node
#   builder.add_node("graph_expansion", graph_expansion_node)
#   builder.add_edge("direct_retrieval", "graph_expansion")
#   builder.add_edge("graph_expansion", "generate_final")

# ---- IN direct_retrieval_node(), also save docs to state: ----
#   state["_retrieved_docs"] = docs
#   state["_site"] = site_name  # "diplomacy.edu" or "dig.watch"


"""
================================================================
FILE 4:  app/api/routes/debug_route.py
ACTION:  Add Neo4j health check endpoint
================================================================
"""

# @router.get("/debug/neo4j-health")
# async def neo4j_health():
#     from app.core.singleton import get_graph_expansion_service
#     service = get_graph_expansion_service()
#     if service is None:
#         return {"status": "disabled", "graph_expansion_enabled": False}
#     healthy = await service._client.health_check()
#     return {"status": "ok" if healthy else "error", "graph_expansion_enabled": True}


"""
================================================================
FILE 5:  app/schemas/chat_schema.py
ACTION:  Add graph expansion toggle to RetrievalConfig
================================================================
"""

# ---- IN class RetrievalConfig, ADD fields: ----

# enable_graph_expansion: Optional[bool] = None
# graph_max_relations: Optional[int] = None
# graph_max_depth: Optional[int] = None

# This allows frontend to control graph expansion per-request via retrieval_config.


"""
================================================================
FILE 6:  requirements.txt
ACTION:  Add neo4j async driver
================================================================
"""

# neo4j>=5.17.0    # Async driver for graph expansion
