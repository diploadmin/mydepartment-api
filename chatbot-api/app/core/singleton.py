from typing import Optional

from app.ai.ai_services.chatbot import (
    initialize_diplomacy_bot_graph,
    initialize_resource_filter,
    get_weaviate_client,
)
from app.core.websocket_manager import WebSocketManager

from app.core.config import DEBUG

# Shared process-lifetime state for the API (bots, retriever, conversation history, …).
# One instance for all requests until the process restarts.
# this could be fixed because it unloads the RAM on restart.
class Singleton:
    _instance = None
# description: the __new__ method is called to create a new instance of the class.
# it is called before the __init__ method. - it remembers __init__ arguments. and uses them to create a new instance of the class.
    def __new__(cls, *args, **kwargs):
        # if the instance is not None, return it
        if cls._instance is None:
            # create a new instance of the class
            cls._instance = super().__new__(cls)
            # Unpack tuple: (agent, retriever) - diplomacy_bot is the agent, label_retriever is the retriever logic
            diplomacy_bot, label_retriever = initialize_diplomacy_bot_graph()
            cls._instance.diplomacy_bot = diplomacy_bot # LLM agent
            cls._instance.label_retriever = label_retriever  # For setting user_type per request
            # A/B graph: sentence + h1 title hybrid (20 titles / 50 sentences) — /api/chat-sh
            diplomacy_bot_sh, label_retriever_sh = initialize_diplomacy_bot_graph(
                retrieval_mode="sentence_header"
            )
            cls._instance.diplomacy_bot_sh = diplomacy_bot_sh
            cls._instance.label_retriever_sh = label_retriever_sh
            cls._instance.conversation_history = {} # conversation history for the bot
            cls._instance.resource_filter = initialize_resource_filter() # resource filter for the bot
            cls._instance.websocket_manager = WebSocketManager() # websocket manager for the bot
            cls._instance.weaviate_client = get_weaviate_client() # weaviate client for the bot

            if(DEBUG):
                cls._instance.conversation_history["admin"] = [] # conversation history for the admin
                
        return cls._instance


# ── Neo4j Graph Expansion singletons ──────────────────────────────────
# These live outside the Singleton class because Neo4j init is async
# (the existing Singleton uses sync __new__).

_neo4j_client = None # neo4j client for the graph expansion service
_graph_expansion_service = None # graph expansion service for the graph expansion service

# context: this is a singleton for the graph expansion service it contains neo4j client that will be used to query the graph, and do the graph expansion.
# it uses separate config - neo4j_config.py for the neo4j client, nad not config.py for other configs.
async def init_graph_expansion():
    """Initialize Neo4j connection and graph expansion service (call on startup)."""
    global _neo4j_client, _graph_expansion_service
    from app.core.neo4j_config import (
        NEO4J_URI, NEO4J_USER, NEO4J_PASS,
        NEO4J_DATABASE_DIPLO, NEO4J_DATABASE_DW,
        GRAPH_EXPANSION_ENABLED,
    )
    if not GRAPH_EXPANSION_ENABLED:
        print("[STARTUP] Neo4j graph expansion: DISABLED", flush=True)
        return

    from app.ai.ai_services.neo4j_graph_client import Neo4jGraphClient
    from app.ai.ai_services.graph_expansion import GraphExpansionService
    # this is the client for the neo4j DB. it will be used to query the graph, and do the graph expansion.
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
        print(f"[STARTUP] Neo4j graph expansion: ENABLED ({NEO4J_URI})", flush=True)
    else:
        print(f"[STARTUP] Neo4j graph expansion: FAILED health check ({NEO4J_URI})", flush=True)
        await _neo4j_client.close()
        _neo4j_client = None

# get the instance of GraphExpansionService and call it _graph_expansion_service
def get_graph_expansion_service() -> Optional["GraphExpansionService"]:
    """Get the graph expansion service singleton (None if disabled)."""
    return _graph_expansion_service

# shutdown the graph expansion service and close the neo4j client
async def shutdown_graph_expansion():
    """Close Neo4j connection (call on shutdown)."""
    global _neo4j_client, _graph_expansion_service
    _graph_expansion_service = None
    if _neo4j_client:
        await _neo4j_client.close()
        _neo4j_client = None