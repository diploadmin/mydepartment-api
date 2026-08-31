"""
Neo4j configuration for Chatbot API graph expansion.
Reads all settings from environment variables with sensible defaults.

Env vars to add to the chatbot's .env or docker-compose:
    NEO4J_URI, NEO4J_USER, NEO4J_PASS,
    NEO4J_DATABASE_DIPLO, NEO4J_DATABASE_DW,
    GRAPH_EXPANSION_ENABLED, GRAPH_EXPANSION_TIMEOUT, ...
"""

import os


# ── Connection ────────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://nimani.diplomacy.edu:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "")

NEO4J_DATABASE_DIPLO = os.getenv("NEO4J_DATABASE_DIPLO", "weaviatediplo")
NEO4J_DATABASE_DW = os.getenv("NEO4J_DATABASE_DW", "weaviatedw")

# ── Feature flags ─────────────────────────────────────────────────────
GRAPH_EXPANSION_ENABLED = os.getenv("GRAPH_EXPANSION_ENABLED", "False").lower() in ("true", "1", "yes")

# Timeout per expansion batch (seconds). If Neo4j doesn't respond in time,
# the pipeline falls back to original retrieval results without graph context.
GRAPH_EXPANSION_TIMEOUT = float(os.getenv("GRAPH_EXPANSION_TIMEOUT", "3.0"))

# Max outgoing + incoming relations to fetch per document
GRAPH_EXPANSION_MAX_RELATIONS = int(os.getenv("GRAPH_EXPANSION_MAX_RELATIONS", "30"))

# Max depth for multi-hop expansion (1 = direct relations only, 2 = friends of friends)
GRAPH_EXPANSION_MAX_DEPTH = int(os.getenv("GRAPH_EXPANSION_MAX_DEPTH", "1"))

# ── Mapping site → database ──────────────────────────────────────────
SITE_DB_MAP: dict[str, str] = {
    "diplomacy.edu": NEO4J_DATABASE_DIPLO,
    "www.diplomacy.edu": NEO4J_DATABASE_DIPLO,
    "dig.watch": NEO4J_DATABASE_DW,
    "www.dig.watch": NEO4J_DATABASE_DW,
}


def get_neo4j_database(site_name: str) -> str:
    """Resolve WEBSITE_NAME or site filter to the correct Neo4j database."""
    for key, db in SITE_DB_MAP.items():
        if key in site_name:
            return db
    return NEO4J_DATABASE_DIPLO
