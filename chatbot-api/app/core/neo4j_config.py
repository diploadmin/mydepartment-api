"""
Neo4j configuration for Chatbot API graph expansion.

Uses Starlette Config (same pattern as config.py) to read from the .env file.
os.getenv() would NOT work because Starlette Config does not export to system env.
"""

from app.core.config import config

# ── Connection ────────────────────────────────────────────────────────
NEO4J_URI: str = config("NEO4J_URI", cast=str, default="bolt://localhost:7687")
NEO4J_USER: str = config("NEO4J_USER", cast=str, default="neo4j")
NEO4J_PASS: str = config("NEO4J_PASS", cast=str, default="")

NEO4J_DATABASE_DIPLO: str = config("NEO4J_DATABASE_DIPLO", cast=str, default="weaviatediplo")
NEO4J_DATABASE_DW: str = config("NEO4J_DATABASE_DW", cast=str, default="weaviatedw")

# ── Feature flags ─────────────────────────────────────────────────────
GRAPH_EXPANSION_ENABLED: bool = config("GRAPH_EXPANSION_ENABLED", cast=bool, default=False)

GRAPH_EXPANSION_TIMEOUT: float = config("GRAPH_EXPANSION_TIMEOUT", cast=float, default=3.0)

GRAPH_EXPANSION_MAX_RELATIONS: int = config("GRAPH_EXPANSION_MAX_RELATIONS", cast=int, default=30)

GRAPH_EXPANSION_MAX_DEPTH: int = config("GRAPH_EXPANSION_MAX_DEPTH", cast=int, default=1)

# Graph context format injected into the LLM prompt:
#   "none"          = no graph context (graph expansion still runs for metadata enrichment)
#   "appended"      = single block appended after all sources
#   "inline"        = compact [Graph: ...] line under each source
#   "boost_inline"  = graph boosts retrieval ranking + inline metadata in prompt (best in benchmarks)
#   "hint"          = conditional short hints only when document text lacks graph info
GRAPH_CONTEXT_FORMAT: str = config("GRAPH_CONTEXT_FORMAT", cast=str, default="inline")

# Alpha weight for graph boost scoring (only used when format is boost_inline)
# Higher = graph connectivity matters more in re-ranking. 0.15 tested well in benchmarks.
GRAPH_BOOST_ALPHA: float = config("GRAPH_BOOST_ALPHA", cast=float, default=0.15)

# How many extra candidates to graph-expand before re-ranking (boost_inline pool size)
GRAPH_BOOST_POOL_K: int = config("GRAPH_BOOST_POOL_K", cast=int, default=16)

# ── Mapping site → database ──────────────────────────────────────────
SITE_DB_MAP: dict[str, str] = {
    "diplomacy.edu": NEO4J_DATABASE_DIPLO,
    "www.diplomacy.edu": NEO4J_DATABASE_DIPLO,
    "dig.watch": NEO4J_DATABASE_DW,
    "www.dig.watch": NEO4J_DATABASE_DW,
}
# context: used in graph_expansion.py - look up parent document hash and hashed url variants and connect url to the correct Neo4j database.
# given inputs: site_name - from url variants that are connected to the parent document hash.
# description: get the correct Neo4j database based on the site name.
# parameters: site_name - string of the site name
# return: string of the Neo4j database name.
def get_neo4j_database(site_name: str) -> str:
    """Resolve WEBSITE_NAME or site filter to the correct Neo4j database."""
    for key, db in SITE_DB_MAP.items():
        if key in site_name:
            return db
    return NEO4J_DATABASE_DIPLO
