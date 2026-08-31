"""
Async Neo4j client for the DiploAI Knowledge Graph.

Provides typed query methods that the chatbot's graph expansion service
calls to enrich retrieval results with structural knowledge.

Uses the same Neo4j databases (weaviatediplo / weaviatedw) populated by
the KG pipeline (run_diplo.py / run_dw.py).
"""

import hashlib
import asyncio
import logging
from typing import Optional
from dataclasses import dataclass, field

from neo4j import AsyncGraphDatabase, AsyncDriver

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class GraphNode:
    """A node from the knowledge graph."""
    node_id: str
    name: str
    labels: list[str]
    document_hash: str = ""
    post_type: str = ""
    url: str = ""
    site: str = ""
    properties: dict = field(default_factory=dict)


@dataclass
class GraphRelation:
    """A single relationship between two entities."""
    source_hash: str
    source_name: str
    relationship: str
    target_hash: str
    target_name: str
    target_labels: list[str]
    target_url: str = ""
    target_post_type: str = ""


@dataclass
class GraphExpansionResult:
    """Aggregated graph context for one retrieved document."""
    document_hash: str
    document_name: str
    relations: list[GraphRelation] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    related_documents: list[GraphNode] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    actors: list[str] = field(default_factory=list)
    subtopic_of: list[str] = field(default_factory=list)


# ── Client ────────────────────────────────────────────────────────────

class Neo4jGraphClient:
    """
    Async Neo4j driver with high-level query methods.

    Lifecycle:
        client = Neo4jGraphClient(uri, user, password)
        await client.connect()          # on app startup
        ...
        await client.close()            # on app shutdown
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database_diplo: str = "weaviatediplo",
        database_dw: str = "weaviatedw",
    ):
        self._uri = uri
        self._user = user
        self._password = password
        self.database_diplo = database_diplo
        self.database_dw = database_dw
        self._driver: Optional[AsyncDriver] = None

    # ── lifecycle ─────────────────────────────────────────────────────

    async def connect(self):
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self._uri, auth=(self._user, self._password)
            )
            logger.info("Neo4j async driver connected: %s", self._uri)

    async def close(self):
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j async driver closed")

    async def health_check(self) -> bool:
        try:
            await self.connect()
            async with self._driver.session(database=self.database_diplo) as session:
                result = await session.run("RETURN 1 AS ok")
                record = await result.single()
                return record is not None and record["ok"] == 1
        except Exception as e:
            logger.error("Neo4j health check failed: %s", e)
            return False

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def url_to_hash(url: str) -> str:
        """Compute document_hash = md5(url), mirroring node_builder._md5."""
        return hashlib.md5(url.encode()).hexdigest()

    def get_database(self, site: str) -> str:
        if "dig.watch" in site:
            return self.database_dw
        return self.database_diplo

    # ── single-document lookups ───────────────────────────────────────

    async def get_document_by_hash(
        self, document_hash: str, database: str
    ) -> Optional[GraphNode]:
        await self.connect()
        query = """
        MATCH (d:Document {document_hash: $hash})
        RETURN d, labels(d) AS labels
        LIMIT 1
        """
        async with self._driver.session(database=database) as session:
            result = await session.run(query, hash=document_hash)
            record = await result.single()
            if not record:
                return None
            node = record["d"]
            return GraphNode(
                node_id=node.get("node_id", ""),
                name=node.get("name", ""),
                labels=record["labels"],
                document_hash=node.get("document_hash", ""),
                post_type=node.get("post_type", ""),
                url=node.get("url", ""),
                site=node.get("site", ""),
                properties=dict(node),
            )

    async def get_document_by_url(
        self, url: str, database: str
    ) -> Optional[GraphNode]:
        """URL-based lookup with www / non-www / trailing-slash normalization."""
        await self.connect()
        query = """
        MATCH (d:Document)
        WHERE d.url IN $variants
        RETURN d, labels(d) AS labels
        LIMIT 1
        """
        variants = _url_variants(url)
        async with self._driver.session(database=database) as session:
            result = await session.run(query, variants=variants)
            record = await result.single()
            if not record:
                return None
            node = record["d"]
            return GraphNode(
                node_id=node.get("node_id", ""),
                name=node.get("name", ""),
                labels=record["labels"],
                document_hash=node.get("document_hash", ""),
                post_type=node.get("post_type", ""),
                url=node.get("url", ""),
                site=node.get("site", ""),
                properties=dict(node),
            )

    # ── relation queries ──────────────────────────────────────────────

    async def get_direct_relations(
        self, document_hash: str, database: str, limit: int = 30
    ) -> list[GraphRelation]:
        """All outgoing relations from a document."""
        await self.connect()
        query = """
        MATCH (d:Document {document_hash: $hash})-[r]->(target)
        RETURN d.name            AS source_name,
               d.document_hash   AS source_hash,
               type(r)           AS rel_type,
               target.name       AS target_name,
               target.document_hash AS target_hash,
               target.url        AS target_url,
               target.post_type  AS target_post_type,
               labels(target)    AS target_labels
        LIMIT $lim
        """
        async with self._driver.session(database=database) as session:
            result = await session.run(query, hash=document_hash, lim=limit)
            records = [r async for r in result]
        return [_record_to_relation(r) for r in records]

    async def get_incoming_relations(
        self, document_hash: str, database: str, limit: int = 15
    ) -> list[GraphRelation]:
        """Incoming edges from other Document nodes (who references me?)."""
        await self.connect()
        query = """
        MATCH (source:Document)-[r]->(d:Document {document_hash: $hash})
        RETURN source.name            AS source_name,
               source.document_hash   AS source_hash,
               type(r)                AS rel_type,
               d.name                 AS target_name,
               d.document_hash        AS target_hash,
               ''                     AS target_url,
               ''                     AS target_post_type,
               labels(source)         AS target_labels
        LIMIT $lim
        """
        async with self._driver.session(database=database) as session:
            result = await session.run(query, hash=document_hash, lim=limit)
            records = [r async for r in result]
        return [_record_to_relation(r) for r in records]

    async def get_shared_topic_documents(
        self, document_hash: str, database: str, limit: int = 8
    ) -> list[GraphNode]:
        """Documents sharing at least one topic with the source document."""
        await self.connect()
        query = """
        MATCH (d:Document {document_hash: $hash})-[r1]->(t:Topic)<-[r2]-(other:Document)
        WHERE other.document_hash <> $hash
          AND type(r1) CONTAINS 'TOPICS'
          AND type(r2) CONTAINS 'TOPICS'
        WITH other, collect(DISTINCT t.name) AS shared_topics, count(DISTINCT t) AS cnt
        ORDER BY cnt DESC
        LIMIT $lim
        RETURN other, labels(other) AS labels, shared_topics, cnt
        """
        async with self._driver.session(database=database) as session:
            result = await session.run(query, hash=document_hash, lim=limit)
            records = [r async for r in result]
        nodes = []
        for rec in records:
            n = rec["other"]
            nodes.append(GraphNode(
                node_id=n.get("node_id", ""),
                name=n.get("name", ""),
                labels=rec["labels"],
                document_hash=n.get("document_hash", ""),
                post_type=n.get("post_type", ""),
                url=n.get("url", ""),
                site=n.get("site", ""),
                properties={
                    "shared_topics": rec["shared_topics"],
                    "topic_overlap": rec["cnt"],
                },
            ))
        return nodes

    async def get_topic_hierarchy(
        self, document_hash: str, database: str
    ) -> list[dict]:
        """Topic chain: topic → parent topic → grandparent (SUBTOPIC_OF)."""
        await self.connect()
        query = """
        MATCH (d:Document {document_hash: $hash})-[r]->(t:Topic)
        WHERE type(r) CONTAINS 'TOPICS'
        OPTIONAL MATCH path = (t)-[:SUBTOPIC_OF*1..3]->(parent:Topic)
        WITH t, [node IN nodes(path) | node.name] AS chain
        RETURN t.name AS topic, chain
        """
        async with self._driver.session(database=database) as session:
            result = await session.run(query, hash=document_hash)
            records = [r async for r in result]
        return [{"topic": r["topic"], "hierarchy": r["chain"] or []} for r in records]

    # ── aggregate expansion ───────────────────────────────────────────

    async def get_expanded_context(
        self,
        document_hash: str,
        database: str,
        max_relations: int = 30,
    ) -> GraphExpansionResult:
        """Full graph expansion for a single document (parallel sub-queries)."""
        direct_task = self.get_direct_relations(document_hash, database, max_relations)
        incoming_task = self.get_incoming_relations(document_hash, database, max_relations // 2)
        hierarchy_task = self.get_topic_hierarchy(document_hash, database)

        direct, incoming, hierarchy = await asyncio.gather(
            direct_task, incoming_task, hierarchy_task, return_exceptions=True
        )

        if isinstance(direct, Exception):
            logger.warning("get_direct_relations failed for %s: %s", document_hash, direct)
            direct = []
        if isinstance(incoming, Exception):
            logger.warning("get_incoming_relations failed for %s: %s", document_hash, incoming)
            incoming = []
        if isinstance(hierarchy, Exception):
            logger.warning("get_topic_hierarchy failed for %s: %s", document_hash, hierarchy)
            hierarchy = []

        return _aggregate_relations(document_hash, direct, incoming, hierarchy)

    async def batch_expand(
        self,
        document_hashes: list[str],
        database: str,
        max_relations_per_doc: int = 20,
    ) -> dict[str, GraphExpansionResult]:
        """Expand multiple documents in parallel."""
        tasks = [
            self.get_expanded_context(h, database, max_relations_per_doc)
            for h in document_hashes
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        expanded = {}
        for h, res in zip(document_hashes, results):
            if isinstance(res, Exception):
                logger.warning("Graph expansion failed for %s: %s", h, res)
                continue
            expanded[h] = res
        return expanded


# ── private helpers ───────────────────────────────────────────────────

def _url_variants(url: str) -> list[str]:
    """Generate www / non-www / trailing-slash permutations."""
    base = set()
    base.add(url)
    if "://www." in url:
        base.add(url.replace("://www.", "://"))
    else:
        base.add(url.replace("://", "://www."))
    expanded = set()
    for u in base:
        expanded.add(u)
        expanded.add(u.rstrip("/") + "/")
        expanded.add(u.rstrip("/"))
    return list(expanded)


def _record_to_relation(record) -> GraphRelation:
    return GraphRelation(
        source_hash=record["source_hash"] or "",
        source_name=record["source_name"] or "",
        relationship=record["rel_type"] or "",
        target_hash=record["target_hash"] or "",
        target_name=record["target_name"] or "",
        target_labels=record["target_labels"] or [],
        target_url=record["target_url"] or "",
        target_post_type=record["target_post_type"] or "",
    )


_CONTENT_LABELS = {"Blog", "Event", "Resource", "Course", "Updates",
                   "Newsletter", "Diplonews", "Project", "PressRelease"}


def _aggregate_relations(
    document_hash: str,
    direct: list[GraphRelation],
    incoming: list[GraphRelation],
    hierarchy: list[dict],
) -> GraphExpansionResult:
    """Classify relations into topics, people, actors, etc."""
    doc_name = ""
    topics, tags, people, actors, subtopic_of = [], [], [], [], []
    related_docs = []
    seen_names = set()

    for rel in direct:
        if not doc_name:
            doc_name = rel.source_name
        tl = set(rel.target_labels) - {"Document"}
        tname = rel.target_name or ""

        if "Topic" in rel.target_labels or "TopicBasket" in rel.target_labels:
            if tname and tname not in seen_names:
                topics.append(tname)
                seen_names.add(tname)
        elif "Tag" in rel.target_labels:
            if tname and tname not in seen_names:
                tags.append(tname)
                seen_names.add(tname)
        elif tl & {"Person", "Expert"}:
            if tname and tname not in seen_names:
                people.append(tname)
                seen_names.add(tname)
        elif "Actor" in rel.target_labels:
            if tname and tname not in seen_names:
                actors.append(tname)
                seen_names.add(tname)
        elif "Date" in rel.target_labels:
            pass
        elif tl & _CONTENT_LABELS:
            label_str = next(iter(tl & _CONTENT_LABELS), "")
            related_docs.append(GraphNode(
                node_id="", name=tname, labels=list(tl),
                document_hash=rel.target_hash,
                url=rel.target_url, post_type=rel.target_post_type,
            ))

    for h_entry in hierarchy:
        chain = h_entry.get("hierarchy", [])
        if len(chain) > 1:
            subtopic_of.append(" → ".join(chain))

    for rel in incoming:
        iname = rel.source_name or ""
        if iname and iname not in seen_names:
            seen_names.add(iname)

    return GraphExpansionResult(
        document_hash=document_hash,
        document_name=doc_name,
        relations=direct + incoming,
        topics=topics,
        related_documents=related_docs[:10],
        tags=tags[:20],
        people=people,
        actors=actors,
        subtopic_of=subtopic_of,
    )
