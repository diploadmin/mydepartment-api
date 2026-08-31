"""
Graph Expansion Service — enriches RAG retrieval results with Neo4j knowledge.

Main workflow:
    1. Receives list[Document] from the retrieval pipeline (post-reranking)
    2. Resolves document_hash for each doc (md5 of URL, with fallback to Neo4j URL lookup)
    3. Batch-queries Neo4j for topics, people, actors, related content
    4. Formats the graph context as an additional block in the LLM prompt
    5. Optionally enriches Document.metadata with graph information

Integration points:
    - Called from direct_retrieval_node() in graph.py (Option A)
    - Or as a standalone LangGraph node (see graph_expansion_node.py, Option C)
"""

## THIS CODE IS NOT USED IN THE CHATBOT-API-DEV PROJECT ##

import hashlib
import asyncio
import time
import logging
from typing import Optional

from langchain_core.documents import Document

from app.core.neo4j_config import (
    GRAPH_EXPANSION_ENABLED,
    GRAPH_EXPANSION_TIMEOUT,
    GRAPH_EXPANSION_MAX_RELATIONS,
    get_neo4j_database,
)
from app.ai.ai_services.neo4j_graph_client import (
    Neo4jGraphClient,
    GraphExpansionResult,
)

logger = logging.getLogger(__name__)


class GraphExpansionService:
    """
    Orchestrates the full graph expansion flow:
        docs → hash resolution → Neo4j queries → formatted context
    """

    def __init__(self, client: Neo4jGraphClient):
        self._client = client

    # ── hash resolution ───────────────────────────────────────────────

    @staticmethod
    def _md5(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    @staticmethod
    def _url_variants(url: str) -> list[str]:
        """www / non-www / trailing slash permutations."""
        base = {url}
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

    def resolve_hashes_from_docs(
        self, docs: list[Document]
    ) -> dict[str, list[str]]:
        """
        For each doc URL, compute candidate document_hash values.

        Returns { url: [hash_variant_1, hash_variant_2, ...] }
        The first variant is md5(original_url), rest are www/trailing-slash permutations.
        """
        url_to_hashes: dict[str, list[str]] = {}
        seen = set()

        for doc in docs:
            url = doc.metadata.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)

            pdh = doc.metadata.get("parent_document_hash", "")
            if pdh:
                url_to_hashes[url] = [pdh]
                continue

            variants = self._url_variants(url)
            url_to_hashes[url] = [self._md5(v) for v in variants]

        return url_to_hashes

    # ── main entry point ──────────────────────────────────────────────

    async def expand_documents(
        self,
        docs: list[Document],
        site: str = "diplomacy.edu",
        max_relations: int | None = None,
    ) -> tuple[dict[str, GraphExpansionResult], float]:
        """
        Expand retrieved documents with knowledge graph context.

        Returns:
            (url_to_expansion, elapsed_seconds)
            url_to_expansion maps doc URL → GraphExpansionResult
        """
        if not GRAPH_EXPANSION_ENABLED:
            return {}, 0.0

        if max_relations is None:
            max_relations = GRAPH_EXPANSION_MAX_RELATIONS

        t0 = time.time()
        database = get_neo4j_database(site)

        url_to_hashes = self.resolve_hashes_from_docs(docs)
        if not url_to_hashes:
            return {}, time.time() - t0

        # Phase 1 — verify which hashes exist in Neo4j
        confirmed: dict[str, str] = {}   # url → confirmed hash
        unresolved_urls: list[str] = []

        all_candidate_hashes = set()
        for hashes in url_to_hashes.values():
            all_candidate_hashes.update(hashes)

        try:
            found_set = await asyncio.wait_for(
                self._verify_hashes(list(all_candidate_hashes), database),
                timeout=GRAPH_EXPANSION_TIMEOUT * 0.4,
            )
        except asyncio.TimeoutError:
            logger.warning("Graph expansion: hash verification timed out")
            return {}, time.time() - t0

        for url, hashes in url_to_hashes.items():
            matched = next((h for h in hashes if h in found_set), None)
            if matched:
                confirmed[url] = matched
            else:
                unresolved_urls.append(url)

        # Phase 2 — URL-based fallback for unresolved docs
        if unresolved_urls:
            try:
                url_resolved = await asyncio.wait_for(
                    self._resolve_by_url(unresolved_urls, database),
                    timeout=GRAPH_EXPANSION_TIMEOUT * 0.3,
                )
                confirmed.update(url_resolved)
            except asyncio.TimeoutError:
                logger.warning("Graph expansion: URL fallback timed out")

        if not confirmed:
            elapsed = time.time() - t0
            logger.info("TIMING graph_expansion: %.3fs (no docs found in graph)", elapsed)
            return {}, elapsed

        # Phase 3 — batch expand all confirmed documents
        unique_hashes = list(set(confirmed.values()))
        try:
            expansions = await asyncio.wait_for(
                self._client.batch_expand(unique_hashes, database, max_relations),
                timeout=GRAPH_EXPANSION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("Graph expansion: batch expand timed out")
            expansions = {}

        url_expansions: dict[str, GraphExpansionResult] = {}
        for url, h in confirmed.items():
            if h in expansions:
                url_expansions[url] = expansions[h]

        elapsed = time.time() - t0
        logger.info(
            "TIMING graph_expansion: %.3fs (%d docs resolved, %d expanded)",
            elapsed, len(confirmed), len(url_expansions),
        )
        return url_expansions, elapsed

    # ── formatting for LLM ────────────────────────────────────────────

    @staticmethod
    def format_graph_context(
        docs: list[Document],
        expansions: dict[str, GraphExpansionResult],
    ) -> str:
        """
        Produce an LLM-readable text block summarizing graph knowledge.

        Injected after the retrieval results in the ToolMessage, e.g.:
            tool_content = retrieval_text + format_graph_context(...)
        """
        if not expansions:
            return ""

        sections: list[str] = []

        for doc in docs:
            url = doc.metadata.get("url", "")
            exp = expansions.get(url)
            if not exp:
                continue

            title = doc.metadata.get("title", "") or exp.document_name or "Unknown"
            lines: list[str] = []

            if exp.topics:
                lines.append(f"  Topics: {', '.join(exp.topics)}")

            if exp.subtopic_of:
                lines.append(f"  Topic hierarchy: {'; '.join(exp.subtopic_of)}")

            if exp.people:
                lines.append(f"  Related people: {', '.join(exp.people)}")

            if exp.actors:
                lines.append(f"  Related organizations: {', '.join(exp.actors)}")

            if exp.related_documents:
                names = []
                for rd in exp.related_documents[:5]:
                    clean_labels = [l for l in rd.labels if l != "Document"]
                    tag = f" [{clean_labels[0]}]" if clean_labels else ""
                    names.append(f"{rd.name}{tag}")
                lines.append(f"  Related content: {'; '.join(names)}")

            if exp.tags:
                lines.append(f"  Tags: {', '.join(exp.tags[:10])}")

            if lines:
                header = f'Graph context for "{title}":'
                sections.append(header + "\n" + "\n".join(lines))

        if not sections:
            return ""

        return (
            "\n\n--- KNOWLEDGE GRAPH CONTEXT ---\n\n"
            + "\n\n".join(sections)
        )

    # ── metadata enrichment ───────────────────────────────────────────

    @staticmethod
    def enrich_document_metadata(
        docs: list[Document],
        expansions: dict[str, GraphExpansionResult],
    ) -> list[Document]:
        """Attach graph data to each Document's metadata dict."""
        for doc in docs:
            url = doc.metadata.get("url", "")
            exp = expansions.get(url)
            if not exp:
                doc.metadata["_graph_expanded"] = False
                continue

            doc.metadata["_graph_expanded"] = True
            doc.metadata["_document_hash"] = exp.document_hash
            doc.metadata["_graph_topics"] = exp.topics
            doc.metadata["_graph_tags"] = exp.tags
            doc.metadata["_graph_people"] = exp.people
            doc.metadata["_graph_actors"] = exp.actors
            doc.metadata["_graph_related_count"] = len(exp.related_documents)
            if exp.related_documents:
                doc.metadata["_graph_related"] = [
                    {"name": rd.name, "url": rd.url, "type": rd.post_type}
                    for rd in exp.related_documents[:5]
                ]
        return docs

    # ── private helpers ───────────────────────────────────────────────

    async def _verify_hashes(
        self, hashes: list[str], database: str
    ) -> set[str]:
        """Check which hashes exist as Document nodes. Returns set of found hashes."""
        await self._client.connect()
        query = """
        UNWIND $hashes AS h
        MATCH (d:Document {document_hash: h})
        RETURN d.document_hash AS hash
        """
        async with self._client._driver.session(database=database) as session:
            result = await session.run(query, hashes=hashes)
            records = [r async for r in result]
        return {r["hash"] for r in records if r["hash"]}

    async def _resolve_by_url(
        self, urls: list[str], database: str
    ) -> dict[str, str]:
        """Fallback: look up documents by URL when hash doesn't match."""
        tasks = [self._client.get_document_by_url(url, database) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        resolved = {}
        for url, res in zip(urls, results):
            if not isinstance(res, Exception) and res is not None and res.document_hash:
                resolved[url] = res.document_hash
        return resolved
