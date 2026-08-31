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
# import the GraphExpansionResult and Neo4jGraphClient classes from the neo4j_graph_client module
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
    # ── init ───────────────────────────────────────────────────────────
    # create a client instance for 
    def __init__(self, client: Neo4jGraphClient):
        self._client = client

    # ── hash resolution ───────────────────────────────────────────────
    # context: this will be used to hash the text to so it can be used to query exact content of the text.
    # description: make md5 hash of the text
    @staticmethod
    def _md5(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    # context: this will be used to make clean url - it will return a list of urls with different variations of the input url.
    # description: using url library make clean url from input url
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

    # context: this will be used to resolve the hashes from the documents.
    # description: for each doc URL, find parent_document_hash and make few variants of the url and hash them.
    # parameters: docs - list of documents
    # return: dictionary key is the parent_document_hash and value is hashed url variants.
    def resolve_hashes_from_docs(
        self, docs: list[Document]
    ) -> dict[str, list[str]]:
        """
        For each doc URL, compute candidate document_hash values.

        Returns { url: [hash_variant_1, hash_variant_2, ...] }
        The first variant is md5(original_url), rest are www/trailing-slash permutations.
        """
        # initialize the dictionary to store the url and the hashes
        url_to_hashes: dict[str, list[str]] = {}
        seen = set()

        # loop through the documents and get the url and the parent_document_hash
        for doc in docs:
        # get the url from the document metadata
            url = doc.metadata.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)

            # for that url, get the parent_document_hash from the document metadata
            pdh = doc.metadata.get("parent_document_hash", "")
            # if the parent_document_hash is not empty, add it to the dictionary
            if pdh:
                url_to_hashes[url] = [pdh]
                continue
            # make few variatns of the same url - www, non-www, trailing slash, etc.
            variants = self._url_variants(url)
            # for each variant, make md5 hash and add it to the dictionary
            url_to_hashes[url] = [self._md5(v) for v in variants]
        # return the dictionary
        return url_to_hashes

    # ── helpers ─────────────────────────────────────────────────────

    # context: this will in expand_documents function to pick the correct Neo4j database based on the document URL domain.
    # description: pick the correct Neo4j database based on the document URL domain.
    @staticmethod
    def _db_for_url(url: str, default_db: str) -> str:
        """Pick the correct Neo4j database based on the document URL domain."""
        if "dig.watch" in url:
            return get_neo4j_database("dig.watch")
        if "faicon.ai" in url:
            return get_neo4j_database("dig.watch")
        return default_db

    # ── main entry point ──────────────────────────────────────────────
    # context: this is called in graph_expansion_node.py to expand the documents retrieved from the graph - using docs site and max_relations(depth).
    async def expand_documents(
        self,
        docs: list[Document],
        site: str = "diplomacy.edu",
        max_relations: int | None = None,
    ) -> tuple[dict[str, GraphExpansionResult], float]:
        """
        Expand retrieved documents with knowledge graph context.

        Retrieval can return a mix of diplomacy.edu and dig.watch URLs.
        Each URL is resolved against the correct Neo4j database based on
        its domain, so both knowledge graphs are consulted in parallel.

        Returns:
            (url_to_expansion, elapsed_seconds)
            url_to_expansion maps doc URL → GraphExpansionResult
        """
        # ceck if graph expansion is enabled to countinue
        if not GRAPH_EXPANSION_ENABLED:
            return {}, 0.0
        # from config
        if max_relations is None:
            max_relations = GRAPH_EXPANSION_MAX_RELATIONS
        # for time mesure
        t0 = time.time()
        default_db = get_neo4j_database(site)
        # mmake a dictionary key is the url and value is the list of hashes.
        url_to_hashes = self.resolve_hashes_from_docs(docs)
        if not url_to_hashes:
            return {}, time.time() - t0

        # Group URLs by database
        # set db_urls dictionary 
        db_urls: dict[str, dict[str, list[str]]] = {}
        # put parent_document_hash and hashed url variants into db_urls dictionary.
        for url, hashes in url_to_hashes.items():
        # look up the correct Neo4j database based on the document URL domain.
            db = self._db_for_url(url, default_db)
        # if db exist, put url and hashes into db_urls dictionary.
        # if db does not exist, create a new dictionary with the db as the key and the url and hashes as the value.
            db_urls.setdefault(db, {})[url] = hashes

        # Resolve + expand each database group in parallel
        async def _resolve_db_group(
            database: str, group: dict[str, list[str]]
        ) -> dict[str, str]:
            """Resolve hashes for one database, returns url→hash."""
            confirmed: dict[str, str] = {}

            all_hashes = set()
            for hashes in group.values():
                all_hashes.update(hashes)

            # Phase 1 — hash verification
            try:
                found_set = await asyncio.wait_for(
                    self._verify_hashes(list(all_hashes), database),
                    timeout=GRAPH_EXPANSION_TIMEOUT * 0.4,
                )
            except asyncio.TimeoutError:
                logger.warning("Graph expansion: hash verification timed out for %s", database)
                return confirmed

            unresolved = []
            for url, hashes in group.items():
                matched = next((h for h in hashes if h in found_set), None)
                if matched:
                    confirmed[url] = matched
                else:
                    unresolved.append(url)

            # Phase 2 — exact URL fallback
            if unresolved:
                try:
                    url_resolved = await asyncio.wait_for(
                        self._resolve_by_url(unresolved, database),
                        timeout=GRAPH_EXPANSION_TIMEOUT * 0.3,
                    )
                    confirmed.update(url_resolved)
                except asyncio.TimeoutError:
                    logger.warning("Graph expansion: URL fallback timed out for %s", database)

            # Phase 2b — slug-based fuzzy fallback
            still_unresolved = [u for u in unresolved if u not in confirmed]
            if still_unresolved:
                try:
                    slug_resolved = await asyncio.wait_for(
                        self._resolve_by_slug(still_unresolved, database),
                        timeout=GRAPH_EXPANSION_TIMEOUT * 0.3,
                    )
                    confirmed.update(slug_resolved)
                except asyncio.TimeoutError:
                    logger.warning("Graph expansion: slug fallback timed out for %s", database)

            return confirmed

        # Run resolution for all databases in parallel
        resolve_tasks = {
            db: _resolve_db_group(db, group)
            for db, group in db_urls.items()
        }
        resolve_results = await asyncio.gather(*resolve_tasks.values(), return_exceptions=True)

        confirmed: dict[str, str] = {}         # url → hash
        url_to_db: dict[str, str] = {}         # url → database
        for db, result in zip(resolve_tasks.keys(), resolve_results):
            if isinstance(result, Exception):
                logger.warning("Graph expansion resolution failed for %s: %s", db, result)
                continue
            confirmed.update(result)
            for url in result:
                url_to_db[url] = db

        if not confirmed:
            elapsed = time.time() - t0
            logger.info("TIMING graph_expansion: %.3fs (no docs found in graph)", elapsed)
            return {}, elapsed

        # Phase 3 — batch expand, grouped by database
        db_to_hashes: dict[str, list[str]] = {}
        for url, h in confirmed.items():
            db = url_to_db[url]
            db_to_hashes.setdefault(db, []).append(h)

        expand_tasks = {
            db: asyncio.wait_for(
                self._client.batch_expand(list(set(hashes)), db, max_relations),
                timeout=GRAPH_EXPANSION_TIMEOUT,
            )
            for db, hashes in db_to_hashes.items()
        }
        expand_results = await asyncio.gather(*expand_tasks.values(), return_exceptions=True)

        all_expansions: dict[str, GraphExpansionResult] = {}
        for db, result in zip(expand_tasks.keys(), expand_results):
            if isinstance(result, Exception):
                logger.warning("Graph expansion batch_expand failed for %s: %s", db, result)
                continue
            all_expansions.update(result)

        url_expansions: dict[str, GraphExpansionResult] = {}
        for url, h in confirmed.items():
            if h in all_expansions:
                url_expansions[url] = all_expansions[h]

        elapsed = time.time() - t0
        dbs_used = list(db_to_hashes.keys())
        logger.info(
            "TIMING graph_expansion: %.3fs (%d docs resolved, %d expanded, dbs=%s)",
            elapsed, len(confirmed), len(url_expansions), dbs_used,
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

    @staticmethod
    def format_inline_source_context(
        doc: Document,
        expansions: dict[str, GraphExpansionResult],
    ) -> str:
        """Compact single-line graph annotation for one source.

        Returns a string like:
            [Graph: Type: blog | Date: Feb 2019 | Topics: AI, Cybersecurity | ...]
        Returns '' if no expansion exists for this document's URL.
        """
        url = doc.metadata.get("url", "")
        exp = expansions.get(url)
        if not exp:
            return ""

        parts: list[str] = []
        if exp.post_type:
            parts.append(f"Type: {exp.post_type}")
        if exp.date_published:
            parts.append(f"Date: {exp.date_published}")
        if exp.topics:
            parts.append(f"Topics: {', '.join(exp.topics[:6])}")
        if exp.countries:
            parts.append(f"Countries: {', '.join(exp.countries[:5])}")
        if exp.people:
            parts.append(f"People: {', '.join(exp.people[:5])}")
        if exp.actors:
            parts.append(f"Orgs: {', '.join(exp.actors[:5])}")
        if exp.processes:
            parts.append(f"Processes: {', '.join(exp.processes[:4])}")
        if exp.related_documents:
            names = []
            for rd in exp.related_documents[:4]:
                clean = [l for l in rd.labels if l != "Document"]
                tag = f" [{clean[0]}]" if clean else ""
                names.append(f"{rd.name[:40]}{tag}")
            parts.append(f"Related: {'; '.join(names)}")

        if not parts:
            return ""
        return "  [Graph: " + " | ".join(parts) + "]"

    # ── retrieval boost ────────────────────────────────────────────────

    @staticmethod
    def compute_graph_boost_scores(
        docs: list[Document],
        expansions: dict[str, GraphExpansionResult],
        question: str,
        alpha: float = 0.15,
    ) -> list[float]:
        """Score each doc by how well its graph neighbours match the question.

        Returns a list of floats (one per doc) in [0, 1] representing the
        graph-derived relevance boost.  The caller adds
        ``alpha * boost`` to the existing retrieval/reranker score.
        """
        # question split into words and remove words less than 3 characters
        q_tokens = set(question.lower().split())
        q_tokens = {t for t in q_tokens if len(t) >= 3}
        boosts: list[float] = []

        # loop through the documents get url and expansion result for that document.
        for doc in docs:
            url = doc.metadata.get("url", "")
            exp = expansions.get(url)
        # if no expansion result, add 0.0 to the boosts list
            if not exp:
                boosts.append(0.0)
                continue
        # create a set of tokens from the expansion result topics, people, actors, countries, and processes.
        # same principle as question tokens split and lowercase.
            graph_tokens: set[str] = set()
            for name in exp.topics + exp.people + exp.actors + exp.countries + exp.processes:
                graph_tokens.update(name.lower().split())

            if not graph_tokens or not q_tokens:
                boosts.append(0.0)
                continue

            # calculate the overlap between the question tokens and the graph tokens.
            overlap = len(q_tokens & graph_tokens)
            # calculate the score based on the overlap and the length of the question tokens.
            score = min(overlap / max(len(q_tokens), 1), 1.0)

            connectivity = min(
                (len(exp.topics) + len(exp.people) + len(exp.actors)) / 10.0,
                1.0,
            )
            boosts.append(0.7 * score + 0.3 * connectivity)

        return boosts

    @staticmethod
    def compute_coherence_scores(
        docs: list[Document],
        expansions: dict[str, GraphExpansionResult],
    ) -> list[float]:
        """Score each doc by how many graph entities it shares with other docs.

        High coherence = this document's topics/people overlap with the rest
        of the retrieved set, suggesting a tight thematic cluster.
        """
        all_entities: list[set[str]] = []
        for doc in docs:
            url = doc.metadata.get("url", "")
            exp = expansions.get(url)
            entities: set[str] = set()
            if exp:
                entities.update(t.lower() for t in exp.topics)
                entities.update(p.lower() for p in exp.people)
                entities.update(a.lower() for a in exp.actors)
            all_entities.append(entities)

        global_pool: set[str] = set()
        for ent in all_entities:
            global_pool.update(ent)

        if not global_pool:
            return [0.0] * len(docs)

        scores: list[float] = []
        for i, ent_i in enumerate(all_entities):
            if not ent_i:
                scores.append(0.0)
                continue
            others = set()
            for j, ent_j in enumerate(all_entities):
                if j != i:
                    others.update(ent_j)
            shared = len(ent_i & others)
            scores.append(min(shared / max(len(ent_i), 1), 1.0))
        return scores

    @staticmethod
    def format_conditional_hint(
        doc: Document,
        expansions: dict[str, GraphExpansionResult],
        question: str,
    ) -> str:
        """Generate a short hint ONLY when the document text lacks info the graph has.

        Unlike ``format_inline_source_context`` which always dumps all metadata,
        this checks whether the document text already covers the graph entities.
        Returns '' if the text already contains the relevant information.
        """
        url = doc.metadata.get("url", "")
        exp = expansions.get(url)
        if not exp:
            return ""

        text_lower = doc.page_content.lower()
        q_lower = question.lower()

        hints: list[str] = []

        for person in exp.people[:3]:
            if person.lower() not in text_lower and any(
                tok in q_lower for tok in person.lower().split() if len(tok) >= 4
            ):
                role_parts = [person]
                if exp.actors:
                    role_parts.append(f"affiliated with {exp.actors[0]}")
                hints.append(", ".join(role_parts))

        for actor in exp.actors[:2]:
            if actor.lower() not in text_lower and any(
                tok in q_lower for tok in actor.lower().split() if len(tok) >= 4
            ):
                hints.append(actor)

        missing_topics = []
        for topic in exp.topics[:4]:
            if topic.lower() not in text_lower and any(
                tok in q_lower for tok in topic.lower().split() if len(tok) >= 4
            ):
                missing_topics.append(topic)
        if missing_topics:
            hints.append(f"related topics: {', '.join(missing_topics)}")

        if not hints:
            return ""
        return f"  (Graph context: {'; '.join(hints)})"

    # ── subgraph data for frontend ───────────────────────────────────

    SUBGRAPH_COLORS = {
        "Source": "#00879a", "Topic": "#00879a", "TopicBasket": "#006d7d",
        "Person": "#d65799", "Expert": "#d65799", "Actor": "#486284",
        "Country": "#b02b2c", "Date": "#8d9191", "Tag": "#4ecac2",
        "Event": "#f7941e", "Process": "#306e7e",
    }

    @staticmethod
    # context: this is called inside graph_expansion_node.py by subgraph var - it will be used to display the subgraph in the UI
    # inputs given: 
    # - question is from the user question - saved in state['_question']
    # - docs is the list of documents - that are fetched and re-ranked by graph boost score and also if called by boost_inline 
    # - expansions is the expansions from the last call to expand_documents - db and hashed url variants - saved in service._last_expansions
    # return: dict - this is the subgraph data for the question, modified docs, and expansions
    def build_subgraph_data(
        question: str,
        docs: list[Document],
        expansions: dict[str, "GraphExpansionResult"],
    ) -> dict:
        """Build a lightweight vis-network graph for the frontend.

        Returns {"nodes": [...], "edges": [...]} suitable for embedding
        in the API response and rendering with vis-network on the client.
        """
        # initialize the nodes, edges, and node_ids
        nodes, edges, node_ids = [], [], set()
        # get the colors pallete for graph from config above
        colors = GraphExpansionService.SUBGRAPH_COLORS

        # create a question node with the question label - with params for HTML display
        q_id = "question"
        nodes.append({"id": q_id, "label": question, "type": "Question", "color": "#28282a", "size": 28})
        node_ids.add(q_id)
        # loop through the documents - get the url and title - and the expansion result for that document.
        for i, doc in enumerate(docs):
            url = doc.metadata.get("url", "")
            title = doc.metadata.get("title", "Unknown")
            exp = expansions.get(url)
            expanded = exp is not None
        # create a source node with the url and title - with params for HTML display
            src_id = f"src_{i}"
            nodes.append({
                "id": src_id,
                "label": f"[{i+1}] {title}",
                "type": "Source",
                "color": "#00879a" if expanded else "#5f8789",
                "size": 14,
                "url": url,
                "expanded": expanded,
            })
            node_ids.add(src_id)
            edges.append({"from": q_id, "to": src_id, "label": f"[{i+1}]", "hop": 1})

            if not exp:
                continue

            label_prefixes = {"Topic": "t_", "TopicBasket": "t_", "Person": "p_", "Actor": "a_", "Country": "c_", "Process": "t_", "Expert": "p_", "Event": "e_", "Tag": "tag_", "Date": "d_", "Blog": "doc_", "Resource": "doc_", "Course": "doc_", "Page": "doc_"}
            hop1_added = 0
            for rel in exp.relations:
                if hop1_added >= 10:
                    break
                tgt_labels = [l for l in (rel.target_labels or []) if l != "Document"]
                tgt_type = next((l for l in tgt_labels if l in colors), None)
                if not tgt_type and rel.target_post_type:
                    pt = rel.target_post_type.capitalize()
                    if pt in colors:
                        tgt_type = pt
                if not tgt_type:
                    tgt_type = "Topic"
                prefix = label_prefixes.get(tgt_type, "n_")
                nid = f"{prefix}{rel.target_name}"[:60]
                is_new = nid not in node_ids
                if is_new:
                    node_data = {"id": nid, "label": rel.target_name, "type": tgt_type, "color": colors.get(tgt_type, "#8d9191"), "size": 7}
                    if rel.target_url:
                        node_data["url"] = rel.target_url
                    nodes.append(node_data)
                    node_ids.add(nid)
                    hop1_added += 1
                rel_label = rel.relationship.replace("_", " ").lower()[:20]
                edges.append({"from": src_id, "to": nid, "label": rel_label, "hop": 2})

        hop2_edge_set = set((e["from"], e["to"]) for e in edges)
        all_prefixes = ["t_", "p_", "a_", "c_", "e_", "tag_", "d_", "n_"]

        def _find_node_id(name):
            for pfx in all_prefixes:
                nid = f"{pfx}{name}"[:60]
                if nid in node_ids:
                    return nid
            return None

        for exp in expansions.values():
            if not exp:
                continue
            for rel in exp.relations:
                from_id = _find_node_id(rel.source_name)
                to_id = _find_node_id(rel.target_name)
                if from_id and to_id and from_id != to_id:
                    edge_key = (from_id, to_id)
                    if edge_key not in hop2_edge_set and edge_key not in set((e["from"], e["to"]) for e in edges if e.get("hop") == 3):
                        rel_label = rel.relationship.replace("_", " ").lower()[:20]
                        edges.append({"from": from_id, "to": to_id, "label": rel_label, "hop": 3})

        return {"nodes": nodes, "edges": edges}

    @staticmethod
    # build_subgraph_html - build the subgraph html for the question, modified docs, and expansions
    def build_subgraph_html(
        question: str,
        docs: list[Document],
        expansions: dict[str, "GraphExpansionResult"],
    ) -> str:
        """Self-contained HTML snippet for embedding a subgraph visualization.

        Returns a complete ``<div>`` + ``<script>`` block that the frontend
        can inject via innerHTML or iframe srcdoc.  Includes vis-network
        from CDN — no additional dependencies needed on the client.
        """
        import json as _json
        graph = GraphExpansionService.build_subgraph_data(question, docs, expansions)
        nodes_json = _json.dumps(graph["nodes"])
        edges_json = _json.dumps(graph["edges"])
        n_nodes = len(graph["nodes"])
        n_edges = len(graph["edges"])

        legend_types = sorted(set(n["type"] for n in graph["nodes"]))
        colors = GraphExpansionService.SUBGRAPH_COLORS
        legend_html = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:12px;font-size:11px;color:#5a5a5a">'
            f'<span style="width:8px;height:8px;border-radius:50%;background:{colors.get(lt,"#8d9191")}"></span>{lt}</span>'
            for lt in legend_types
        )

        return f'''<div class="graph-subgraph" style="background:#f4f6f8;border:1px solid #dde0e3;border-radius:10px;overflow:hidden;font-family:'DM Sans',system-ui,sans-serif">
  <div style="padding:8px 14px;border-bottom:1px solid #e5e7eb;display:flex;justify-content:space-between;align-items:center">
    <span style="font-size:13px;font-weight:700;color:#00879a">Knowledge Subgraph</span>
    <span style="font-size:11px;color:#8d9191">{n_nodes} nodes · {n_edges} edges</span>
  </div>
  <div id="graphViz" style="width:100%;height:350px;background:#f4f6f8"></div>
  <div style="padding:6px 14px;border-top:1px solid #e5e7eb">{legend_html}</div>
</div>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<script>
(function(){{
  const NODES={nodes_json};
  const EDGES={edges_json};
  const ds=new vis.DataSet(NODES.map(n=>{{
    const fc=n.type==='Question'?'#fff':'#000';
    return{{id:n.id,label:n.label,color:{{background:n.color,border:n.color}},
      size:n.size,shape:n.type==='Question'?'diamond':'dot',
      font:{{color:fc,size:11,strokeWidth:3,strokeColor:'#fff'}},
      borderWidth:n.expanded?2:0,_raw:n}};
  }}));
  const es=new vis.DataSet(EDGES.map((e,i)=>{{
    return{{id:'e'+i,from:e.from,to:e.to,label:e.label||'',
      font:{{size:8,color:'#888',strokeWidth:0}},
      color:{{color:'#c8ced6',opacity:0.6}},width:1,
      arrows:{{to:{{enabled:true,scaleFactor:0.25}}}},smooth:{{type:'continuous'}}}};
  }}));
  const net=new vis.Network(document.getElementById('graphViz'),{{nodes:ds,edges:es}},{{
    physics:{{solver:'forceAtlas2Based',forceAtlas2Based:{{gravitationalConstant:-60,centralGravity:0.005,springLength:100,springConstant:0.02,damping:0.4,avoidOverlap:0.5}},stabilization:{{iterations:300,fit:true}},maxVelocity:30}},
    interaction:{{hover:true,zoomView:true,dragView:true}}
  }});
  net.once('stabilizationIterationsDone',()=>{{net.setOptions({{physics:false}});net.fit();}});
}})();
</script>'''

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

    async def _resolve_by_slug(
        self, urls: list[str], database: str
    ) -> dict[str, str]:
        """Fuzzy fallback: match by URL slug when exact URL differs."""
        tasks = [self._client.get_document_by_slug(url, database) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        resolved = {}
        for url, res in zip(urls, results):
            if not isinstance(res, Exception) and res is not None and res.document_hash:
                resolved[url] = res.document_hash
                logger.info("Slug fallback matched: %s → %s", url[:60], res.url[:60])
        return resolved
