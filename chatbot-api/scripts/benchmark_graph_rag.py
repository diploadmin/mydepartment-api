#!/usr/bin/env -S python3 -u
"""
Graph RAG A/B Benchmark — standalone script.

Compares baseline RAG (Weaviate retrieval → LLM) vs graph-enriched RAG
(Weaviate retrieval → Neo4j expansion → LLM) using real production
questions from MongoDB.

Usage:
    cd chatbot-api
    python scripts/benchmark_graph_rag.py                  # 100 questions
    python scripts/benchmark_graph_rag.py --n 20           # first 20
    python scripts/benchmark_graph_rag.py --n 5 --no-judge # skip LLM judge
    python scripts/benchmark_graph_rag.py --questions "What is AI?" "What is diplomacy?"

All credentials are read from the repo-level .env file (or ENV_FILE override).
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

import argparse
import asyncio
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from starlette.config import Config

# ── Load secrets / config from .env ───────────────────────────────────
env_file = os.getenv("ENV_FILE", "../.env")
_config = Config(env_file)

# Neo4j
NEO4J_URI = _config("NEO4J_URI", cast=str, default="bolt://localhost:7687")
NEO4J_USER = _config("NEO4J_USER", cast=str, default="neo4j")
NEO4J_PASS = _config("NEO4J_PASS", cast=str)
NEO4J_DATABASE = _config("NEO4J_DATABASE_DIPLO", cast=str, default="weaviatediplo")

# Weaviate (parse WV_CLIENT_URL for host/port)
WV_CLIENT_URL = _config("WV_CLIENT_URL", cast=str)
_wv_parsed = urlparse(WV_CLIENT_URL)
WEAVIATE_HOST = _wv_parsed.hostname
if not WEAVIATE_HOST:
    raise SystemExit("WV_CLIENT_URL must include a hostname")
if _wv_parsed.port is None:
    raise SystemExit("WV_CLIENT_URL must include a port, e.g. http://host:8591")
WEAVIATE_PORT = _wv_parsed.port
WEAVIATE_GRPC_PORT = _config("WV_GRPC_PORT", cast=int, default=0)
if WEAVIATE_GRPC_PORT <= 0:
    raise SystemExit("WV_GRPC_PORT must be set in .env")
WEAVIATE_API_KEY = _config("WV_KEY", cast=str)

# MongoDB
MONGO_URI = _config("DB_CONNECTION", cast=str)
MONGO_DB = _config("BENCHMARK_MONGO_DB", cast=str, default="chatbot_humainism_ai")

# LLM (local TEI/vLLM-compatible endpoint)
LLM_URL = _config("LOCAL_LLM_URL", cast=str)
LLM_MODEL = _config("LOCAL_LLM_MODEL", cast=str)
LLM_KEY = _config("LOCAL_LLM_KEY", cast=str)

# Embeddings (local TEI endpoint)
EMBEDDING_URL = _config("LOCAL_EMBEDDING_URL", cast=str)
EMBEDDING_KEY = _config("LOCAL_EMBEDDING_KEY", cast=str)

# ── Mirror production env BEFORE any app imports ──────────────────────
# The app layer expects these in os.environ; only non-sensitive keys
# have hardcoded defaults, secrets are pulled from .env above.
_PROD_ENV = {
    "WEBSITE_NAME": _config("WEBSITE_NAME", cast=str, default="humainism.ai"),
    "HOST": "localhost",
    "PORT": "8561",
    "SECRET_KEY": _config("SECRET_KEY", cast=str, default="benchmarkscript"),
    "DB_CONNECTION": MONGO_URI,
    "DB_USER": _config("DB_USER", cast=str, default="admin"),
    "DB_PWD": _config("DB_PWD", cast=str, default=""),
    "DB_NAME": _config("DB_NAME", cast=str, default="admin"),
    "OPENAI_KEY": _config("OPENAI_KEY", cast=str, default="unused"),
    "OPENAI_ORGANIZATION": _config("OPENAI_ORGANIZATION", cast=str, default="unused"),
    "WV_CLIENT_URL": WV_CLIENT_URL,
    "WV_GRPC_PORT": str(WEAVIATE_GRPC_PORT),
    "WV_KEY": WEAVIATE_API_KEY,
    "INDEX_NAME": _config("INDEX_NAME", cast=str, default="DiploParagraph"),
    "INDEX_NAME2": _config("INDEX_NAME2", cast=str, default="DiploParagraph"),
    "EMBEDDING_PROVIDER": _config("EMBEDDING_PROVIDER", cast=str, default="local"),
    "LOCAL_EMBEDDING_URL": EMBEDDING_URL,
    "LOCAL_EMBEDDING_KEY": EMBEDDING_KEY,
    "RETRIEVAL_MODE": _config("RETRIEVAL_MODE", cast=str, default="sentence"),
    "HYBRID_ALPHA": _config("HYBRID_ALPHA", cast=str, default="0.75"),
    "SENTENCE_RETRIEVAL_K": _config("SENTENCE_RETRIEVAL_K", cast=str, default="200"),
    "SENTENCE_INDEX_NAME": _config("SENTENCE_INDEX_NAME", cast=str, default="DiploChunk"),
    "USE_CONTEXTUAL_COLLECTIONS": _config("USE_CONTEXTUAL_COLLECTIONS", cast=str, default="true"),
    "USE_SENTENCE_RETRIEVAL": _config("USE_SENTENCE_RETRIEVAL", cast=str, default="True"),
    "USE_SENTENCE_BLOCKLIST": _config("USE_SENTENCE_BLOCKLIST", cast=str, default="True"),
    "RETRIEVAL_CHUNKS": _config("RETRIEVAL_CHUNKS", cast=str, default="8"),
    "USE_RERANKER": _config("USE_RERANKER", cast=str, default="True"),
    "RERANKER_URL": _config("RERANKER_URL", cast=str, default="https://tei-reranker.diplomacy.edu"),
    "RERANKER_API_KEY": _config("RERANKER_API_KEY", cast=str),
    "RERANKER_TOP_K": _config("RERANKER_TOP_K", cast=str, default="25"),
    "SENTENCE_RERANKER_TOP_K": _config("SENTENCE_RERANKER_TOP_K", cast=str, default="50"),
    "TWOPHASE_RERANKER_TOP_K": _config("TWOPHASE_RERANKER_TOP_K", cast=str, default="40"),
    "FORCE_INCLUDE_H1": _config("FORCE_INCLUDE_H1", cast=str, default="True"),
    "FORCE_INCLUDE_SECTION": _config("FORCE_INCLUDE_SECTION", cast=str, default="True"),
    "FORCE_INCLUDE_OVERLAP": _config("FORCE_INCLUDE_OVERLAP", cast=str, default="0.6"),
    "FORCE_INCLUDE_MAX": _config("FORCE_INCLUDE_MAX", cast=str, default="7"),
    "HEADING_BOOST_WEIGHT": _config("HEADING_BOOST_WEIGHT", cast=str, default="0.5"),
    "HEADING_INJECT_MIN_SIM": _config("HEADING_INJECT_MIN_SIM", cast=str, default="0.65"),
    "HEADING_INJECT_MAX": _config("HEADING_INJECT_MAX", cast=str, default="10"),
    "HEADING_INJECT_MIN_SENTS": _config("HEADING_INJECT_MIN_SENTS", cast=str, default="5"),
    "SENTENCE_CE_CANDIDATES": _config("SENTENCE_CE_CANDIDATES", cast=str, default="3"),
    "SENTENCE_CE_MAX_GROUPS": _config("SENTENCE_CE_MAX_GROUPS", cast=str, default="20"),
    "SENTENCE_HIGHLIGHT_MODE": _config("SENTENCE_HIGHLIGHT_MODE", cast=str, default="multi"),
    "MAX_SECTIONS_PER_URL": _config("MAX_SECTIONS_PER_URL", cast=str, default="3"),
    "PARAGRAPH_RETRIEVAL_K": _config("PARAGRAPH_RETRIEVAL_K", cast=str, default="25"),
}
for k, v in _PROD_ENV.items():
    os.environ.setdefault(k, v)

import weaviate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pymongo import MongoClient

from app.ai.ai_services.retrievers.factory import create_retriever
from app.ai.ai_services.reranker import TEIReranker

SYSTEM_PROMPT = (
    "You are Diplorene, a helpful AI assistant specializing in diplomacy, "
    "internet governance, and digital policy. Answer based ONLY on the provided "
    "context. Cite sources as [1], [2], etc. If the context doesn't contain "
    "enough information, say so honestly."
)

JUDGE_PROMPT = """You are an expert evaluator comparing two answers to the same question.
Both answers were generated by a RAG system using retrieved documents as context.

Question: {question}

=== Answer A ===
{answer_a}

=== Answer B ===
{answer_b}

Evaluate EACH answer on these criteria (1-5 scale):
1. **Completeness**: How thoroughly does it address all aspects of the question?
2. **Accuracy**: Does it stick to information from the provided context without hallucinating?
3. **Specificity**: Does it provide concrete details, names, dates, examples (vs vague generalities)?
4. **Source_usage**: How well does it cite and reference sources?

Then decide an overall **winner**: "A", "B", or "TIE".

Respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{{
  "scores_a": {{"completeness": N, "accuracy": N, "specificity": N, "source_usage": N}},
  "scores_b": {{"completeness": N, "accuracy": N, "specificity": N, "source_usage": N}},
  "winner": "A" or "B" or "TIE",
  "reasoning": "1-2 sentence explanation"
}}"""


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class ABResult:
    question: str
    baseline_answer: str = ""
    baseline_sources: list = field(default_factory=list)
    t_baseline_retrieval: float = 0.0
    t_baseline_llm: float = 0.0
    t_baseline_total: float = 0.0
    graph_answer: str = ""
    graph_sources: list = field(default_factory=list)
    graph_context_text: str = ""
    t_graph_retrieval: float = 0.0
    t_graph_expansion: float = 0.0
    t_graph_llm: float = 0.0
    t_graph_total: float = 0.0
    n_expanded: int = 0
    judge_verdict: str = ""
    judge_reasoning: str = ""
    judge_scores: dict = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────

def load_production_questions(n: int, seed: int = 42) -> list[str]:
    """Load unique questions from MongoDB production."""
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[MONGO_DB]
    raw = list(db.messages.find().sort("timestamp", -1))
    client.close()

    seen = {}
    for m in raw:
        q = m.get("message", "").strip()
        key = q.lower()
        if key and len(q) > 10 and key not in seen:
            seen[key] = q

    all_q = list(seen.values())
    print(f"MongoDB: {len(raw)} total messages, {len(all_q)} unique (>10 chars)")

    n = min(n, len(all_q))
    random.seed(seed)
    selected = random.sample(all_q, n)
    return selected


GRAPH_FORMATS = ["none", "inline", "appended", "retrieval_boost", "rerank", "hint", "all", "boost_inline"]


def retrieve_with_production_pipeline(retriever, reranker, question: str, final_k: int = 8):
    """Use the exact same retrieval pipeline as the production chatbot."""
    candidates = retriever.invoke(question)
    if reranker and len(candidates) > final_k:
        candidates = reranker.rerank(question, candidates, top_k=len(candidates))
    return candidates[:final_k]


async def retrieve_with_graph_boost(
    retriever, reranker, graph_service, question: str,
    final_k: int = 8, pool_k: int = 16, alpha: float = 0.15,
):
    """Retrieval where graph connectivity boosts candidate ranking.

    1. Retrieve a larger pool (pool_k) via standard pipeline
    2. Graph-expand all candidates
    3. Compute graph boost scores per candidate
    4. Re-sort by  reranker_score + alpha * graph_boost
    5. Return top final_k — LLM sees NO graph metadata
    """
    from app.ai.ai_services.graph_expansion import GraphExpansionService

    candidates = retriever.invoke(question)
    if reranker and len(candidates) > pool_k:
        candidates = reranker.rerank(question, candidates, top_k=len(candidates))
    pool = candidates[:pool_k]

    expansions, _ = await graph_service.expand_documents(pool, site="diplomacy.edu")

    boosts = GraphExpansionService.compute_graph_boost_scores(
        pool, expansions, question, alpha=1.0,
    )

    scored = []
    for i, doc in enumerate(pool):
        base = doc.metadata.get("_reranker_score", doc.metadata.get("_final_score", 1.0 - i * 0.01))
        scored.append((base + alpha * boosts[i], doc))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored[:final_k]], expansions


async def graph_rerank_docs(
    docs, all_candidates, graph_service, question,
    replace_n: int = 2,
):
    """Swap low-coherence docs with higher-coherence alternatives from the pool.

    1. Graph-expand top-8 + broader pool
    2. Compute coherence per doc (shared topics/people with the rest)
    3. Replace bottom replace_n docs with best alternatives from pool
    """
    from app.ai.ai_services.graph_expansion import GraphExpansionService

    combined = list(docs) + [c for c in all_candidates if c not in docs]
    expansions, _ = await graph_service.expand_documents(combined[:20], site="diplomacy.edu")

    coh = GraphExpansionService.compute_coherence_scores(docs, expansions)

    doc_scores = list(zip(docs, coh))
    doc_scores.sort(key=lambda x: x[1])

    alt_pool = [c for c in all_candidates if c not in docs]
    if alt_pool:
        alt_exp, _ = await graph_service.expand_documents(alt_pool[:10], site="diplomacy.edu")
        expansions.update(alt_exp)
        test_set = list(docs) + alt_pool[:10]
        alt_coh = GraphExpansionService.compute_coherence_scores(test_set, expansions)
        alt_with_scores = list(zip(alt_pool[:10], alt_coh[len(docs):]))
        alt_with_scores.sort(key=lambda x: x[1], reverse=True)

        replaced = 0
        for alt_doc, alt_score in alt_with_scores:
            if replaced >= replace_n:
                break
            if replaced < len(doc_scores) and alt_score > doc_scores[replaced][1]:
                idx = docs.index(doc_scores[replaced][0])
                docs[idx] = alt_doc
                replaced += 1

    return docs, expansions


def format_context_for_llm(docs, expansions=None, fmt="inline", question=""):
    """Format retrieval context for LLM, with configurable graph injection.

    fmt: "none"             — plain text, no graph
         "inline"           — full [Graph: ...] metadata per source
         "appended"         — graph block after all sources
         "hint"             — conditional short hints only when text lacks info
         "retrieval_boost"  — no graph in prompt (boost was applied to retrieval)
         "rerank"           — no graph in prompt (reranking was graph-aware)
         "all"              — retrieval_boost + rerank already applied, add hints
         "boost_inline"     — retrieval_boost for doc selection + inline metadata in prompt
    """
    from app.ai.ai_services.graph_expansion import GraphExpansionService

    inject_in_prompt = fmt in ("inline", "appended", "hint", "all")

    parts = []
    sources = []
    for i, doc in enumerate(docs):
        title = doc.metadata.get("title", "Unknown")
        url = doc.metadata.get("url", "")
        block = f"[{i + 1}] {title}\n{doc.page_content}"

        if fmt in ("inline", "boost_inline") and expansions:
            graph_line = GraphExpansionService.format_inline_source_context(doc, expansions)
            if graph_line:
                block += "\n" + graph_line
        elif fmt in ("hint", "all") and expansions and question:
            hint = GraphExpansionService.format_conditional_hint(doc, expansions, question)
            if hint:
                block += "\n" + hint

        parts.append(block)
        sources.append({"idx": i + 1, "title": title, "url": url})

    context = "\n\n---\n\n".join(parts)

    if fmt == "appended" and expansions:
        graph_block = GraphExpansionService.format_graph_context(docs, expansions)
        if graph_block:
            context += graph_block

    return context, sources


async def judge_pair(judge_llm, question: str, baseline_answer: str, graph_answer: str) -> dict:
    swap = random.random() > 0.5
    if swap:
        a_answer, b_answer = graph_answer, baseline_answer
    else:
        a_answer, b_answer = baseline_answer, graph_answer

    prompt = JUDGE_PROMPT.format(
        question=question, answer_a=a_answer, answer_b=b_answer
    )

    resp = await judge_llm.ainvoke([HumanMessage(content=prompt)])
    raw = resp.content.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": raw[:300], "swap": swap}

    if swap:
        baseline_scores = result.get("scores_b", {})
        graph_scores = result.get("scores_a", {})
        raw_winner = result.get("winner", "TIE")
        winner_map = {"A": "graph", "B": "baseline", "TIE": "TIE"}
    else:
        baseline_scores = result.get("scores_a", {})
        graph_scores = result.get("scores_b", {})
        raw_winner = result.get("winner", "TIE")
        winner_map = {"A": "baseline", "B": "graph", "TIE": "TIE"}

    return {
        "baseline_scores": baseline_scores,
        "graph_scores": graph_scores,
        "winner": winner_map.get(raw_winner, "TIE"),
        "reasoning": result.get("reasoning", ""),
        "swap": swap,
    }


# ── Markdown report ──────────────────────────────────────────────────

def write_report(ab_results: list[ABResult], out_path: Path, meta: dict):
    criteria = ["completeness", "accuracy", "specificity", "source_usage"]
    lines = []
    w = lines.append

    w(f"# Graph RAG A/B Benchmark — {meta['timestamp']}\n")
    w(f"**LLM**: `{LLM_MODEL}` @ `{LLM_URL}`  ")
    w(f"**Retrieval**: Production pipeline (SentenceFirst + reranker, contextual, α=0.75, k=200→8)  ")
    w(f"**Graph variant**: `{meta.get('graph_fmt', 'inline')}`  ")
    w(f"**Graph DB**: Neo4j `{NEO4J_DATABASE}` @ `{NEO4J_URI}`  ")
    w(f"**Questions**: {len(ab_results)} (from MongoDB production)  ")
    w(f"**Judge**: {'LLM-as-Judge' if meta.get('with_judge') else 'Disabled'}\n")

    # Summary table
    w("## Summary\n")
    w("| # | Question | B time | G time | Δ expand | B avg | G avg | Winner |")
    w("|---|----------|--------|--------|----------|-------|-------|--------|")

    wins = {"baseline": 0, "graph": 0, "TIE": 0, "ERROR": 0}
    total_b = {c: 0 for c in criteria}
    total_g = {c: 0 for c in criteria}
    n_judged = 0

    for i, r in enumerate(ab_results):
        bs = r.judge_scores.get("baseline", {})
        gs = r.judge_scores.get("graph", {})
        b_avg = sum(bs.values()) / max(len(bs), 1) if bs else 0
        g_avg = sum(gs.values()) / max(len(gs), 1) if gs else 0
        verdict = r.judge_verdict or "—"
        winner_str = {
            "baseline": "Baseline",
            "graph": "**Graph**",
            "TIE": "Tie",
            "ERROR": "Error",
        }.get(verdict, "—")
        wins[verdict] = wins.get(verdict, 0) + 1
        if bs and gs:
            for c in criteria:
                total_b[c] += bs.get(c, 0)
                total_g[c] += gs.get(c, 0)
            n_judged += 1

        w(
            f"| {i + 1} | {r.question[:55]} | {r.t_baseline_total:.2f}s "
            f"| {r.t_graph_total:.2f}s | +{r.t_graph_expansion:.2f}s "
            f"| {b_avg:.1f} | {g_avg:.1f} | {winner_str} |"
        )

    avg_bt = sum(r.t_baseline_total for r in ab_results) / len(ab_results)
    avg_gt = sum(r.t_graph_total for r in ab_results) / len(ab_results)
    avg_exp = sum(r.t_graph_expansion for r in ab_results) / len(ab_results)
    w("")
    w(
        f"**Average times**: Baseline {avg_bt:.2f}s · Graph {avg_gt:.2f}s "
        f"· Expansion overhead +{avg_exp:.3f}s  "
    )
    if meta.get("with_judge"):
        w(
            f"**Wins**: Baseline {wins.get('baseline', 0)} · "
            f"Graph {wins.get('graph', 0)} · Tie {wins.get('TIE', 0)}  "
        )
    w("")

    # Detailed scores
    if n_judged > 0:
        w("## Detailed Scores\n")
        w("| # | Criterion | Baseline | Graph |")
        w("|---|-----------|----------|-------|")
        for i, r in enumerate(ab_results):
            bs = r.judge_scores.get("baseline", {})
            gs = r.judge_scores.get("graph", {})
            if not bs:
                continue
            for j, c in enumerate(criteria):
                q_col = f"**Q{i + 1}** {r.question[:40]}" if j == 0 else ""
                w(f"| {q_col} | {c} | {bs.get(c, '-')} | {gs.get(c, '-')} |")
            w(
                f"| | **average** | **{sum(bs.values()) / max(len(bs), 1):.1f}** "
                f"| **{sum(gs.values()) / max(len(gs), 1):.1f}** |"
            )
        w("")

    # Full answers
    w("## Full Answers\n")
    for i, r in enumerate(ab_results):
        w(f"### Q{i + 1}: {r.question}\n")
        bs = r.judge_scores.get("baseline", {})
        gs = r.judge_scores.get("graph", {})
        b_avg = sum(bs.values()) / max(len(bs), 1) if bs else 0
        g_avg = sum(gs.values()) / max(len(gs), 1) if gs else 0

        w("| | Baseline | Graph-Enriched |")
        w("|---|---|---|")
        w(
            f"| Time | {r.t_baseline_total:.2f}s "
            f"| {r.t_graph_total:.2f}s (+{r.t_graph_expansion:.2f}s expansion) |"
        )
        w(f"| Score avg | {b_avg:.1f}/5 | {g_avg:.1f}/5 |")
        w(f"| Expanded docs | — | {r.n_expanded} |")
        w(
            f"| **Winner** | {'**✓**' if r.judge_verdict == 'baseline' else ''} "
            f"| {'**✓**' if r.judge_verdict == 'graph' else ''} |"
        )
        w("")
        if r.judge_reasoning:
            w(f"> **Judge:** {r.judge_reasoning}\n")

        w("<details><summary>Baseline answer</summary>\n")
        w(r.baseline_answer)
        w("\n</details>\n")
        w("<details><summary>Graph-enriched answer</summary>\n")
        w(r.graph_answer)
        w("\n</details>\n")
        if r.graph_context_text:
            w("<details><summary>Graph context injected</summary>\n")
            w(f"```\n{r.graph_context_text}\n```")
            w("\n</details>\n")
        w("---\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────

async def run_benchmark(questions: list[str], with_judge: bool = True, graph_fmt: str = "inline"):
    from app.ai.ai_services.embeddings import TEIEmbeddings
    from app.ai.ai_services.graph_expansion import GraphExpansionService
    from app.ai.ai_services.neo4j_graph_client import Neo4jGraphClient

    import app.core.neo4j_config as _neo4j_cfg
    _neo4j_cfg.GRAPH_EXPANSION_ENABLED = True
    import app.ai.ai_services.graph_expansion as _ge_mod
    _ge_mod.GRAPH_EXPANSION_ENABLED = True

    print(f"\n{'=' * 70}")
    print(f"  Graph RAG A/B Benchmark — {len(questions)} questions")
    print(f"  Judge: {'ON' if with_judge else 'OFF'}  Format: {graph_fmt}")
    print(f"{'=' * 70}\n")

    # ── connections ──
    print("Connecting to Weaviate...", end=" ", flush=True)
    wv_client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
        auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
        additional_config=weaviate.classes.init.AdditionalConfig(
            timeout=weaviate.classes.init.Timeout(init=30)
        ),
    )
    print(f"OK (ready={wv_client.is_ready()})")

    print("Connecting to Neo4j...", end=" ", flush=True)
    neo4j_client = Neo4jGraphClient(
        uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASS,
        database_diplo=NEO4J_DATABASE,
    )
    await neo4j_client.connect()
    healthy = await neo4j_client.health_check()
    print(f"OK (healthy={healthy})")

    print("Initializing embeddings...", end=" ", flush=True)
    embeddings = TEIEmbeddings(url=EMBEDDING_URL, api_key=EMBEDDING_KEY)
    print("OK")

    print("Creating production retriever...", end=" ", flush=True)
    retriever, _ = create_retriever(wv_client, embeddings)
    from app.core.config import USE_RERANKER as _UR, RERANKER_URL as _RU, RERANKER_API_KEY as _RK, RERANKER_TOP_K as _RT
    reranker = TEIReranker(url=_RU, api_key=_RK, top_k=_RT) if _UR else None
    print(f"OK ({type(retriever).__name__}, reranker={reranker is not None})")

    graph_service = GraphExpansionService(neo4j_client)

    llm = ChatOpenAI(
        model_name=LLM_MODEL, api_key=LLM_KEY, base_url=LLM_URL, temperature=0.3,
    )
    judge_llm = ChatOpenAI(
        model_name=LLM_MODEL, api_key=LLM_KEY, base_url=LLM_URL, temperature=0.0,
    )

    # ── run ──
    ab_results: list[ABResult] = []
    t_total_start = time.time()

    for qi, question in enumerate(questions):
        print(f"\n[{qi + 1}/{len(questions)}] {question[:70]}")
        r = ABResult(question=question)

        # BASELINE
        try:
            t0 = time.time()
            docs = retrieve_with_production_pipeline(retriever, reranker, question)
            r.t_baseline_retrieval = time.time() - t0

            context_text, r.baseline_sources = format_context_for_llm(docs, fmt="none")
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {question}"),
            ]
            t0 = time.time()
            resp = await llm.ainvoke(messages)
            r.t_baseline_llm = time.time() - t0
            r.baseline_answer = resp.content.strip()
            r.t_baseline_total = r.t_baseline_retrieval + r.t_baseline_llm
            print(f"  BASELINE  retr={r.t_baseline_retrieval:.2f}s llm={r.t_baseline_llm:.2f}s total={r.t_baseline_total:.2f}s")
        except Exception as e:
            print(f"  BASELINE ERROR: {e}")
            r.baseline_answer = f"[ERROR] {e}"
            r.t_baseline_total = time.time() - t0

        # GRAPH-ENRICHED (variant depends on graph_fmt)
        try:
            t0 = time.time()

            expansions = {}
            if graph_fmt in ("retrieval_boost", "all", "boost_inline"):
                docs_g, expansions = await retrieve_with_graph_boost(
                    retriever, reranker, graph_service, question,
                    final_k=8, pool_k=16, alpha=0.15,
                )
                r.t_graph_retrieval = time.time() - t0
                r.n_expanded = len(expansions)
            else:
                docs_g = retrieve_with_production_pipeline(retriever, reranker, question)
                r.t_graph_retrieval = time.time() - t0

            if graph_fmt in ("rerank", "all"):
                t0 = time.time()
                if not expansions:
                    expansions, _ = await graph_service.expand_documents(docs_g, site="diplomacy.edu")
                broader_pool = retriever.invoke(question)[:20]
                docs_g, expansions = await graph_rerank_docs(
                    docs_g, broader_pool, graph_service, question, replace_n=2,
                )
                r.t_graph_expansion = time.time() - t0
                r.n_expanded = len(expansions)
            elif not expansions:
                t0 = time.time()
                expansions, _ = await graph_service.expand_documents(docs_g, site="diplomacy.edu")
                r.t_graph_expansion = time.time() - t0
                r.n_expanded = len(expansions)

            ctx_g, r.graph_sources = format_context_for_llm(
                docs_g, expansions=expansions, fmt=graph_fmt, question=question,
            )
            baseline_ctx = format_context_for_llm(docs_g, fmt="none")[0]
            r.graph_context_text = ctx_g[len(baseline_ctx):] if len(ctx_g) > len(baseline_ctx) else ""

            messages_g = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Context:\n{ctx_g}\n\nQuestion: {question}"),
            ]
            t0 = time.time()
            resp_g = await llm.ainvoke(messages_g)
            r.t_graph_llm = time.time() - t0
            r.graph_answer = resp_g.content.strip()
            r.t_graph_total = r.t_graph_retrieval + r.t_graph_expansion + r.t_graph_llm
            print(f"  GRAPH({graph_fmt})  retr={r.t_graph_retrieval:.2f}s exp={r.t_graph_expansion:.2f}s llm={r.t_graph_llm:.2f}s total={r.t_graph_total:.2f}s expanded={r.n_expanded}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  GRAPH ERROR: {e}")
            r.graph_answer = f"[ERROR] {e}"
            r.t_graph_total = time.time() - t0

        # JUDGE
        if with_judge and not r.baseline_answer.startswith("[ERROR") and not r.graph_answer.startswith("[ERROR"):
            try:
                verdict = await judge_pair(judge_llm, question, r.baseline_answer, r.graph_answer)
                if "error" in verdict:
                    r.judge_verdict = "ERROR"
                    r.judge_reasoning = verdict.get("error", "")[:200]
                    print(f"  JUDGE     parse error")
                else:
                    r.judge_scores = {
                        "baseline": verdict["baseline_scores"],
                        "graph": verdict["graph_scores"],
                    }
                    r.judge_verdict = verdict["winner"]
                    r.judge_reasoning = verdict["reasoning"]
                    print(f"  JUDGE     winner={r.judge_verdict}")
            except Exception as e:
                print(f"  JUDGE ERROR: {e}")
                r.judge_verdict = "ERROR"

        ab_results.append(r)

    total_elapsed = time.time() - t_total_start

    # ── print summary ──
    print(f"\n{'=' * 70}")
    print(f"  COMPLETED {len(ab_results)} questions in {total_elapsed:.1f}s")
    print(f"{'=' * 70}\n")

    criteria = ["completeness", "accuracy", "specificity", "source_usage"]
    wins = Counter(r.judge_verdict for r in ab_results if r.judge_verdict)
    n_judged = sum(1 for r in ab_results if r.judge_scores.get("baseline"))

    if n_judged > 0:
        b_avgs = {c: 0 for c in criteria}
        g_avgs = {c: 0 for c in criteria}
        for r in ab_results:
            bs = r.judge_scores.get("baseline", {})
            gs = r.judge_scores.get("graph", {})
            if bs:
                for c in criteria:
                    b_avgs[c] += bs.get(c, 0)
                    g_avgs[c] += gs.get(c, 0)

        print("Criteria averages (1-5):")
        for c in criteria:
            print(f"  {c:15s}  Baseline={b_avgs[c] / n_judged:.2f}  Graph={g_avgs[c] / n_judged:.2f}")

    avg_bt = sum(r.t_baseline_total for r in ab_results) / len(ab_results)
    avg_gt = sum(r.t_graph_total for r in ab_results) / len(ab_results)
    avg_exp = sum(r.t_graph_expansion for r in ab_results) / len(ab_results)
    print(f"\nTiming averages:")
    print(f"  Baseline total:      {avg_bt:.2f}s")
    print(f"  Graph total:         {avg_gt:.2f}s")
    print(f"  Graph expansion:     +{avg_exp:.3f}s overhead")

    if wins:
        print(f"\nWins: Baseline={wins.get('baseline', 0)}  Graph={wins.get('graph', 0)}  Tie={wins.get('TIE', 0)}  Error={wins.get('ERROR', 0)}")

    # ── save report ──
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = Path("notebooks/benchmark_results")
    fmt_tag = f"_{graph_fmt}" if graph_fmt != "inline" else ""
    md_path = write_report(
        ab_results,
        out_dir / f"ab_comparison_{ts}{fmt_tag}.md",
        {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "with_judge": with_judge, "graph_fmt": graph_fmt},
    )
    print(f"\nMarkdown report: {md_path}")

    json_path = out_dir / f"ab_comparison_{ts}{fmt_tag}.json"
    json_data = []
    for r in ab_results:
        json_data.append({
            "question": r.question,
            "baseline_answer": r.baseline_answer,
            "graph_answer": r.graph_answer,
            "graph_context": r.graph_context_text,
            "n_expanded": r.n_expanded,
            "t_baseline_retrieval": r.t_baseline_retrieval,
            "t_baseline_llm": r.t_baseline_llm,
            "t_baseline_total": r.t_baseline_total,
            "t_graph_retrieval": r.t_graph_retrieval,
            "t_graph_expansion": r.t_graph_expansion,
            "t_graph_llm": r.t_graph_llm,
            "t_graph_total": r.t_graph_total,
            "judge_verdict": r.judge_verdict,
            "judge_reasoning": r.judge_reasoning,
            "judge_scores": r.judge_scores,
        })
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON data:       {json_path}")
    print(f"File sizes:      {md_path.stat().st_size / 1024:.1f} KB (md), {json_path.stat().st_size / 1024:.1f} KB (json)")

    # ── cleanup ──
    await neo4j_client.close()
    wv_client.close()
    print("\nConnections closed. Done.")


async def run_compare(questions: list[str], formats: list[str], with_judge: bool = True):
    """Run benchmark for multiple graph variants on the same questions and print comparison."""
    results_by_fmt: dict[str, dict] = {}

    for fmt in formats:
        print(f"\n{'#' * 70}")
        print(f"  VARIANT: {fmt}")
        print(f"{'#' * 70}")
        await run_benchmark(questions, with_judge=with_judge, graph_fmt=fmt)

        latest_json = sorted(Path("notebooks/benchmark_results").glob(f"ab_comparison_*_{fmt}.json" if fmt != "inline" else "ab_comparison_*.json"))
        if not latest_json:
            latest_json = sorted(Path("notebooks/benchmark_results").glob("ab_comparison_*.json"))
        if latest_json:
            with open(latest_json[-1]) as f:
                data = json.load(f)
            wins = Counter(d.get("judge_verdict", "") for d in data)
            b_avg = sum(
                sum(d["judge_scores"].get("baseline", {}).values()) / max(len(d["judge_scores"].get("baseline", {})), 1)
                for d in data if d.get("judge_scores", {}).get("baseline")
            ) / max(sum(1 for d in data if d.get("judge_scores", {}).get("baseline")), 1)
            g_avg = sum(
                sum(d["judge_scores"].get("graph", {}).values()) / max(len(d["judge_scores"].get("graph", {})), 1)
                for d in data if d.get("judge_scores", {}).get("graph")
            ) / max(sum(1 for d in data if d.get("judge_scores", {}).get("graph")), 1)
            results_by_fmt[fmt] = {
                "baseline_wins": wins.get("baseline", 0),
                "graph_wins": wins.get("graph", 0),
                "ties": wins.get("TIE", 0),
                "b_avg": b_avg,
                "g_avg": g_avg,
                "file": str(latest_json[-1]),
            }

    print(f"\n{'=' * 70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'=' * 70}\n")
    print(f"{'Format':<20s} {'B wins':>7s} {'G wins':>7s} {'Ties':>6s} {'B avg':>7s} {'G avg':>7s}")
    print("-" * 60)
    for fmt, r in results_by_fmt.items():
        print(f"{fmt:<20s} {r['baseline_wins']:>7d} {r['graph_wins']:>7d} {r['ties']:>6d} {r['b_avg']:>7.2f} {r['g_avg']:>7.2f}")
    print()

    summary_path = Path("notebooks/benchmark_results") / f"compare_{datetime.now().strftime('%Y-%m-%d_%H%M')}.json"
    summary_path.write_text(json.dumps(results_by_fmt, indent=2), encoding="utf-8")
    print(f"Comparison saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Graph RAG A/B Benchmark")
    parser.add_argument("--n", type=int, default=100, help="Number of production questions to benchmark (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for question sampling (default: 42)")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-Judge evaluation")
    parser.add_argument("--questions", nargs="+", help="Override: provide specific questions instead of MongoDB")
    parser.add_argument("--format", choices=GRAPH_FORMATS, default="inline",
                        help=f"Graph context format: {'/'.join(GRAPH_FORMATS)} (default: inline)")
    parser.add_argument("--compare", nargs="*", metavar="FMT",
                        help="Run multiple variants and compare. No args = all variants. "
                             "Or list specific: --compare inline hint retrieval_boost")
    args = parser.parse_args()

    if args.questions:
        questions = args.questions
        print(f"Using {len(questions)} manually provided questions")
    else:
        questions = load_production_questions(args.n, seed=args.seed)

    if args.compare is not None:
        formats = args.compare if args.compare else [f for f in GRAPH_FORMATS if f != "none"]
        asyncio.run(run_compare(questions, formats, with_judge=not args.no_judge))
    else:
        asyncio.run(run_benchmark(questions, with_judge=not args.no_judge, graph_fmt=args.format))


if __name__ == "__main__":
    main()
