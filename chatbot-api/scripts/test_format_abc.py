#!/usr/bin/env -S python3 -u
"""
A/B/C Format Test — compares 3 ways of injecting graph context into the LLM prompt.

  A: Baseline        — no graph context at all
  B: Appended block  — current production format (graph block at the end)
  C: Inline per-src  — new format (graph metadata line under each source)

Runs the same retrieval + graph expansion for each question, then generates
3 answers (one per format) and has LLM-as-Judge rank them.

Usage:
    cd chatbot-api
    python scripts/test_format_abc.py                  # 10 questions
    python scripts/test_format_abc.py --n 5

All credentials are read from the repo-level .env file (or ENV_FILE override).
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

import argparse
import asyncio
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from starlette.config import Config

import weaviate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from weaviate.classes.query import Filter, HybridFusion

import app.core.neo4j_config as _neo4j_cfg
_neo4j_cfg.GRAPH_EXPANSION_ENABLED = True
import app.ai.ai_services.graph_expansion as _ge_mod
_ge_mod.GRAPH_EXPANSION_ENABLED = True

from app.ai.ai_services.embeddings import TEIEmbeddings
from app.ai.ai_services.graph_expansion import GraphExpansionService
from app.ai.ai_services.neo4j_graph_client import Neo4jGraphClient, GraphExpansionResult

# ── Config (loaded from .env) ─────────────────────────────────────────

env_file = os.getenv("ENV_FILE", "../.env")
_config = Config(env_file)

# Neo4j
NEO4J_URI = _config("NEO4J_URI", cast=str, default="bolt://localhost:7687")
NEO4J_USER = _config("NEO4J_USER", cast=str, default="neo4j")
NEO4J_PASS = _config("NEO4J_PASS", cast=str)
NEO4J_DATABASE = _config("NEO4J_DATABASE_DIPLO", cast=str, default="weaviatediplo")

# Weaviate (parse WV_CLIENT_URL for host/port)
_WV_URL = _config("WV_CLIENT_URL", cast=str)
_wv_parsed = urlparse(_WV_URL)
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

SYSTEM_PROMPT = (
    "You are Diplorene, a helpful AI assistant specializing in diplomacy, "
    "internet governance, and digital policy. Answer based ONLY on the provided "
    "context. Cite sources as [1], [2], etc. If the context doesn't contain "
    "enough information, say so honestly."
)

# ── Formatting functions ──────────────────────────────────────────────

def format_A_baseline(docs: list[Document]) -> str:
    """Format A: plain sources, no graph."""
    parts = []
    for i, doc in enumerate(docs):
        title = doc.metadata.get("title", "Unknown")
        parts.append(f"[{i+1}] {title}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def format_B_appended(docs: list[Document], expansions: dict[str, GraphExpansionResult]) -> str:
    """Format B: sources + appended graph block (current production format)."""
    parts = []
    for i, doc in enumerate(docs):
        title = doc.metadata.get("title", "Unknown")
        parts.append(f"[{i+1}] {title}\n{doc.page_content}")
    context = "\n\n---\n\n".join(parts)

    graph_sections = []
    for doc in docs:
        url = doc.metadata.get("url", "")
        exp = expansions.get(url)
        if not exp:
            continue
        title = doc.metadata.get("title", "") or exp.document_name or "Unknown"
        lines = []
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
                clean = [l for l in rd.labels if l != "Document"]
                tag = f" [{clean[0]}]" if clean else ""
                names.append(f"{rd.name[:50]}{tag}")
            lines.append(f"  Related content: {'; '.join(names)}")
        if lines:
            graph_sections.append(f'Graph context for "{title}":\n' + "\n".join(lines))

    if graph_sections:
        context += "\n\n--- KNOWLEDGE GRAPH CONTEXT ---\n\n" + "\n\n".join(graph_sections)
    return context


def _inline_graph_line(exp: GraphExpansionResult) -> str:
    """Build a compact [Graph: ...] line from an expansion result."""
    parts = []
    if exp.topics:
        parts.append(f"Topics: {', '.join(exp.topics[:6])}")
    if exp.people:
        parts.append(f"People: {', '.join(exp.people[:5])}")
    if exp.actors:
        parts.append(f"Orgs: {', '.join(exp.actors[:5])}")
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


def format_C_inline(docs: list[Document], expansions: dict[str, GraphExpansionResult]) -> str:
    """Format C: graph metadata inline under each source."""
    parts = []
    for i, doc in enumerate(docs):
        title = doc.metadata.get("title", "Unknown")
        block = f"[{i+1}] {title}\n{doc.page_content}"
        url = doc.metadata.get("url", "")
        exp = expansions.get(url)
        if exp:
            graph_line = _inline_graph_line(exp)
            if graph_line:
                block += "\n" + graph_line
        parts.append(block)
    return "\n\n---\n\n".join(parts)


# ── Judge ─────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an expert evaluator comparing three answers (A, B, C) to the same question.
All answers were generated by the same RAG system with different context formatting.

Question: {question}

=== Answer A ===
{answer_a}

=== Answer B ===
{answer_b}

=== Answer C ===
{answer_c}

Evaluate EACH answer on these criteria (1-5 scale):
1. **Completeness**: How thoroughly does it address all aspects of the question?
2. **Accuracy**: Does it stick to provided context without hallucinating?
3. **Specificity**: Concrete details, names, dates, examples vs vague generalities?
4. **Source_usage**: How well does it cite sources as [1], [2], etc.?

Then pick the overall **winner**: "A", "B", or "C".

Respond ONLY with valid JSON:
{{
  "scores_a": {{"completeness": N, "accuracy": N, "specificity": N, "source_usage": N}},
  "scores_b": {{"completeness": N, "accuracy": N, "specificity": N, "source_usage": N}},
  "scores_c": {{"completeness": N, "accuracy": N, "specificity": N, "source_usage": N}},
  "winner": "A" or "B" or "C",
  "reasoning": "1-2 sentence explanation"
}}"""


# ── Retrieval ─────────────────────────────────────────────────────────

async def retrieve_and_group(wv_client, embeddings, question, limit=30):
    qv = embeddings.embed_query(question)
    col = wv_client.collections.get("DiploChunk_contextual")
    res = col.query.hybrid(
        query=question, vector=qv, alpha=0.75,
        fusion_type=HybridFusion.RELATIVE_SCORE, limit=limit,
        filters=Filter.by_property("chunk_level").equal("sentence")
        & Filter.by_property("visibility").not_equal("private"),
        query_properties=["sentence"], return_metadata=["score"],
    )
    groups = defaultdict(lambda: {"sentences": [], "scores": [], "title": "", "url": ""})
    for obj in res.objects:
        p = obj.properties
        score = obj.metadata.score if obj.metadata else 0
        url = p.get("link", "")
        section = p.get("section", "general")
        key = (url, section)
        groups[key]["sentences"].append({"text": p.get("sentence", ""), "score": score})
        groups[key]["scores"].append(score)
        groups[key]["title"] = p.get("h1", "") or p.get("last_h_title", "")
        groups[key]["url"] = url
    sorted_g = sorted(groups.items(), key=lambda x: max(x[1]["scores"]), reverse=True)
    docs = []
    for (url, _), g in sorted_g[:8]:
        content = " ".join(s["text"] for s in g["sentences"][:5])
        docs.append(Document(
            page_content=content,
            metadata={"url": url, "title": g["title"], "parent_document_hash": ""},
        ))
    return docs


# ── Main ──────────────────────────────────────────────────────────────

async def run_test(questions: list[str]):
    print(f"\n{'='*70}")
    print(f"  Format A/B/C Test — {len(questions)} questions")
    print(f"  A=baseline  B=appended block  C=inline per-source")
    print(f"{'='*70}\n")

    wv_client = weaviate.connect_to_local(
        host=WEAVIATE_HOST, port=WEAVIATE_PORT, grpc_port=WEAVIATE_GRPC_PORT,
        auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
        additional_config=weaviate.classes.init.AdditionalConfig(
            timeout=weaviate.classes.init.Timeout(init=30)),
    )
    neo_client = Neo4jGraphClient(
        uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASS,
        database_diplo=NEO4J_DATABASE,
    )
    await neo_client.connect()
    embeddings = TEIEmbeddings(url=EMBEDDING_URL, api_key=EMBEDDING_KEY)
    graph_service = GraphExpansionService(neo_client)

    llm = ChatOpenAI(model_name=LLM_MODEL, api_key=LLM_KEY, base_url=LLM_URL, temperature=0.3)
    judge_llm = ChatOpenAI(model_name=LLM_MODEL, api_key=LLM_KEY, base_url=LLM_URL, temperature=0.0)

    results = []

    for qi, question in enumerate(questions):
        print(f"\n[{qi+1}/{len(questions)}] {question[:70]}")

        docs = await retrieve_and_group(wv_client, embeddings, question)
        expansions, exp_time = await graph_service.expand_documents(docs, site="diplomacy.edu")
        n_expanded = len(expansions)
        print(f"  Expanded: {n_expanded} docs ({exp_time:.2f}s)")

        if n_expanded == 0:
            print(f"  SKIP — no graph data, all 3 formats would be identical")
            continue

        ctx_a = format_A_baseline(docs)
        ctx_b = format_B_appended(docs, expansions)
        ctx_c = format_C_inline(docs, expansions)

        answers = {}
        times = {}
        for label, ctx in [("A", ctx_a), ("B", ctx_b), ("C", ctx_c)]:
            msgs = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Context:\n{ctx}\n\nQuestion: {question}"),
            ]
            t0 = time.time()
            resp = await llm.ainvoke(msgs)
            times[label] = time.time() - t0
            answers[label] = resp.content.strip()

        print(f"  LLM times: A={times['A']:.1f}s  B={times['B']:.1f}s  C={times['C']:.1f}s")

        # Randomize order for judge to avoid position bias
        order = ["A", "B", "C"]
        random.shuffle(order)
        label_map = {order[i]: chr(65 + i) for i in range(3)}
        reverse_map = {chr(65 + i): order[i] for i in range(3)}

        prompt = JUDGE_PROMPT.format(
            question=question,
            answer_a=answers[order[0]],
            answer_b=answers[order[1]],
            answer_c=answers[order[2]],
        )
        resp = await judge_llm.ainvoke([HumanMessage(content=prompt)])
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

        try:
            verdict = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  JUDGE ERROR: {raw[:100]}")
            continue

        real_scores = {}
        for slot in ["a", "b", "c"]:
            real_label = reverse_map[slot.upper()]
            real_scores[real_label] = verdict.get(f"scores_{slot}", {})

        raw_winner = verdict.get("winner", "?").upper()
        real_winner = reverse_map.get(raw_winner, "?")

        for label in ["A", "B", "C"]:
            s = real_scores[label]
            avg = sum(s.values()) / max(len(s), 1)
            print(f"  {label}: avg={avg:.1f} {s}")
        print(f"  Winner: {real_winner}  ({verdict.get('reasoning', '')})")

        results.append({
            "question": question,
            "n_expanded": n_expanded,
            "answers": answers,
            "times": times,
            "scores": real_scores,
            "winner": real_winner,
            "reasoning": verdict.get("reasoning", ""),
            "ctx_a_len": len(ctx_a),
            "ctx_b_len": len(ctx_b),
            "ctx_c_len": len(ctx_c),
        })

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  RESULTS ({len(results)} questions with graph data)")
    print(f"{'='*70}\n")

    wins = {"A": 0, "B": 0, "C": 0}
    criteria = ["completeness", "accuracy", "specificity", "source_usage"]
    totals = {l: {c: 0 for c in criteria} for l in "ABC"}
    n = len(results)

    for r in results:
        wins[r["winner"]] = wins.get(r["winner"], 0) + 1
        for label in "ABC":
            for c in criteria:
                totals[label][c] += r["scores"][label].get(c, 0)

    print(f"  Wins:  A(baseline)={wins['A']}  B(appended)={wins['B']}  C(inline)={wins['C']}\n")
    print(f"  {'Criterion':<15s}  {'A(baseline)':>11s}  {'B(appended)':>11s}  {'C(inline)':>11s}")
    print(f"  {'-'*52}")
    for c in criteria:
        a_avg = totals["A"][c] / n if n else 0
        b_avg = totals["B"][c] / n if n else 0
        c_avg = totals["C"][c] / n if n else 0
        print(f"  {c:<15s}  {a_avg:>11.2f}  {b_avg:>11.2f}  {c_avg:>11.2f}")

    a_total = sum(totals["A"][c] for c in criteria) / (n * 4) if n else 0
    b_total = sum(totals["B"][c] for c in criteria) / (n * 4) if n else 0
    c_total = sum(totals["C"][c] for c in criteria) / (n * 4) if n else 0
    print(f"  {'-'*52}")
    print(f"  {'OVERALL':<15s}  {a_total:>11.2f}  {b_total:>11.2f}  {c_total:>11.2f}")

    # Save
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = Path("notebooks/benchmark_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"format_abc_test_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    await neo_client.close()
    wv_client.close()


def main():
    parser = argparse.ArgumentParser(description="Format A/B/C Test")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
    random.seed(args.seed)
    questions = random.sample(all_q, min(args.n, len(all_q)))
    print(f"Selected {len(questions)} questions from {len(all_q)} unique production queries")

    asyncio.run(run_test(questions))


if __name__ == "__main__":
    main()
