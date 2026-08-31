#!/usr/bin/env python3
"""
Benchmark: End-to-end API response speed.
Measures total latency for the full pipeline:
  embedding → retrieval → reranker → LLM generation → response

Usage:
    python scripts/benchmark_speed.py
"""
import requests
import time
import json
import statistics
import sys

BASE_URL = "http://127.0.0.1:8561/api"

# Test queries — diverse topics and complexity
QUERIES = [
    "What is digital diplomacy?",
    "How does AI affect international relations?",
    "What are the main principles of the Vienna Convention?",
    "Explain cybersecurity in diplomatic context",
    "What is the role of UNESCO in digital governance?",
]

# Timeout per request (seconds)
TIMEOUT = 120


def get_conversation_id() -> str:
    """Get a fresh conversation ID from the API."""
    resp = requests.post(
        f"{BASE_URL}/conversation/get_id",
        json={"conversation_id": None},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["conversationId"]


def chat(conversation_id: str, message: str) -> dict:
    """Send a chat message and return the full response + timing."""
    payload = {
        "user_ip": "127.0.0.1",
        "message": message,
        "user_type": "general",
    }
    t0 = time.perf_counter()
    resp = requests.post(
        f"{BASE_URL}/chat/{conversation_id}",
        json=payload,
        timeout=TIMEOUT,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    return {
        "elapsed_s": round(elapsed, 3),
        "status": resp.status_code,
        "data": resp.json(),
    }


def run_benchmark():
    print("=" * 70)
    print("  BENCHMARK: End-to-end API Speed")
    print("=" * 70)
    
    # Read current config
    try:
        with open("/opt/humainism_ai_chatbot_api/.env") as f:
            env_text = f.read()
        for key in ["RETRIEVAL_MODE", "USE_RERANKER", "RERANKER_TOP_K",
                     "SENTENCE_RERANKER_TOP_K", "TWOPHASE_RERANKER_TOP_K",
                     "RETRIEVAL_CHUNKS", "SENTENCE_RETRIEVAL_K",
                     "MAX_SECTIONS_PER_URL", "SKIP_TOOL_DECISION",
                     "PARAGRAPH_FETCH_LIMIT", "TWOPHASE_K_URLS"]:
            for line in env_text.splitlines():
                stripped = line.strip()
                if stripped.startswith(key + " ") or stripped.startswith(key + "="):
                    print(f"  {stripped}")
                    break
    except Exception:
        pass

    print("-" * 70)
    print()

    # Warmup — first request is always slower (model loading, connection pool)
    print("[Warmup] Getting conversation ID...")
    try:
        warmup_cid = get_conversation_id()
        print(f"[Warmup] CID: {warmup_cid}")
        print("[Warmup] Sending warmup query...")
        warmup_result = chat(warmup_cid, "Hello")
        print(f"[Warmup] Done in {warmup_result['elapsed_s']}s")
    except Exception as e:
        print(f"[Warmup] FAILED: {e}")
        print("  Make sure the API is running on port 8561")
        sys.exit(1)

    print()
    print("-" * 70)
    print(f"  Running {len(QUERIES)} benchmark queries...")
    print("-" * 70)
    print()

    results = []

    for i, query in enumerate(QUERIES, 1):
        # Fresh conversation for each query (no history overhead)
        cid = get_conversation_id()
        print(f"[{i}/{len(QUERIES)}] \"{query}\"")
        
        try:
            result = chat(cid, query)
            elapsed = result["elapsed_s"]
            data = result["data"]
            answer_len = len(data.get("answer", ""))
            sources_count = len(data.get("sources", []))
            related_count = len(data.get("related_questions", []))
            
            results.append({
                "query": query,
                "elapsed_s": elapsed,
                "answer_chars": answer_len,
                "sources": sources_count,
                "related_questions": related_count,
            })
            
            print(f"        Time: {elapsed:.2f}s | Answer: {answer_len} chars | "
                  f"Sources: {sources_count} | Related Q: {related_count}")
            
        except requests.exceptions.Timeout:
            print(f"        TIMEOUT after {TIMEOUT}s")
            results.append({
                "query": query, "elapsed_s": TIMEOUT,
                "answer_chars": 0, "sources": 0, "related_questions": 0,
                "error": "timeout"
            })
        except Exception as e:
            print(f"        ERROR: {e}")
            results.append({
                "query": query, "elapsed_s": 0,
                "answer_chars": 0, "sources": 0, "related_questions": 0,
                "error": str(e)
            })
        print()

    # Summary
    successful = [r for r in results if "error" not in r]
    
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    
    if successful:
        times = [r["elapsed_s"] for r in successful]
        print(f"  Queries tested:  {len(QUERIES)}")
        print(f"  Successful:      {len(successful)}")
        print(f"  Failed:          {len(results) - len(successful)}")
        print()
        print(f"  Min time:        {min(times):.2f}s")
        print(f"  Max time:        {max(times):.2f}s")
        print(f"  Mean time:       {statistics.mean(times):.2f}s")
        print(f"  Median time:     {statistics.median(times):.2f}s")
        if len(times) > 1:
            print(f"  Std dev:         {statistics.stdev(times):.2f}s")
        print()
        
        avg_chars = statistics.mean([r["answer_chars"] for r in successful])
        avg_sources = statistics.mean([r["sources"] for r in successful])
        print(f"  Avg answer len:  {avg_chars:.0f} chars")
        print(f"  Avg sources:     {avg_sources:.1f}")
    else:
        print("  No successful queries!")
    
    print()
    print("-" * 70)
    print("  DETAIL TABLE")
    print("-" * 70)
    print(f"  {'#':<3} {'Time':>7} {'Chars':>7} {'Src':>4} {'Query'}")
    print(f"  {'—'*3} {'—'*7} {'—'*7} {'—'*4} {'—'*40}")
    for i, r in enumerate(results, 1):
        status = "ERR" if "error" in r else ""
        print(f"  {i:<3} {r['elapsed_s']:>6.2f}s {r['answer_chars']:>7} {r['sources']:>4} {r['query'][:45]}{status}")
    
    print("=" * 70)
    
    # Save results
    output_file = "/opt/humainism_ai_chatbot_api/benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "retrieval_mode": "twophase",  # from .env
            },
            "warmup_time_s": warmup_result["elapsed_s"] if warmup_result else None,
            "results": results,
            "summary": {
                "total_queries": len(QUERIES),
                "successful": len(successful),
                "min_s": min(times) if successful else None,
                "max_s": max(times) if successful else None,
                "mean_s": round(statistics.mean(times), 3) if successful else None,
                "median_s": round(statistics.median(times), 3) if successful else None,
            }
        }, f, indent=2)
    print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    run_benchmark()
