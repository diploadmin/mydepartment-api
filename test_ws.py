#!/usr/bin/env python3
"""
Manual WebSocket test client (legacy or safe ticket route).

Usage:
  python3 test_ws.py "your question"
  python3 test_ws.py --safe "your question"
  CHAT_HTTP_API_KEY=... python3 test_ws.py --safe --replay "your question"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

import websockets

BASE = os.environ.get("CHAT_API_BASE", "http://localhost:8561")
WS_BASE = os.environ.get("CHAT_API_WS_BASE", "ws://localhost:8561")
ORIGIN = "https://www.diplomacy.edu"
SUBPROTOCOL = "chatbot.ticket.v1"


def _post_json(url: str, payload: dict, headers: dict | None = None) -> dict:
    body = json.dumps(payload).encode()
    req_headers = {"Content-Type": "application/json", "Origin": ORIGIN}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


async def run_ws(uri: str, question: str, subprotocols: list[str] | None = None) -> list[dict]:
    headers = {"Origin": ORIGIN}
    kwargs = {"max_size": 2**20, "additional_headers": headers}
    if subprotocols:
        kwargs["subprotocols"] = subprotocols
    all_messages: list[dict] = []
    async with websockets.connect(uri, **kwargs) as ws:
        welcome = await ws.recv()
        print(f"Question: {question}")
        print("Connected OK")
        print(f"Negotiated subprotocol: {ws.subprotocol!r}")
        payload = {
            "message": question,
            "user_ip": "127.0.0.1",
            "user_type": "general",
        }
        await ws.send(json.dumps(payload))
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=120)
                data = json.loads(msg)
                all_messages.append(data)
        except (websockets.exceptions.ConnectionClosedOK, asyncio.TimeoutError):
            pass
    return all_messages


def print_sources(all_messages: list[dict]) -> None:
    sources = None
    for m in all_messages:
        if m.get("status") == "sources":
            sources = m.get("text", [])
    if not sources:
        print("No sources found!")
        print(f"Total messages received: {len(all_messages)}")
        for m in all_messages:
            print(f"  status={m.get('status')} text_len={len(str(m.get('text','')))}")
        return
    print(f"\nTotal sources: {len(sources)}")
    print("=" * 80)
    for i, s in enumerate(sources):
        url = s.get("deep_link_url", s.get("url", "NO URL"))
        title = s.get("title", "NO TITLE")
        print(f"[{i+1}] {title[:70]}")
        print(f"    URL: {url}")
        print()


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+", help="chat question")
    parser.add_argument(
        "--safe",
        action="store_true",
        help="use /get_id_safe + /ws-safe with Sec-WebSocket-Protocol ticket",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="with --safe: attempt a second connect with the same ticket (must fail)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CHAT_HTTP_API_KEY", ""),
        help="X-Chat-Api-Key (or env CHAT_HTTP_API_KEY)",
    )
    args = parser.parse_args()
    question = " ".join(args.question)

    if args.safe:
        if not args.api_key:
            print("ERROR: --safe requires --api-key or CHAT_HTTP_API_KEY", file=sys.stderr)
            return 1
        data = _post_json(
            f"{BASE}/api/conversation/get_id_safe",
            {"conversation_id": None},
            headers={"X-Chat-Api-Key": args.api_key},
        )
        conv_id = data["conversationId"]
        ticket = data["wsTicket"]
        print(f"Safe conversationId={conv_id} expiresIn={data.get('expiresIn')}")
        uri = f"{WS_BASE}/api/chat/ws-safe/{conv_id}"
        messages = await run_ws(uri, question, subprotocols=[SUBPROTOCOL, ticket])
        print_sources(messages)

        if args.replay:
            print("\nReplay same ticket (expect failure)...")
            try:
                await run_ws(uri, question, subprotocols=[SUBPROTOCOL, ticket])
                print("ERROR: replay unexpectedly succeeded", file=sys.stderr)
                return 1
            except Exception as e:
                print(f"Replay rejected as expected: {type(e).__name__}: {e}")
        return 0

    data = _post_json(
        f"{BASE}/api/conversation/get_id",
        {"conversation_id": None},
    )
    conv_id = data["conversationId"]
    uri = f"{WS_BASE}/api/chat/ws/{conv_id}"
    messages = await run_ws(uri, question)
    print_sources(messages)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except urllib.error.HTTPError as e:
        print(f"HTTPError {e.code}: {e.read()[:300]!r}", file=sys.stderr)
        raise SystemExit(1)
