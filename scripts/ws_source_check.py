"""Ad-hoc WS probe: prints Sources hrefs and citation URLs for one question."""

import asyncio
import json
import os
import sys

import httpx
import websockets

BASE = os.environ.get("PROBE_BASE", "http://127.0.0.1:8566")
QUESTION = os.environ.get("PROBE_Q", "what is digital diplomacy?")


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE, timeout=30) as client:
        r = await client.post("/api/conversation/get_id", json={"conversationId": None})
        r.raise_for_status()
        conv = r.json()
    conv_id = conv.get("conversationId") or conv.get("conversation_id")
    print("conversation:", conv_id, flush=True)

    ws_url = BASE.replace("http://", "ws://") + f"/api/chat/ws/{conv_id}"
    citations, sources = {}, []
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({
            "userIp": "127.0.0.1",
            "message": QUESTION,
            "userType": "general",
        }))
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=180)
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                break
            msg = json.loads(raw)
            status = msg.get("status")
            if status == "citation_url":
                citations[msg["num"]] = msg["url"]
            elif status == "citation_urls":
                citations.update(msg.get("text") or {})
            elif status == "sources":
                sources = msg.get("text") or []
            elif status == "info_message" and msg.get("text") == "See you soon!":
                break

    print(f"\n=== SOURCES ({len(sources)}) ===")
    for i, s in enumerate(sources, 1):
        dl = s.get("deep_link_url") or ""
        url = s.get("url") or ""
        href = dl or (url + "/?diplo-deep-link-text=..." if url else "/?diplo-deep-link-text=...")
        print(f"[{i}] {s.get('title', '')[:70]}")
        print(f"    via_proxy={dl.startswith('https://chatbot-via.diplomacy.edu/')} "
              f"has_dl={bool(dl)} url={'yes' if url else 'MISSING'}")
        print(f"    href={href[:130]}")

    print(f"\n=== CITATIONS ({len(citations)}) ===")
    for num, url in sorted(citations.items(), key=lambda kv: int(kv[0])):
        print(f"[{num}] via_proxy={str(url).startswith('https://chatbot-via.diplomacy.edu/')} {str(url)[:130]}")

    broken = [i for i, s in enumerate(sources, 1) if not (s.get("deep_link_url") or s.get("url"))]
    print(f"\nsources without any link: {broken}")
    return 0 if sources else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
