"""Invoke a MyDepartment chatbot by uid or by its public share link.

A chatbot here is a Chatbot Generator assistant. Its system prompt, knowledge
scope, attached documents and org-scoped retrieval parameters are assembled by
the Department backend on every run, so this gateway drives that backend's
public embed endpoints rather than talking to LangGraph directly — the same
path the shareable link takes in a browser. Consequently only chatbots whose
access level is public can be reached: those endpoints are what authorise an
anonymous run.
"""

import json
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from app.core.config import (
    CHATBOT_GENERATOR_API_PATH,
    CHATBOT_INVOKE_TIMEOUT,
    DEPARTMENT_API_BASE_URL,
    DEPARTMENT_API_KEY,
    DEPARTMENT_API_KEY_HEADER,
)
from app.core.logging import logger

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)

# The chatbot graph streams its reply from the ``generate`` node; the other two
# are canvas-era names for the same step, kept so a graph swap does not silently
# produce empty answers.
_ANSWER_NODES = frozenset({"generate", "replyToGeneralInput", "generateFollowup"})

_MAX_SOURCES = 20
_MAX_SUMMARY = 500


class ChatbotNotFound(Exception):
    """Unknown uid, or the chatbot is not shared publicly."""


class ChatbotUpstreamError(Exception):
    """The Department backend or the chatbot run itself failed."""

    def __init__(self, detail: str, status_code: int = 502):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def resolve_chatbot_uid(
    chatbot_uid: Optional[str] = None, public_link: Optional[str] = None
) -> str:
    """Chatbot uid from an explicit value or from a public share link.

    The share link is ``.../coworkers/chatbot/public/<uid>``, optionally with
    embed query params, so the uid is simply the UUID inside it.
    """
    raw = (chatbot_uid or public_link or "").strip()
    if not raw:
        raise ValueError("Provide either chatbot_uid or public_link")
    match = _UUID_RE.search(raw)
    if not match:
        raise ValueError("No chatbot uid found in the value provided")
    return match.group(0).lower()


def _content_text(message: Any) -> str:
    """Text of a serialized LangChain message or streamed chunk.

    Providers disagree on the shape: most send a plain string, Anthropic sends
    a list of content blocks, and astream_events sometimes wraps the chunk as
    ``[node, message]``.
    """
    if isinstance(message, (list, tuple)):
        if not message:
            return ""
        message = message[1] if len(message) > 1 else message[0]
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text"):
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return ""


def _last_message_text(output: Any) -> str:
    """Answer text from a node's output state.

    Short-circuit replies (clarifications, guardrail answers) are returned by
    the node without ever hitting the model, so they never appear in the
    streamed chunks — only here.
    """
    if not isinstance(output, dict):
        return ""
    messages = output.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    return _content_text(messages[-1]).strip()


def _sources(output: Any) -> List[Dict[str, Optional[str]]]:
    if not isinstance(output, dict):
        return []
    retrieved = output.get("retrieved_data")
    if not isinstance(retrieved, list):
        return []
    sources: List[Dict[str, Optional[str]]] = []
    for item in retrieved[:_MAX_SOURCES]:
        if not isinstance(item, dict):
            continue
        # Document hits and web hits reach us with different field names for
        # the same three things.
        summary = item.get("summary") or item.get("gist") or item.get("content")
        sources.append(
            {
                "title": item.get("title"),
                "url": item.get("source_url") or item.get("url"),
                "summary": str(summary)[:_MAX_SUMMARY] if summary else None,
                "source_type": item.get("source_type"),
            }
        )
    return sources


async def _iter_sse(response: httpx.Response) -> AsyncIterator[Tuple[str, str]]:
    """Yield ``(event, data)`` pairs from a text/event-stream response."""
    event = "message"
    data: List[str] = []
    async for line in response.aiter_lines():
        if line.startswith(":"):
            continue
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = "message", []
            continue
        if line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:") :].lstrip())
    if data:
        yield event, "\n".join(data)


class ChatbotService:
    """Thin client for the Department backend's public chatbot endpoints."""

    def __init__(self) -> None:
        base = (DEPARTMENT_API_BASE_URL or "").rstrip("/")
        path = "/" + (CHATBOT_GENERATOR_API_PATH or "").strip("/")
        self.base_url = f"{base}{path}"

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(CHATBOT_INVOKE_TIMEOUT, connect=10.0)
        )

    @staticmethod
    def _fail(action: str, exc: Exception) -> ChatbotUpstreamError:
        logger.error(f"[chatbot] {action} failed: {exc}")
        return ChatbotUpstreamError(f"Chatbot service unreachable ({action})", 503)

    async def list_chatbots(self) -> List[Dict[str, Any]]:
        """Every chatbot in the deployment, with its organization and owner.

        The names behind the ids live in the Department backend's own tables,
        so the catalog is assembled there and only relayed here.
        """
        headers = (
            {DEPARTMENT_API_KEY_HEADER: DEPARTMENT_API_KEY}
            if DEPARTMENT_API_KEY
            else {}
        )
        try:
            async with self._client() as client:
                response = await client.get(
                    f"{self.base_url}/assistants/catalog", headers=headers
                )
        except httpx.HTTPError as exc:
            raise self._fail("chatbot list", exc) from exc

        if response.status_code in (401, 403):
            raise ChatbotUpstreamError(
                "Chatbot catalog refused the service key; check DEPARTMENT_API_KEY",
                502,
            )
        if response.status_code >= 400:
            raise ChatbotUpstreamError(
                f"Chatbot list returned {response.status_code}", 502
            )
        chatbots = response.json()
        if not isinstance(chatbots, list):
            raise ChatbotUpstreamError("Chatbot list returned no catalog", 502)
        return chatbots

    async def get_chatbot(self, chatbot_uid: str) -> Dict[str, Any]:
        """Public chatbot record, or ChatbotNotFound if it is not shared."""
        try:
            async with self._client() as client:
                response = await client.post(
                    f"{self.base_url}/assistants/public/search",
                    json={"assistant_id": chatbot_uid},
                )
        except httpx.HTTPError as exc:
            raise self._fail("chatbot lookup", exc) from exc

        if response.status_code == 404:
            raise ChatbotNotFound(chatbot_uid)
        if response.status_code >= 400:
            raise ChatbotUpstreamError(
                f"Chatbot lookup returned {response.status_code}", 502
            )
        chatbot = response.json()
        if not isinstance(chatbot, dict) or not chatbot.get("assistant_id"):
            raise ChatbotNotFound(chatbot_uid)
        return chatbot

    async def create_thread(self, chatbot_uid: str) -> str:
        """Start a conversation attributed to this chatbot."""
        try:
            async with self._client() as client:
                response = await client.post(
                    f"{self.base_url}/threads/public",
                    json={"metadata": {"assistant_id": chatbot_uid}},
                )
        except httpx.HTTPError as exc:
            raise self._fail("thread create", exc) from exc

        if response.status_code >= 400:
            raise ChatbotUpstreamError(
                f"Thread create returned {response.status_code}", 502
            )
        thread_id = (response.json() or {}).get("thread_id")
        if not thread_id:
            raise ChatbotUpstreamError("Thread create returned no thread_id", 502)
        return thread_id

    @staticmethod
    def default_model(chatbot: Dict[str, Any]) -> Optional[str]:
        configurable = (chatbot.get("config") or {}).get("configurable") or {}
        return configurable.get("defaultModelName")

    def _run_body(
        self, chatbot_uid: str, message: str, model: Optional[str]
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "assistant_id": chatbot_uid,
            "input": {"messages": [{"role": "user", "content": message}]},
        }
        if model:
            body["config"] = {"configurable": {"customModelName": model}}
        return body

    def _run_url(self, thread_id: str) -> str:
        return f"{self.base_url}/public/threads/{thread_id}/runs/stream"

    @staticmethod
    async def _upstream_detail(response: httpx.Response) -> str:
        try:
            return (await response.aread()).decode("utf-8", "ignore")[:200]
        except Exception:
            return ""

    async def invoke(
        self,
        chatbot_uid: str,
        message: str,
        thread_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run one turn and return the finished answer."""
        chatbot = await self.get_chatbot(chatbot_uid)
        thread_id = thread_id or await self.create_thread(chatbot_uid)

        streamed: List[str] = []
        final_answer = ""
        sources: List[Dict[str, Optional[str]]] = []
        run_id: Optional[str] = None
        error: Optional[str] = None

        try:
            async with self._client() as client:
                async with client.stream(
                    "POST",
                    self._run_url(thread_id),
                    json=self._run_body(chatbot_uid, message, model),
                ) as response:
                    if response.status_code == 404:
                        raise ChatbotNotFound(chatbot_uid)
                    if response.status_code >= 400:
                        detail = await self._upstream_detail(response)
                        raise ChatbotUpstreamError(
                            f"Chatbot run returned {response.status_code}: {detail}",
                            502,
                        )

                    async for event, raw in _iter_sse(response):
                        try:
                            payload = json.loads(raw) if raw else {}
                        except json.JSONDecodeError:
                            continue

                        if event == "metadata":
                            run_id = run_id or payload.get("run_id")
                            continue
                        if event == "error":
                            error = str(payload.get("error") or "Chatbot run failed")
                            continue
                        if event != "events" or not isinstance(payload, dict):
                            continue

                        node = (payload.get("metadata") or {}).get("langgraph_node")
                        if node not in _ANSWER_NODES:
                            continue
                        inner = payload.get("event")
                        data = payload.get("data") or {}
                        if inner == "on_chat_model_stream":
                            streamed.append(_content_text(data.get("chunk")))
                        elif inner == "on_chain_end":
                            output = data.get("output")
                            final_answer = _last_message_text(output) or final_answer
                            sources = _sources(output) or sources
        except httpx.HTTPError as exc:
            raise self._fail("chatbot run", exc) from exc

        answer = final_answer or "".join(streamed).strip()
        if not answer:
            raise ChatbotUpstreamError(error or "Chatbot returned no answer", 502)

        return {
            "chatbot_uid": chatbot_uid,
            "chatbot_name": chatbot.get("name"),
            "thread_id": thread_id,
            "run_id": run_id,
            "answer": answer,
            "model": model or self.default_model(chatbot),
            "sources": sources,
        }

    async def stream(
        self,
        chatbot_uid: str,
        thread_id: str,
        message: str,
        model: Optional[str] = None,
    ) -> AsyncIterator[bytes]:
        """Forward the run's server-sent events untouched.

        The thread is created by the caller so its id can be returned in a
        response header before the first byte of the stream is written.
        """
        try:
            async with self._client() as client:
                async with client.stream(
                    "POST",
                    self._run_url(thread_id),
                    json=self._run_body(chatbot_uid, message, model),
                ) as response:
                    if response.status_code >= 400:
                        detail = await self._upstream_detail(response)
                        logger.error(
                            f"[chatbot] stream returned {response.status_code}: {detail}"
                        )
                        yield _sse_error(f"Chatbot run returned {response.status_code}")
                        return
                    async for chunk in response.aiter_bytes():
                        yield chunk
        except httpx.HTTPError as exc:
            logger.error(f"[chatbot] stream failed: {exc}")
            yield _sse_error("Chatbot service unreachable")


def _sse_error(message: str) -> bytes:
    return f"event: error\ndata: {json.dumps({'error': message})}\n\n".encode()
