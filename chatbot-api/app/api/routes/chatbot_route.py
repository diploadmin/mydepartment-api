"""Call a MyDepartment chatbot by its uid or by its public share link.

Two identifiers, one behaviour: the uid can sit in the path or, together with
the share link, in the body. Every route runs the chatbot exactly as the
public embed does, so answers match what the shareable link produces.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.auth.chatbot_auth import verify_chatbot_api_key
from app.schemas.chatbot_schema import (
    ChatbotInfoResponse,
    ChatbotInvokeRequest,
    ChatbotInvokeResponse,
    ChatbotMessageRequest,
)
from app.services.chatbot_service import (
    ChatbotNotFound,
    ChatbotService,
    ChatbotUpstreamError,
    resolve_chatbot_uid,
)

router = APIRouter(dependencies=[Depends(verify_chatbot_api_key)])

_NOT_FOUND = "Chatbot not found, or it is not shared publicly"

# Streamed answers must reach the client token by token, so opt out of the
# buffering nginx applies to proxied responses by default.
_STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _uid(chatbot_uid: Optional[str], public_link: Optional[str] = None) -> str:
    try:
        return resolve_chatbot_uid(chatbot_uid, public_link)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _answer(
    service: ChatbotService, chatbot_uid: str, data: ChatbotMessageRequest
) -> ChatbotInvokeResponse:
    try:
        result = await service.invoke(
            chatbot_uid, data.message, data.thread_id, data.model
        )
    except ChatbotNotFound as exc:
        raise HTTPException(status_code=404, detail=_NOT_FOUND) from exc
    except ChatbotUpstreamError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return ChatbotInvokeResponse(**result)


async def _stream(
    service: ChatbotService, chatbot_uid: str, data: ChatbotMessageRequest
) -> StreamingResponse:
    # Resolving the chatbot and thread up front turns setup failures into real
    # HTTP errors; once the stream starts, the status code is already sent.
    try:
        await service.get_chatbot(chatbot_uid)
        thread_id = data.thread_id or await service.create_thread(chatbot_uid)
    except ChatbotNotFound as exc:
        raise HTTPException(status_code=404, detail=_NOT_FOUND) from exc
    except ChatbotUpstreamError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return StreamingResponse(
        service.stream(chatbot_uid, thread_id, data.message, data.model),
        media_type="text/event-stream",
        headers={**_STREAM_HEADERS, "X-Thread-Id": thread_id},
    )


# Declared before /{chatbot_uid} so these literal paths win the match.
@router.post("/invoke", summary="Ask a chatbot identified in the body")
async def invoke_chatbot(
    data: ChatbotInvokeRequest,
    service: ChatbotService = Depends(ChatbotService),
) -> ChatbotInvokeResponse:
    return await _answer(service, _uid(data.chatbot_uid, data.public_link), data)


@router.post("/stream", summary="Stream a chatbot identified in the body")
async def stream_chatbot(
    data: ChatbotInvokeRequest,
    service: ChatbotService = Depends(ChatbotService),
) -> StreamingResponse:
    return await _stream(service, _uid(data.chatbot_uid, data.public_link), data)


@router.get("/{chatbot_uid}", summary="Details of a publicly shared chatbot")
async def get_chatbot(
    chatbot_uid: str,
    service: ChatbotService = Depends(ChatbotService),
) -> ChatbotInfoResponse:
    resolved = _uid(chatbot_uid)
    try:
        chatbot = await service.get_chatbot(resolved)
    except ChatbotNotFound as exc:
        raise HTTPException(status_code=404, detail=_NOT_FOUND) from exc
    except ChatbotUpstreamError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return ChatbotInfoResponse(
        chatbot_uid=resolved,
        name=chatbot.get("name"),
        description=chatbot.get("description"),
        model=service.default_model(chatbot),
    )


@router.post("/{chatbot_uid}", summary="Ask a chatbot by uid")
async def ask_chatbot(
    chatbot_uid: str,
    data: ChatbotMessageRequest,
    service: ChatbotService = Depends(ChatbotService),
) -> ChatbotInvokeResponse:
    return await _answer(service, _uid(chatbot_uid), data)


@router.post("/{chatbot_uid}/stream", summary="Stream a chatbot answer by uid")
async def stream_chatbot_by_uid(
    chatbot_uid: str,
    data: ChatbotMessageRequest,
    service: ChatbotService = Depends(ChatbotService),
) -> StreamingResponse:
    return await _stream(service, _uid(chatbot_uid), data)
