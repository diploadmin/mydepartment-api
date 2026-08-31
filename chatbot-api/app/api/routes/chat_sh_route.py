"""A/B chat routes: sentence + h1 title hybrid retrieval (sentence_header graph).

Same request/response contract as /api/chat — only the compiled LangGraph /
retriever differs. Point WP API Settings at these URLs for PR testing.
"""

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.schemas.chat_schema import *
from app.services.chat_service import (
    ChatService,
    get_chat_service_sentence_header,
)
from app.services.conversation_service import ConversationService
from app.core.singleton import Singleton
from app.core.logging import logger

router = APIRouter()


@router.post(
    "/{conversation_id}",
    summary="A/B chat: sentence + header (h1) retrieval",
)
async def chat_sentence_header(
    conversation_id: str,
    data: ChatRouteRequest,
    chat_service: ChatService = Depends(get_chat_service_sentence_header),
    conversation_service: ConversationService = Depends(ConversationService),
) -> ChatRouteResponse:
    logger.debug(
        f"[chat-sh] conversation={conversation_id} request={data}"
    )
    try:
        if not conversation_service.validate_conversation_id(conversation_id):
            raise ValueError("Conversation ID not found")
        response = await chat_service.handle_chat_request(conversation_id, data)
        return ChatRouteResponse(**response)
    except ValueError as e:
        logger.exception(f"[chat-sh] Error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"[chat-sh] An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.websocket("/ws/{conversation_id}")
async def websocket_sentence_header(
    conversation_id: str,
    websocket: WebSocket,
    chat_service: ChatService = Depends(get_chat_service_sentence_header),
    conversation_service: ConversationService = Depends(ConversationService),
):
    await Singleton().websocket_manager.connect(websocket)

    if not conversation_service.validate_conversation_id(conversation_id):
        await Singleton().websocket_manager.send_message(
            {"status": "error", "text": "Wrong conversation ID. Please try again"},
            websocket,
        )
        return
    # after connection is established -> send a message to the frontend
    await Singleton().websocket_manager.send_message(
        {
            "status": "info_message",
            "text": "Welcome! Connection established (sentence+header A/B).",
        },
        websocket,
    )

    while True:
        try:
            data = await websocket.receive_text()
            try:
                chat_request = ChatRouteRequest.parse_raw(data)
                await Singleton().websocket_manager.send_message(
                    {
                        "status": "info_message",
                        "text": "Please wait while we process your request...",
                    },
                    websocket,
                )
                await chat_service.handle_chat_request_WS(
                    conversation_id, chat_request, websocket
                )
                await Singleton().websocket_manager.send_message(
                    {"status": "info_message", "text": "See you soon!"},
                    websocket,
                )
                await Singleton().websocket_manager.disconnect(websocket)
                break
            except ValidationError:
                await Singleton().websocket_manager.send_message(
                    {"status": "error", "text": "Invalid request format."},
                    websocket,
                )
                await Singleton().websocket_manager.disconnect(websocket)
                break
        except WebSocketDisconnect:
            await Singleton().websocket_manager.disconnect(websocket)
            break
