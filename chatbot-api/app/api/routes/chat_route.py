from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from starlette.requests import Request
from apscheduler.schedulers.background import BackgroundScheduler
from pydantic import ValidationError

from app.models.chat_domain import *
from app.schemas.chat_schema import *
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService

from app.core.singleton import Singleton
from app.core.logging import logger

router = APIRouter()

    
@router.post("/{conversation_id}", summary="Main chat route for asking questions")
async def chat(conversation_id: str, data: ChatRouteRequest, 
               chat_service: ChatService = Depends(ChatService), 
               conversation_service: ConversationService = Depends(ConversationService)
               ) -> ChatRouteResponse:
    
    logger.debug(f"For conversation: {conversation_id}\n new chat request: {data}")
    
    try:
        if(not conversation_service.validate_conversation_id(conversation_id)):
            raise ValueError("Conversation ID not found")
        
        response = await chat_service.handle_chat_request(conversation_id, data)
        
        return ChatRouteResponse(**response)
    
    except ValueError as e:
        logger.exception(f"Error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.put("/feedback/{message_id}", summary="Update feedback for a specific message")
def update_feedback(message_id: str, feedback: FeedbackRouteRequest, 
                    chat_service: ChatService = Depends(ChatService)):
    
    try:

        chat_service.validate_message(message_id)
        chat_service.validate_response(message_id)
        chat_service.validate_feedback_type(feedback.feedback_type)

        chat_service.save_feedback(message_id, feedback)

    except ValueError as e:
        logger.exception(f"Error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
    return {"message": "Feedback updated successfully"}

@router.websocket("/ws/{conversation_id}")
async def websocket_endpoint(conversation_id: str, websocket: WebSocket,
            chat_service: ChatService = Depends(ChatService), 
            conversation_service: ConversationService = Depends(ConversationService)):
    
    await Singleton().websocket_manager.connect(websocket)

    # this is validating the conversation id - if it is not valid, send an error message and return
    if(not conversation_service.validate_conversation_id(conversation_id)):
        await Singleton().websocket_manager.send_message({"status": "error", "text": "Wrong conversation ID. Please try again"}, websocket)
        return

    
    # Send an initial message upon successful WebSocket connection
    await Singleton().websocket_manager.send_message({"status": "info_message", "text": "Welcome! Connection established."}, websocket)
    # infinite loop - keep the websocket open this is 
    while True:
        try:
            data = await websocket.receive_text()
            
            # Validate incoming data against the ChatRouteRequest schema
            try:
                chat_request = ChatRouteRequest.parse_raw(data)
                # send a message to the client to let them know that we are processing their request
                await Singleton().websocket_manager.send_message({"status": "info_message", "text": "Please wait while we process your request..."}, websocket)
                
                response = await chat_service.handle_chat_request_WS(conversation_id, chat_request, websocket)
                
                await Singleton().websocket_manager.send_message({"status": "info_message", "text": "See you soon!"}, websocket)
                await Singleton().websocket_manager.disconnect(websocket)
                break
                
            except ValidationError as e:
                # Handle validation errors, e.g., by sending an error message back to the client
                await Singleton().websocket_manager.send_message({"status": "error", "text": "Invalid request format."}, websocket)
                await Singleton().websocket_manager.disconnect(websocket)
                break
                
        except WebSocketDisconnect:
            await Singleton().websocket_manager.disconnect(websocket)
            break