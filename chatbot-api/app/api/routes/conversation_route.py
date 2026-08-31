from fastapi import APIRouter, Depends

from app.core.logging import logger
from app.core.singleton import Singleton
from app.schemas.conversation_schema import ConversationGetRequest, ConversationGetResponse
from app.services.conversation_service import ConversationService

router = APIRouter()


# @router.get('/create', summary="Generate conversation id for chat routes")
# def generator_user_id(conversation_service: ConversationService = Depends(ConversationService)) -> ConversationGetResponse:
    
#     conversation_id = conversation_service.generate_conversation_id()

#     # Initialize conversation history for the user
#     Singleton().conversation_history[conversation_id] = []
    
#     logger.debug(Singleton().conversation_history)
    
#     return ConversationGetRequest(conversation_id=conversation_id)
    
'''
    null => treba da ti kreiram i vratim novi
    neki id => treba da ti proverim
        ako je validan vracatm isti
        ako ne vracam novi

'''
@router.post('/get_id', summary="Generate conversation id for chat routes")
def generator_user_id(conversationGetRequest: ConversationGetRequest, 
                      conversation_service: ConversationService = Depends(ConversationService)
                    ) -> ConversationGetResponse:
    
    logger.debug(conversationGetRequest)
    
    if(not conversation_service.validate_conversation_id(conversationGetRequest.conversation_id)):
        conversation_id = conversation_service.generate_conversation_id()

        # Initialize conversation history for the user
        Singleton().conversation_history[conversation_id] = []
        logger.debug(f"New user: {conversation_id}")
    
    else:
        conversation_id = conversationGetRequest.conversation_id
        logger.debug(f"Welcome back user: {conversation_id}")
    
    return {'conversationId':conversation_id}