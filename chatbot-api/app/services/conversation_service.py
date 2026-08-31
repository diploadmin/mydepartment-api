from datetime import datetime
from app.core.logging import logger
from app.core.singleton import Singleton
from app.models.chat_domain import *
from app.schemas.chat_schema import *
from app.services.chatService import *

class ConversationService:
    def __init__(self):
        self.conversation_history = Singleton().conversation_history

    def validate_conversation_id(self, conversation_id: str):
        if conversation_id is None or  conversation_id == '' or conversation_id not in self.conversation_history:
           return False
        else:
            return True 

    def generate_conversation_id(self):
        import random,string
        import os
        
        return 'conversation_' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))