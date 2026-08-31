from typing import List
from app.models.chat_domain import *

class ChatRepository:
    
    def save(self, user_ip: str, message: str, timestamp: str) -> ChatMessageModel:
        chat = ChatMessageModel(
            user_ip=user_ip,
            message=message,
            timestamp=datetime.fromisoformat(timestamp)  # Convert string to datetime
        )
        chat.save()
        return chat

    def get_by_id(self, chat_id: str) -> ChatMessageModel:
        return ChatMessageModel.objects(id=chat_id).first()

    def get_all(self) -> List[ChatMessageModel]:
        return list(ChatMessageModel.objects.all())
