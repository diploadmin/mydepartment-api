from typing import List
from app.models.chat_domain import ChatResponseModel

class ResponseRepository:
    def save(self, message, response, response_time) -> ChatResponseModel:
        response = ChatResponseModel(
            message = message,
            response = response,
            response_time = response_time
        )
        response.save()
        return response

    def get_by_id(self, response_id: str) -> ChatResponseModel:
        return ChatResponseModel.objects(id=response_id).first()
    
    def get_by_message_id(self, message_id: str) -> ChatResponseModel:
        return ChatResponseModel.objects(message=message_id).first()

    def get_all(self) -> List[ChatResponseModel]:
        return list(ChatResponseModel.objects.all())