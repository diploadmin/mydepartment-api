"""Request/response models for the MyDepartment chatbot gateway.

Plain snake_case models (not RWSchema) so the external contract stays the same
in the request body, the response and the OpenAPI schema.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class ChatbotMessageRequest(BaseModel):
    """Body for the routes that carry the chatbot uid in the path."""

    message: str = Field(min_length=1)
    # Omit to start a new conversation. Pass back the thread_id from a previous
    # answer to continue that conversation with its history intact.
    thread_id: Optional[str] = None
    # Overrides the chatbot's own default model.
    model: Optional[str] = None


class ChatbotInvokeRequest(ChatbotMessageRequest):
    """Body for the routes that identify the chatbot in the payload."""

    chatbot_uid: Optional[str] = None
    # The shareable link handed out by the UI; the uid is parsed out of it.
    public_link: Optional[str] = None

    @model_validator(mode="after")
    def _require_identifier(self) -> "ChatbotInvokeRequest":
        if not (self.chatbot_uid or self.public_link):
            raise ValueError("Provide either chatbot_uid or public_link")
        return self


class ChatbotSource(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None
    source_type: Optional[str] = None


class ChatbotInvokeResponse(BaseModel):
    chatbot_uid: str
    chatbot_name: Optional[str] = None
    thread_id: str
    run_id: Optional[str] = None
    answer: str
    model: Optional[str] = None
    sources: List[ChatbotSource] = []


class ChatbotInfoResponse(BaseModel):
    chatbot_uid: str
    name: Optional[str] = None
    description: Optional[str] = None
    model: Optional[str] = None


class ChatbotListItem(BaseModel):
    chatbot_uid: str
    name: Optional[str] = None
    organization_id: Optional[str] = None
    organization_name: Optional[str] = None
    owner: Optional[str] = None
    owner_email: Optional[str] = None
    # Only a public chatbot answers on its link; for the rest it is the link
    # the owner would have to share first.
    public_link: Optional[str] = None
    access_level: Optional[str] = None


class ChatbotListResponse(BaseModel):
    count: int
    chatbots: List[ChatbotListItem] = []
