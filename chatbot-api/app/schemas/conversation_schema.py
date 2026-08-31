# from enum import Enum
# from typing import Optional, List
# from pydantic import BaseModel, EmailStr, HttpUrl
from typing import Optional
from app.schemas.rwschema import RWSchema


class ConversationGetRequest(RWSchema):
    conversation_id: Optional[str] = None
    
class ConversationGetResponse(RWSchema):
    conversation_id: str
    