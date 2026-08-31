from mongoengine import Document, StringField, DateTimeField, ReferenceField, IntField
from datetime import datetime

# 
class ChatMessageModel(Document):
    user_ip = StringField(required=True)
    message = StringField(required=True)
    timestamp = DateTimeField(default=datetime.utcnow)

    meta = {'collection': 'messages'}  # Specify the collection name

    def __str__(self):
        return f"Message from {self.user_ip}: {self.message} at {self.timestamp}"

class ChatResponseModel(Document):
    message = ReferenceField(ChatMessageModel, required=True)
    response = StringField(required=True)
    response_time = IntField(required=True)
    feedback_type = IntField(default=0)
    feedback_message = StringField(default="")

    meta = {'collection': 'responses'}  # Specify the collection name

    def __str__(self):
        return f"Response to message {self.message.id}: {self.response} with response time {self.response_time}"
