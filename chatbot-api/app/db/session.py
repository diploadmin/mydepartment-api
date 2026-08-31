from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import DATABASE_URL, DB_NAME


def get_database():
    """Get a synchronous MongoDB database connection."""
    client = MongoClient(str(DATABASE_URL))
    return client[DB_NAME or "chatbot_diplomacy_edu"]


def get_messages_collection():
    dbname = get_database()
    return dbname["messages"]