from motor.motor_asyncio import AsyncIOMotorClient
from mongoengine import connect

from app.core.config import DATABASE_URL, DB_NAME


async def connect_database(app):
    """Connect to MongoDB using config from .env (DB_CONNECTION, DB_NAME)."""
    mongo_url = str(DATABASE_URL)
    db_name = DB_NAME or "chatbot_diplomacy_edu"

    app.state.db = AsyncIOMotorClient(mongo_url)[db_name]
    connect(db=db_name, host=mongo_url)


async def close_database_connection(app):
    app.state.db.client.close()
