"""
Redis client for deep link storage.
"""
import redis
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# context: this is a singleton class for the Redis client - that way we can have only one instance of the Redis client
# - used by direct_retrieval_node to cache RAG results for a certain time
# description: function to connect to Redis and get the client
class RedisClient:
    """Singleton Redis client for deep link storage."""
    
    _instance: Optional[redis.Redis] = None
    
    @classmethod
    # description: function to get the client - if not exists, create it
    def get_client(cls, host: str = None, port: int = None, db: int = None) -> redis.Redis:
        """Get or create Redis client instance. Reads defaults from app config."""
        # create var to hold all params for connecting to Redis
        if cls._instance is None:
            # from config pull .env variables for Redis host, port and db
            from app.core.config import REDIS_HOST, REDIS_PORT, REDIS_DB
            host = host or REDIS_HOST
            port = port or REDIS_PORT
            db = db if db is not None else REDIS_DB
            # try to connect to Redis - if fails, raise error
            try:
                # create the Redis client - with port, host, db
                # additional configuration: decode responses, socket connect timeout, socket timeout, health check interval
                cls._instance = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    decode_responses=True, # decode responses to True - that way we can get the results as strings
                    socket_connect_timeout=5, # timeout if Redis is not responding
                    socket_timeout=5, # timeout if Redis is not responding
                    health_check_interval=30 # check if Redis is alive
                )
                cls._instance.ping()
                logger.info(f"Redis client connected to {host}:{port}/{db}")
            except redis.RedisError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                cls._instance = None
                raise
    # return the instance of the Redis client
        return cls._instance
    # description: function to close the connection to Redis
    @classmethod
    def close(cls):
        """Close Redis connection."""
        # if there is an instance of the Redis client, close the connection
        if cls._instance:
            cls._instance.close()
            cls._instance = None
            logger.info("Redis client connection closed")
