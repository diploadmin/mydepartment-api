"""
Service for deep link URL shortening using Redis.
"""
import json
import secrets
import string
from typing import List, Dict, Any, Optional
import redis
from app.utils.redis_client import RedisClient
import logging

logger = logging.getLogger(__name__)

# Constants
DEEP_LINK_PREFIX = "deep_link:"
DEFAULT_TTL = 2592000  # 30 days in seconds
ID_LENGTH = 12  # Base62: a-zA-Z0-9


class DeepLinkService:
    """Service for creating and retrieving short deep links."""
    
    def __init__(self):
        self.redis_client = RedisClient.get_client()
        self.alphabet = string.ascii_letters + string.digits  # a-zA-Z0-9 (base62)
    
    def _generate_id(self, length: int = ID_LENGTH) -> str:
        """Generate random alphanumeric ID."""
        return ''.join(secrets.choice(self.alphabet) for _ in range(length))
    
    def _get_unique_id(self, max_retries: int = 10) -> str:
        """Generate unique ID (retry on collision)."""
        for attempt in range(max_retries):
            dl_id = self._generate_id()
            key = f"{DEEP_LINK_PREFIX}{dl_id}"
            
            # Check if key exists
            if not self.redis_client.exists(key):
                return dl_id
        
        # Failed to generate unique ID
        raise RuntimeError(f"Failed to generate unique ID after {max_retries} attempts")
    
    def create_deep_link(
        self,
        section_text: str,
        sentences: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        ttl: int = DEFAULT_TTL
    ) -> str:
        """
        Create a short deep link and store in Redis.
        
        Args:
            section_text: Section content for context matching
            sentences: List of sentences to highlight
            metadata: Optional metadata (url, title, query)
            ttl: Time-to-live in seconds (default: 30 days)
        
        Returns:
            Short alphanumeric ID (e.g., 'abc123XYZ')
        
        Raises:
            RuntimeError: If failed to generate unique ID
            redis.RedisError: If Redis operation fails
        """
        dl_id = self._get_unique_id()
        key = f"{DEEP_LINK_PREFIX}{dl_id}"
        
        # Prepare payload
        payload = {
            "section_text": section_text,
            "sentences": sentences,
            "metadata": metadata or {}
        }
        
        # Store in Redis with TTL
        try:
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(payload, ensure_ascii=False)
            )
            logger.info(f"Created deep link: {dl_id} (TTL: {ttl}s, sentences: {len(sentences)})")
            return dl_id
        
        except redis.RedisError as e:
            logger.error(f"Failed to create deep link {dl_id}: {e}")
            raise
    
    def create_deep_links_batch(
        self,
        links_data: List[Dict[str, Any]],
        ttl: int = DEFAULT_TTL
    ) -> List[str]:
        """
        Create multiple deep links in a single Redis pipeline (1 round-trip).
        
        Args:
            links_data: List of dicts with 'section_text', 'sentences', 'metadata'
            ttl: Time-to-live in seconds
        
        Returns:
            List of IDs in same order as input (empty string on per-item failure)
        """
        if not links_data:
            return []

        ids = []
        pipe = self.redis_client.pipeline()

        for data in links_data:
            try:
                dl_id = self._get_unique_id()
                key = f"{DEEP_LINK_PREFIX}{dl_id}"
                payload = {
                    "section_text": data["section_text"],
                    "sentences": data["sentences"],
                    "metadata": data.get("metadata") or {}
                }
                pipe.setex(key, ttl, json.dumps(payload, ensure_ascii=False))
                ids.append(dl_id)
            except Exception as e:
                logger.error(f"Failed to prepare deep link in batch: {e}")
                ids.append("")

        try:
            pipe.execute()
            logger.info(f"Batch created {len([i for i in ids if i])} deep links (TTL: {ttl}s)")
        except redis.RedisError as e:
            logger.error(f"Redis pipeline execute failed: {e}")
            ids = [""] * len(links_data)

        return ids
    
    def get_deep_link(self, dl_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve deep link data by ID.
        
        Args:
            dl_id: Short alphanumeric ID
        
        Returns:
            Dict with 'section_text', 'sentences', 'metadata' or None if not found/expired
        """
        # Validate ID format
        if not dl_id.isalnum() or len(dl_id) != ID_LENGTH:
            logger.warning(f"Invalid deep link ID format: {dl_id}")
            return None
        
        key = f"{DEEP_LINK_PREFIX}{dl_id}"
        
        try:
            data = self.redis_client.get(key)
            
            if not data:
                logger.info(f"Deep link not found or expired: {dl_id}")
                return None
            
            payload = json.loads(data)
            logger.info(f"Retrieved deep link: {dl_id} ({len(payload.get('sentences', []))} sentences)")
            
            return payload
        
        except (json.JSONDecodeError, redis.RedisError) as e:
            logger.error(f"Failed to retrieve deep link {dl_id}: {e}")
            return None
    
    def get_ttl(self, dl_id: str) -> int:
        """
        Get remaining TTL for a deep link.
        
        Returns:
            Remaining TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        key = f"{DEEP_LINK_PREFIX}{dl_id}"
        return self.redis_client.ttl(key)
    
    def delete_deep_link(self, dl_id: str) -> bool:
        """Delete a deep link (admin operation)."""
        key = f"{DEEP_LINK_PREFIX}{dl_id}"
        result = self.redis_client.delete(key)
        return result > 0
