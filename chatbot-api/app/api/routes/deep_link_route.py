"""
API routes for deep link URL shortening.
"""
from fastapi import APIRouter, HTTPException, status
from app.schemas.deep_link_schema import (
    DeepLinkCreate,
    DeepLinkBatchCreate,
    DeepLinkCreateResponse,
    DeepLinkBatchCreateResponse,
    DeepLinkData
)
from app.services.deep_link_service import DeepLinkService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
deep_link_service = DeepLinkService()


@router.post("/create", response_model=DeepLinkCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_deep_link(data: DeepLinkCreate):
    """
    Create a short deep link for sentence highlighting.
    
    **Internal endpoint** - called by chat_service.py during response generation.
    
    Request body:
    ```json
    {
        "section_text": "Full section content...",
        "sentences": ["Sentence 1", "Sentence 2", ...],
        "metadata": {
            "url": "https://...",
            "title": "...",
            "query": "..."
        }
    }
    ```
    
    Returns:
    ```json
    {
        "id": "abc123XYZ",
        "expires_in": 2592000
    }
    ```
    """
    try:
        dl_id = deep_link_service.create_deep_link(
            section_text=data.section_text,
            sentences=data.sentences,
            metadata=data.metadata
        )
        
        return DeepLinkCreateResponse(
            id=dl_id,
            expires_in=2592000  # 30 days
        )
    
    except RuntimeError as e:
        logger.error(f"Failed to generate unique ID: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate unique deep link ID"
        )
    
    except Exception as e:
        logger.error(f"Error creating deep link: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create deep link"
        )


@router.post("/create-batch", response_model=DeepLinkBatchCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_deep_links_batch(data: DeepLinkBatchCreate):
    """
    Create multiple deep links in a single request (batch operation).
    
    **Internal endpoint** - for optimized bulk creation.
    
    Returns IDs in the same order as request.
    If any creation fails, returns empty string for that position.
    """
    try:
        links_data = [
            {
                "section_text": link.section_text,
                "sentences": link.sentences,
                "metadata": link.metadata
            }
            for link in data.links
        ]
        
        ids = deep_link_service.create_deep_links_batch(links_data)
        
        return DeepLinkBatchCreateResponse(ids=ids)
    
    except Exception as e:
        logger.error(f"Error creating deep links batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create deep links batch"
        )


@router.get("/{dl_id}", response_model=DeepLinkData)
async def get_deep_link(dl_id: str):
    """
    Retrieve deep link data for highlighting.
    
    **Public endpoint** - called by WordPress JavaScript (diplo-deep-link-finder.js).
    CORS must be enabled for this endpoint.
    
    Path parameter:
    - `dl_id`: 12-character alphanumeric ID (e.g., 'abc123XYZ')
    
    Returns:
    ```json
    {
        "section_text": "Section content...",
        "sentences": ["Sentence 1", "Sentence 2", ...],
        "metadata": {
            "url": "https://...",
            "title": "...",
            "query": "..."
        }
    }
    ```
    
    Returns 404 if deep link not found or expired.
    """
    # Validate ID format
    if not dl_id.isalnum() or len(dl_id) != 12:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid deep link ID format (must be 12 alphanumeric characters)"
        )
    
    # Retrieve from Redis
    data = deep_link_service.get_deep_link(dl_id)
    
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deep link not found or expired"
        )
    
    return DeepLinkData(**data)
