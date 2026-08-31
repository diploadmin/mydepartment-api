from fastapi import APIRouter
from urllib.parse import quote_plus

from app.api.routes import chat_route
from app.api.routes import chat_sh_route
from app.api.routes import conversation_route
from app.api.routes import ingest_route
from app.api.routes import deep_link_route
from app.api.routes import debug_route
from app.api.routes import person_route
from app.api.routes import metadata_route
from app.api.routes import custom_filter_routes
from app.api.routes import doc_view_route

router = APIRouter()

router.include_router(conversation_route.router, tags=["Conversation"], prefix="/conversation")
router.include_router(chat_route.router, tags=["Chat"], prefix="/chat")
# A/B: sentence + h1 title hybrid (does not replace /chat)
router.include_router(chat_sh_route.router, tags=["ChatSentenceHeader"], prefix="/chat-sh")
router.include_router(ingest_route.router, tags=["Ingest"], prefix="/ingest")
router.include_router(deep_link_route.router, tags=["DeepLink"], prefix="/deep-link")
router.include_router(debug_route.router, tags=["Debug"], prefix="/debug")
router.include_router(person_route.router, tags=["Person"], prefix="/person")
router.include_router(metadata_route.router, tags=["Metadata"], prefix="/metadata-filters")
router.include_router(custom_filter_routes.router, tags=["CustomFilters"], prefix="/weaviate")
# Public: source cards for URL-less documents link here from the browser.
router.include_router(doc_view_route.router, tags=["DocViewer"], prefix="/doc")
