"""
Per-request retrieval parameter overrides using contextvars.

Allows the WordPress frontend to send retrieval config that overrides
the server-side .env defaults for a single request.

Usage in pipeline code:
    from app.core.retrieval_context import get_param
    from app.core.config import HYBRID_ALPHA  # .env default

    alpha = get_param('hybrid_alpha', HYBRID_ALPHA)
"""

# this sets up retreival configuration for one request - this is used to override the default retrieval configuration inside .env
# we need ContextVar to store the overrides for one request becuase if there are multiple requests
# without ContextVar, the overrides would be used in the next request or parallel request

from contextvars import ContextVar
from typing import Any, Optional
# context variable to store the retrieval overrides - if there are multiple requests, the overrides are stored in this context variable
# that way overrides are used in current request and not the next request or parallel request
_overrides: ContextVar[dict] = ContextVar('retrieval_overrides', default={})

# context: called by chat_service.py to set the retrieval overrides for one request
# put the configuration sent from WP (frontend) inside ContextVar - ContextVar is a dictionary that is shared through the app
def set_retrieval_overrides(overrides: dict):
    """Set per-request retrieval parameter overrides."""
    _overrides.set(overrides or {})

# clear the retrieval overrides
def clear_retrieval_overrides():
    """Clear overrides after request completes."""
    _overrides.set({})

# context: called by ai_services/graph.py to get the retrieval parameter from the overrides or the default
# description: shared configuration across multiple app
# parameters:
# - name: str - the name of the parameter - hybrid_alpha, sentence_retrieval_k, paragraph_retrieval_k, combined_retrieval_k, twophase_retrieval_k
# - default: Any = None - the default value - usually from config
# returns:
# - either default value or the value from the overrides dictionarys (ContextVar)
def get_param(name: str, default: Any = None) -> Any:
    """Get a retrieval parameter: per-request override first, then default (usually from config)."""
    # set up the dictionary for one request
    overrides = _overrides.get()
    # put the name of the parameter into the dictionary and get the value - if not found, return the default
    # value can be - HYBRID_ALPHA e.g. this sets up the retrival model for Weaviate query
    val = overrides.get(name)
    if val is not None:
        return type(default)(val) if default is not None and not isinstance(val, type(default)) else val
    return default
