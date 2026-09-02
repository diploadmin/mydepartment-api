"""FastAPI dependency guarding the MyDepartment chatbot gateway."""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from app.core.config import CHATBOT_INVOKE_API_KEY


async def verify_chatbot_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    """Require CHATBOT_INVOKE_API_KEY, presented as X-API-Key or Bearer.

    With no key configured the check is off and the routes stay behind the
    service-wide IP allowlist, so an unconfigured deployment is never more
    exposed than the rest of the API.
    """
    if not CHATBOT_INVOKE_API_KEY:
        return

    presented = x_api_key
    if not presented and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            presented = token.strip()

    if not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
        )
    if not secrets.compare_digest(presented, CHATBOT_INVOKE_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
