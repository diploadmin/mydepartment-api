"""FastAPI dependencies for WordPress → API sync authentication."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.core.wp_weaviate_auth import (
    verify_wp_weaviate_credentials,
    wp_weaviate_auth_enabled,
)

security = HTTPBasic(auto_error=False)


async def verify_wp_weaviate_basic_auth(
    credentials: HTTPBasicCredentials | None = Depends(security),
) -> None:
    if not wp_weaviate_auth_enabled():
        return

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )

    if not verify_wp_weaviate_credentials(credentials.username, credentials.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
