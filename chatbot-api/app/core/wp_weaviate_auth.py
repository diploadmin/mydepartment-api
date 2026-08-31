"""HTTP Basic Auth for GET /api/weaviate/* (WordPress admin sync)."""

from __future__ import annotations

import secrets

from app.core.config_wp_api import (
    WP_WEAVIATE_BASIC_AUTH_PASSWORD,
    WP_WEAVIATE_BASIC_AUTH_USER,
)


def wp_weaviate_auth_enabled() -> bool:
    return bool(WP_WEAVIATE_BASIC_AUTH_USER and WP_WEAVIATE_BASIC_AUTH_PASSWORD)


def verify_wp_weaviate_credentials(username: str, password: str) -> bool:
    if not wp_weaviate_auth_enabled():
        return True
    user_ok = secrets.compare_digest(
        (username or "").encode(),
        WP_WEAVIATE_BASIC_AUTH_USER.encode(),
    )
    pass_ok = secrets.compare_digest(
        (password or "").encode(),
        WP_WEAVIATE_BASIC_AUTH_PASSWORD.encode(),
    )
    return user_ok and pass_ok
