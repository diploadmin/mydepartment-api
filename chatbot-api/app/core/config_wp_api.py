"""WordPress → API sync credentials (Weaviate catalog routes)."""

import os

from starlette.config import Config


def _load_config() -> Config:
    env_file = os.getenv("ENV_FILE", ".env")
    if env_file and os.path.isfile(env_file):
        return Config(env_file)
    return Config()


config = _load_config()

WP_WEAVIATE_BASIC_AUTH_USER: str = config(
    "WP_WEAVIATE_BASIC_AUTH_USER", cast=str, default=""
)
WP_WEAVIATE_BASIC_AUTH_PASSWORD: str = config(
    "WP_WEAVIATE_BASIC_AUTH_PASSWORD", cast=str, default=""
)

