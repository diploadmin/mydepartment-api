"""Langfuse tracing for LangChain / LangGraph runs."""

from __future__ import annotations

import os
from typing import Any

from app.core.config import (
    LANGFUSE_DEBUG,
    LANGFUSE_ENABLED,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    WEBSITE_NAME,
)

_handler = None


def is_langfuse_enabled() -> bool:
    return bool(
        LANGFUSE_ENABLED
        and LANGFUSE_PUBLIC_KEY
        and LANGFUSE_SECRET_KEY
        and LANGFUSE_HOST
    )


def init_langfuse() -> None:
    """Apply SDK env vars and log whether tracing is active."""
    from app.core.logging import logger

    if LANGFUSE_DEBUG:
        os.environ["LANGFUSE_DEBUG"] = "true"

    if not is_langfuse_enabled():
        logger.info("Langfuse tracing disabled (set LANGFUSE_ENABLED=true and API keys)")
        return

    host = LANGFUSE_HOST.rstrip("/")
    os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_HOST"] = host
    os.environ["LANGFUSE_BASE_URL"] = host

    try:
        from langfuse import Langfuse

        Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=host,
        )
    except Exception as e:
        logger.warning(f"Langfuse client init failed (tracing may be incomplete): {e}")

    logger.info(f"Langfuse tracing enabled -> {host}")


def get_langfuse_callback_handler():
    global _handler
    if not is_langfuse_enabled():
        return None
    if _handler is None:
        from langfuse.langchain import CallbackHandler

        # update_trace=True so custom metadata (e.g. filters_debug) lands on the trace.
        _handler = CallbackHandler(
            public_key=LANGFUSE_PUBLIC_KEY or None,
            update_trace=True,
        )
    return _handler


def build_thread_config(thread_id: str) -> dict[str, Any]:
    """Minimal LangGraph config for checkpoint/state updates (no Langfuse callbacks)."""
    return {"configurable": {"thread_id": thread_id}}


def patch_langfuse_trace_metadata(
    extra_metadata: dict[str, Any] | None,
    *,
    trace_id: str | None = None,
) -> None:
    """Attach filter snapshot to the Langfuse trace after a LangGraph run."""
    from app.core.logging import logger

    if not is_langfuse_enabled() or not extra_metadata:
        return

    if trace_id is None:
        handler = get_langfuse_callback_handler()
        trace_id = getattr(handler, "last_trace_id", None) if handler else None
    if not trace_id:
        try:
            from langfuse import get_client

            trace_id = get_client().get_current_trace_id()
        except Exception:
            trace_id = None
    if not trace_id:
        logger.warning("Langfuse filter metadata patch skipped: no trace_id")
        return

    try:
        from langfuse import get_client

        client = get_client()
        span = client.start_span(
            name="filter_metadata",
            trace_context={"trace_id": trace_id},
            metadata=extra_metadata,
        )
        span.update_trace(metadata=extra_metadata)
        span.end()
        client.flush()
        logger.info(f"Langfuse filter metadata patched to trace {trace_id}")
    except Exception as e:
        logger.warning(f"Langfuse filter metadata patch failed: {e}")


def build_langfuse_run_config(
    *,
    thread_id: str,
    run_name: str = "chat",
    user_id: str | None = None,
    tags: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """RunnableConfig for LangGraph invoke/stream with Langfuse metadata."""
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if not is_langfuse_enabled():
        return config

    handler = get_langfuse_callback_handler()
    metadata: dict[str, Any] = {"langfuse_session_id": thread_id}
    if user_id:
        metadata["langfuse_user_id"] = user_id

    tag_list = [WEBSITE_NAME, run_name]
    if tags:
        tag_list.extend(t for t in tags if t)
    metadata["langfuse_tags"] = tag_list
    if extra_metadata:
        metadata.update(extra_metadata)

    config["callbacks"] = [handler]
    config["metadata"] = metadata
    config["run_name"] = run_name
    return config


def langfuse_invoke_config(
    *,
    thread_id: str | None = None,
    run_name: str = "llm",
    user_id: str | None = None,
) -> dict[str, Any]:
    """Config for standalone LangChain invoke/ainvoke calls."""
    if not is_langfuse_enabled():
        return {}

    metadata: dict[str, Any] = {"langfuse_tags": [WEBSITE_NAME, run_name]}
    if thread_id:
        metadata["langfuse_session_id"] = thread_id
    if user_id:
        metadata["langfuse_user_id"] = user_id

    return {
        "callbacks": [get_langfuse_callback_handler()],
        "metadata": metadata,
        "run_name": run_name,
    }


def flush_langfuse() -> None:
    if not is_langfuse_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as e:
        from app.core.logging import logger

        logger.warning(f"Langfuse flush failed: {e}")
