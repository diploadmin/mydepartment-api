from contextlib import asynccontextmanager

from fastapi import FastAPI
from typing import Callable
from loguru import logger

from app.db.events import connect_database, close_database_connection
from app.core.singleton import Singleton, init_graph_expansion, shutdown_graph_expansion
from app.core.langfuse_tracing import init_langfuse, flush_langfuse

# function to create start app handler
def create_start_app_handler(app: FastAPI):
    async def start_app() -> None:
        init_langfuse()
        await connect_database(app)
        Singleton()
        await init_graph_expansion()
    return start_app


def create_stop_app_handler(app: FastAPI):
    async def stop_app() -> None:
        await close_database_connection(app)
        await Singleton().websocket_manager.close_all_connections()
        await shutdown_graph_expansion()
        Singleton().weaviate_client.close()
        flush_langfuse()
    return stop_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks; Starlette 1.x dropped the on_event API."""
    await create_start_app_handler(app)()
    try:
        yield
    finally:
        await create_stop_app_handler(app)()