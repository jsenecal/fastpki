from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from app.core.config import settings


def build_engine_kwargs(url: str) -> dict[str, Any]:
    """Build create_async_engine kwargs with connection-resilience settings.

    Adds pool pre-ping and recycle so dead/stale pooled connections are
    detected or retired instead of hanging forever, and (for asyncpg only) a
    per-command timeout so a query on a half-open socket raises rather than
    blocking indefinitely. See issue #57.
    """
    connect_args: dict[str, Any] = dict(settings.DATABASE_CONNECT_ARGS)
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False, **connect_args}
    elif url.startswith("postgresql+asyncpg") and settings.DB_COMMAND_TIMEOUT > 0:
        connect_args.setdefault("command_timeout", settings.DB_COMMAND_TIMEOUT)

    return {
        "echo": False,
        "future": True,
        "pool_pre_ping": settings.DB_POOL_PRE_PING,
        "pool_recycle": settings.DB_POOL_RECYCLE,
        "connect_args": connect_args,
    }


engine = create_async_engine(
    str(settings.DATABASE_URL),  # Cast to string for type safety
    **build_engine_kwargs(str(settings.DATABASE_URL)),
)


async def create_db_and_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async_session_factory = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session
