from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
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


@asynccontextmanager
async def unit_of_work(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Open a session that commits once on success and rolls back on error.

    Services only ``add``/``flush``; the unit of work owns the single commit so
    a mutation and its audit row are durable together or not at all (#55).
    Closing the session without committing rolls back any pending work.
    """
    async with factory() as session:
        yield session
        await session.commit()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Request-scoped unit of work.

    Must be declared as ``Depends(get_session, scope="function")``: with the
    default ``"request"`` scope FastAPI runs this teardown (the commit) only
    *after* the response has been sent, so a failed commit would still reach
    the client as a success. ``"function"`` scope tears down as soon as the
    endpoint returns, before the response is sent, so a commit failure
    surfaces as a 500.
    """
    async with unit_of_work(async_session_factory) as session:
        yield session
