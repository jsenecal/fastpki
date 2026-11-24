import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse, Response

from app.api import api_router
from app.api.auth import limiter
from app.api.pki import ca_router, crl_router
from app.core.config import logger, settings
from app.db.session import create_db_and_tables, get_session
from app.services.encryption import encrypt_existing_keys

# Request fields whose values must never be echoed back in a 422 response.
_SENSITIVE_FIELDS = frozenset({"password", "refresh_token"})
_REDACTED = "[redacted]"


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _REDACTED if k in _SENSITIVE_FIELDS else _redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


async def _redacting_validation_handler(
    request: StarletteRequest, exc: Exception
) -> Response:
    """FastAPI's default 422 handler, minus sensitive values in ``input``.

    Pydantic echoes the offending value (or, for missing fields, the whole
    request body) in each error's ``input``; that would leak passwords.
    """
    assert isinstance(exc, RequestValidationError)
    errors = []
    for err in exc.errors():
        err = dict(err)
        if "input" in err:
            loc: tuple[Any, ...] = tuple(err.get("loc") or ())
            sensitive = bool(loc) and loc[-1] in _SENSITIVE_FIELDS
            err["input"] = _REDACTED if sensitive else _redact(err["input"])
        errors.append(err)
    return await request_validation_exception_handler(
        request, RequestValidationError(errors)
    )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: StarletteRequest,
        call_next: Callable[[StarletteRequest], Awaitable[Response]],
    ) -> Response:
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains"
        )
        return response


async def token_gc_loop() -> None:
    """Periodically clean up expired blocklisted and refresh tokens."""
    from app.db.session import async_session_factory
    from app.services.token import TokenService

    while True:
        try:
            async with async_session_factory() as session:
                token_service = TokenService(session)
                deleted = await token_service.cleanup_expired_tokens()
                if deleted > 0:
                    logger.info("Token GC: cleaned up %d expired entries", deleted)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Token GC: error during cleanup")
        await asyncio.sleep(3600)  # Run every hour


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await create_db_and_tables()
    await encrypt_existing_keys()
    gc_task = asyncio.create_task(token_gc_loop())
    try:
        yield
    finally:
        gc_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await gc_task


def create_app(enable_docs: bool | None = None) -> FastAPI:
    """Build the FastPKI application.

    When `enable_docs` is None, the value is taken from `settings.ENABLE_DOCS`.
    Disabling docs sets `docs_url`, `redoc_url`, and `openapi_url` to None so
    the OpenAPI schema and Swagger UI are not reachable.
    """
    docs_enabled = settings.ENABLE_DOCS if enable_docs is None else enable_docs

    application = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=(f"{settings.API_V1_STR}/openapi.json" if docs_enabled else None),
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        lifespan=lifespan,
    )

    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    application.add_middleware(SlowAPIMiddleware)
    application.add_exception_handler(
        RequestValidationError, _redacting_validation_handler
    )

    if settings.BACKEND_CORS_ORIGINS:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    application.add_middleware(SecurityHeadersMiddleware)

    application.include_router(api_router, prefix=settings.API_V1_STR)
    application.include_router(crl_router, prefix="/crl", tags=["pki"])
    application.include_router(ca_router, prefix="/ca", tags=["pki"])

    @application.get("/")
    async def root() -> dict[str, str]:
        return {"message": "Welcome to FastPKI - API-based PKI management system."}

    @application.get("/health")
    async def health(
        db: AsyncSession = Depends(get_session),  # noqa: B008
    ) -> Response:
        """Liveness/readiness probe that verifies database connectivity.

        K8s probes should target this rather than `/` so a wedged or
        unreachable database is actually detected (issue #57).
        """
        try:
            await db.execute(text("SELECT 1"))
        except Exception:
            logger.exception("Health check failed: database unreachable")
            return JSONResponse(status_code=503, content={"status": "unavailable"})
        return JSONResponse(status_code=200, content={"status": "ok"})

    return application


app = create_app()
