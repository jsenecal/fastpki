import logging
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fastpki")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "FastPKI"

    # Database settings
    DATABASE_URL: str | None = "sqlite+aiosqlite:///./fastpki.db"
    DATABASE_CONNECT_ARGS: dict[str, Any] = {}

    @field_validator("DATABASE_URL")
    def validate_database_url(cls, v: str | None) -> Any:  # noqa: N805
        if v and v.startswith("sqlite"):
            if not v.startswith("sqlite+aiosqlite"):
                return v.replace("sqlite", "sqlite+aiosqlite")
            return v
        if v and v.startswith("postgresql"):
            return v.replace("postgresql", "postgresql+asyncpg")
        return v

    # CA settings
    CA_KEY_SIZE: int = 4096
    CA_CERT_DAYS: int = 3650  # 10 years
    CERT_KEY_SIZE: int = 2048
    CERT_DAYS: int = 365  # 1 year

    # Security settings
    SECRET_KEY: str = "supersecretkey"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15  # 15 minutes
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    ALGORITHM: str = "HS256"

    # Rate limiting
    AUTH_RATE_LIMIT: str = "5/minute"

    @field_validator("SECRET_KEY")
    def validate_secret_key(cls, v: str) -> str:  # noqa: N805
        if v == "supersecretkey":
            import warnings

            warnings.warn(
                "SECRET_KEY is set to the default value. "
                "Set a secure SECRET_KEY in production.",
                stacklevel=1,
            )
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")  # noqa: TRY003
        return v

    # Service-account API token pepper (NetBox-v2-style token hashing).
    # If unset, falls back to SECRET_KEY (dev-only degradation).
    SERVICE_ACCOUNT_TOKEN_PEPPER: str | None = None

    # Private key encryption
    PRIVATE_KEY_ENCRYPTION_KEY: str | None = None

    @field_validator("PRIVATE_KEY_ENCRYPTION_KEY")
    def validate_encryption_key(cls, v: str | None) -> str | None:  # noqa: N805
        if v is None:
            return v
        from cryptography.fernet import Fernet

        try:
            Fernet(v.encode("utf-8"))
        except Exception as e:
            raise ValueError(  # noqa: TRY003
                "PRIVATE_KEY_ENCRYPTION_KEY must be a valid "
                "Fernet key. Generate one with: python -c "
                "'from cryptography.fernet import Fernet; "
                "print(Fernet.generate_key().decode())'"
            ) from e
        return v

    # Logging
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # CORS settings
    BACKEND_CORS_ORIGINS: list[str] = []

    @field_validator("BACKEND_CORS_ORIGINS")
    def validate_cors_origins(cls, v: list[str]) -> list[str]:  # noqa: N805
        if "*" in v:
            import warnings

            warnings.warn(
                "BACKEND_CORS_ORIGINS contains '*'. This allows any website to "
                "make authenticated cross-origin requests. Do not use in production.",
                stacklevel=1,
            )
        return v

    # Registration settings
    ALLOW_UNAUTHENTICATED_REGISTRATION: bool = False

    # API documentation exposure
    # Defaults to False so the OpenAPI schema isn't leaked in production. Set
    # ENABLE_DOCS=true to expose /docs, /redoc, and /api/v1/openapi.json.
    ENABLE_DOCS: bool = False


settings = Settings()

# Update logging level based on settings
log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logger.setLevel(log_level)
