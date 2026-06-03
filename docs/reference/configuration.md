# Configuration Reference

All settings are configured via environment variables and can be placed in a `.env` file in the project root.

## Settings Table

| Variable | Type | Default | Validation | Description |
|----------|------|---------|------------|-------------|
| `API_V1_STR` | `str` | `/api/v1` | — | URL prefix for all API routes |
| `PROJECT_NAME` | `str` | `FastPKI` | — | OpenAPI docs title |
| `DATABASE_URL` | `str` | `sqlite+aiosqlite:///./fastpki.db` | Auto-converts `sqlite` to `sqlite+aiosqlite` and `postgresql` to `postgresql+asyncpg` | Database connection string |
| `DATABASE_CONNECT_ARGS` | `dict` | `{}` | — | Additional connection arguments passed to the engine |
| `CA_KEY_SIZE` | `int` | `4096` | — | Default RSA key size for CAs |
| `CA_CERT_DAYS` | `int` | `3650` | — | Default CA certificate validity (days) |
| `CERT_KEY_SIZE` | `int` | `2048` | — | Default RSA key size for certificates |
| `CERT_DAYS` | `int` | `365` | — | Default certificate validity (days) |
| `SECRET_KEY` | `str` | `supersecretkey` | Must be >= 32 characters; warns if default | JWT signing key |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `int` | `15` | — | Access token lifetime in minutes |
| `REFRESH_TOKEN_EXPIRE_MINUTES` | `int` | `1440` | — | Refresh token lifetime in minutes |
| `ALGORITHM` | `str` | `HS256` | — | JWT signing algorithm |
| `PRIVATE_KEY_ENCRYPTION_KEY` | `str` or `null` | `null` | Must be a valid Fernet key if set | Encryption key for private keys at rest |
| `LOG_LEVEL` | `str` | `INFO` | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | Application log level |
| `BACKEND_CORS_ORIGINS` | `list[str]` | `["*"]` | — | Allowed CORS origins |
| `ALLOW_UNAUTHENTICATED_REGISTRATION` | `bool` | `false` | — | Allow unauthenticated user registration. First user bootstrap always works regardless. |
| `ENABLE_DOCS` | `bool` | `false` | — | When `true`, exposes `/docs` (Swagger UI), `/redoc`, and `/api/v1/openapi.json`. Disabled by default to avoid leaking the API schema in production. |

## Database URL Auto-Conversion

The `DATABASE_URL` value is automatically adjusted at startup:

| You provide | FastPKI uses |
|-------------|-------------|
| `sqlite:///./data/fastpki.db` | `sqlite+aiosqlite:///./data/fastpki.db` |
| `postgresql://user:pass@host/db` | `postgresql+asyncpg://user:pass@host/db` |
| `sqlite+aiosqlite:///...` | No change |
| `postgresql+asyncpg://...` | No change |

## Configuration Source

Settings are loaded by [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) from the following sources (in priority order):

1. Environment variables
2. `.env` file in the working directory

The `extra = "ignore"` setting means unrecognized variables in the `.env` file are silently ignored.
