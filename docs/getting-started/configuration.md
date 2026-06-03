# Configuration

FastPKI is configured through environment variables. You can set them in a `.env` file in the project root or pass them directly to the process / Docker container.

## Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATABASE_URL` | `str` | `sqlite+aiosqlite:///./fastpki.db` | Database connection string. Supports SQLite and PostgreSQL. |
| `SECRET_KEY` | `str` | `supersecretkey` | Key used to sign JWT tokens. **Must be at least 32 characters.** Change this in production. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `int` | `15` | Lifetime of an access token in minutes. |
| `REFRESH_TOKEN_EXPIRE_MINUTES` | `int` | `1440` (24 h) | Lifetime of a refresh token in minutes. |
| `ALGORITHM` | `str` | `HS256` | JWT signing algorithm. |
| `CA_KEY_SIZE` | `int` | `4096` | Default RSA key size for new CAs. |
| `CA_CERT_DAYS` | `int` | `3650` (10 years) | Default validity period for CA certificates. |
| `CERT_KEY_SIZE` | `int` | `2048` | Default RSA key size for issued certificates. |
| `CERT_DAYS` | `int` | `365` (1 year) | Default validity period for issued certificates. |
| `PRIVATE_KEY_ENCRYPTION_KEY` | `str` or `null` | `null` | Fernet key for encrypting private keys at rest. See [Encryption at Rest](../security/encryption.md). |
| `LOG_LEVEL` | `str` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |
| `BACKEND_CORS_ORIGINS` | `list[str]` | `["*"]` | Allowed CORS origins. |
| `ALLOW_UNAUTHENTICATED_REGISTRATION` | `bool` | `false` | Allow unauthenticated user registration. First user bootstrap always works. |
| `ENABLE_DOCS` | `bool` | `false` | When `true`, exposes Swagger UI (`/docs`), ReDoc (`/redoc`), and the OpenAPI schema (`/api/v1/openapi.json`). Disabled by default to avoid leaking the API schema in production. |
| `API_V1_STR` | `str` | `/api/v1` | URL prefix for all API routes. |
| `PROJECT_NAME` | `str` | `FastPKI` | Name shown in the OpenAPI docs title. |

## Example `.env` File

```bash
# Database
DATABASE_URL=sqlite+aiosqlite:///./data/fastpki.db

# For PostgreSQL:
# DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/fastpki

# CA defaults
CA_KEY_SIZE=4096
CA_CERT_DAYS=3650
CERT_KEY_SIZE=2048
CERT_DAYS=365

# Security — change this!
SECRET_KEY=generate-a-secure-random-key-at-least-32-chars-long
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_MINUTES=1440

# Private key encryption (optional)
# PRIVATE_KEY_ENCRYPTION_KEY=

# Logging
# LOG_LEVEL=INFO

# CORS
BACKEND_CORS_ORIGINS=["*"]

# API docs (Swagger UI / ReDoc / OpenAPI schema) — disabled by default
# ENABLE_DOCS=true
```

## Generating a Fernet Encryption Key

```bash
python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
```

Set the output as `PRIVATE_KEY_ENCRYPTION_KEY` in your `.env` file. On the next startup FastPKI will automatically encrypt any existing plaintext private keys in the database.
