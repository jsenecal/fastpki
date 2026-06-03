# FastPKI

[![CI](https://github.com/jsenecal/FastPKI/actions/workflows/ci.yml/badge.svg)](https://github.com/jsenecal/FastPKI/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jsenecal/FastPKI/graph/badge.svg)](https://codecov.io/gh/jsenecal/FastPKI)
[![Docs](https://github.com/jsenecal/FastPKI/actions/workflows/docs.yml/badge.svg)](https://jsenecal.github.io/fastpki/)

FastPKI is an API-based PKI management system that provides an easier alternative to Easy-RSA. It allows you to create and manage Certificate Authorities, issue certificates, and revoke them through a RESTful API.

Full documentation: [jsenecal.github.io/fastpki](https://jsenecal.github.io/fastpki/)

## Features

- Create and manage Certificate Authorities (CAs)
- Intermediate CA hierarchy with path length constraints
- Automatic `allow_leaf_certs` policy enforcement
- Issue certificates (server, client, and CA)
- Revoke certificates
- Organization management for multi-tenant deployments
- Role-based access control (SUPERUSER, ADMIN, USER)
- Per-user capability flags for fine-grained permissions
- Ownership-based access control on CAs and certificates
- RESTful API for easy integration
- Compatible with SQLite and PostgreSQL
- Database migrations with Alembic
- Docker support for easy deployment
- Type checking with mypy
- Code quality with ruff linter and formatter

## Requirements

- Python 3.9+
- FastAPI
- SQLModel
- Cryptography
- Docker (optional)

## Quick Start

### Using the Container Image

Pre-built images are available on GitHub Container Registry:

```bash
docker pull ghcr.io/jsenecal/fastpki:latest
```

Available tags:
- `latest` — Latest release
- `<version>` — Specific release (e.g. `0.1.0`)
- `master` — Latest build from master branch

### Using Docker Compose

```bash
# Clone the repository
git clone https://github.com/jsenecal/fastpki.git
cd fastpki

# Create a .env file from the example
cp .env.example .env

# Create data directory for SQLite database
mkdir -p data

# Development mode with SQLite (recommended for local development)
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d
```

For production deployments with PostgreSQL:

```bash
# Production mode with PostgreSQL
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d
```

The API will be available at http://localhost:8000

### Local Development

```bash
# Clone the repository
git clone https://github.com/jsenecal/fastpki.git
cd fastpki

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies with uv (faster)
uv pip install -e ".[dev]"

# Create a .env file from the example
cp .env.example .env

# Create data directory for SQLite database
mkdir -p data

# Run the application
uvicorn app.main:app --reload
```

## Development

### Code Quality

We use the following tools to ensure code quality:

- **ruff**: Fast Python linter and formatter
- **mypy**: Static type checker

Run linting and type checking:

```bash
# Using make
make lint

# Or directly
ruff check app tests
mypy app
```

Format code:

```bash
# Using make
make format

# Or directly
ruff format app tests
```

### Pre-commit Hooks

Install pre-commit hooks to automatically check your code before committing:

```bash
pre-commit install
```

## Documentation

The full user and reference documentation is published at [jsenecal.github.io/fastpki](https://jsenecal.github.io/fastpki/). The source lives in [`docs/`](docs/) and is built with Zensical.

When the application is running, you can also access the automatic API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

These endpoints are disabled by default. Set `ENABLE_DOCS=true` in your environment to expose them (typically only in development).

## Authentication, Authorization & Permissions

### User Roles

FastPKI has three user roles:

- **SUPERUSER** — Full access to all resources across the system
- **ADMIN** — Full access to all resources within their organization
- **USER** — Read access within their organization; write actions require capability flags

### Permission Hierarchy

Permission checks follow this order:

1. **Superusers** have full access to everything
2. **Resource creators** have full access to their own resources
3. **Admins** have full access to all resources within their organization
4. **Users** can read resources within their organization
5. **Per-user capability flags** grant specific write actions to regular users

### Capability Flags

Regular users can be granted fine-grained permissions via boolean capability flags:

| Flag | Description |
|------|-------------|
| `can_create_ca` | Create Certificate Authorities |
| `can_create_cert` | Create certificates |
| `can_revoke_cert` | Revoke certificates |
| `can_export_private_key` | Export private keys |
| `can_delete_ca` | Delete Certificate Authorities |

### First User Creation

FastPKI implements a first-user privilege system for initial setup:

1. The first user created in the system can be assigned any role, including `superuser`. This is a bootstrap mechanism that allows for the initial setup of the system.

2. After the first user is created, only existing superusers can create other users with elevated privileges (`admin` or `superuser` roles).

3. Regular users can only create other regular users.

To create the first superuser:

```bash
# Make a POST request to create the first user with superuser role
curl -X POST http://localhost:8000/api/v1/users/ \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "email": "admin@example.com",
    "password": "securepassword",
    "role": "superuser"
  }'
```

This first superuser can then authenticate and manage other users through the API.

## Database Support

FastPKI supports both SQLite and PostgreSQL:

- **SQLite** (default for development):
  - Data is stored in the `data/fastpki.db` file for persistence
  - Uses the aiosqlite driver for async support
  - Connection string: `sqlite+aiosqlite:///./data/fastpki.db`

- **PostgreSQL** (recommended for production):
  - Uses the asyncpg driver for async support
  - Connection string: `postgresql+asyncpg://postgres:postgres@db:5432/fastpki`
  - Configure using the `DATABASE_URL` environment variable

## Project Structure

```
/alembic              # Database migrations
/app                  # Main application code
  /api                # API endpoints
  /core               # Core configuration
  /db                 # Database models and session management
  /schemas            # Pydantic schemas for API requests/responses
  /services           # Business logic (CA, certificate, user, organization, permission)
/tests                # Test suite (279 tests, 90% coverage)
/docker               # Docker configuration
/data                 # SQLite database files and other persistent data
```

## Testing

The test suite includes 279 tests with 90% code coverage. Tests are written using pytest:

```bash
# Run tests (using make)
make test

# Run tests with coverage
make test-cov

# Or directly
pytest
pytest --cov=app
```

## License

[GNU Affero General Public License v3.0](LICENSE)
