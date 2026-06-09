# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastPKI is an API-based PKI management system that provides an easier alternative to Easy-RSA. It manages Certificate Authorities (including intermediate CA hierarchies), issues server/client/CA certificates, and revokes them with CRL distribution — all through a RESTful API, with a companion CLI client.

## Repository Layout & Packaging (important)

This repo holds two deliverables with different distribution models:

- **`cli/`** — the **published PyPI package** (`pip install fastpki`). It is a thin Typer + httpx client; its only runtime deps are `typer`, `rich`, `httpx` (see `[project]` in `pyproject.toml`). `packages = ["cli"]` means *only* `cli/` ships in the wheel.
- **`app/`** — the **FastAPI server**. It is **not** packaged in the wheel; it runs from source (`uvicorn app.main:app`) or the Docker image. Its dependencies live in the `server` PEP 735 dependency-group, not in `[project.dependencies]`.

Consequence: there are no `[project.optional-dependencies]` extras. Use dependency-groups (`uv sync` installs the default `dev` group), **not** `pip install fastpki[dev]`.

### Directory structure

- `/app` — FastAPI server
  - `/api` — endpoint routers (`auth`, `ca`, `certs`, `pki`, `export`, `audit`, `organizations`, `users`); `deps.py` holds auth dependencies
  - `/core` — `config.py` (pydantic-settings `Settings`)
  - `/db` — `models.py` (SQLModel tables) and `session.py` (async engine/session)
  - `/schemas` — Pydantic request/response models
  - `/services` — business logic (CA, cert, user, organization, permission, token, encryption, audit)
- `/cli` — Typer CLI; one module per command group, `client.py` wraps API calls, `config.py` stores server URL + token
- `/alembic` — database migrations
- `/tests` — `test_api/` (endpoint-level) and `test_services/` (unit-level)
- `/docs` — documentation built with **Zensical** (`zensical.toml`)
- `/docker` — Dockerfile and compose files

## High-Level Architecture

**Layering: API → Service → DB.** Routers in `app/api/` are thin; they validate input via `app/schemas/`, resolve the caller via dependencies in `app/api/deps.py`, then delegate to a service in `app/services/`. Services own all business logic and DB access and raise typed errors from `app/services/exceptions.py`. Keep new logic in services, not routers.

**App factory & lifespan (`app/main.py`).** `create_app()` builds the FastAPI app and is what `app = create_app()` (and tests) call. The lifespan handler: creates tables, runs `encrypt_existing_keys()` (migrates plaintext private keys to encrypted-at-rest), and starts a background `token_gc_loop` that purges expired blocklisted/refresh tokens hourly. Middleware adds security headers, rate limiting (slowapi), and optional CORS.

**Two router mount points.** The API router mounts under `settings.API_V1_STR` (`/api/v1`). The PKI distribution routers (`ca_router`, `crl_router`) mount at **`/ca`** and **`/crl`** — these are public endpoints for serving CA certs and CRLs, intentionally outside the authenticated API prefix.

**AuthN/AuthZ.**
- JWT access (15 min) + refresh (24 h) tokens, `HS256`, validated in `deps.py::_validate_token`.
- Token revocation works two ways: a JTI blocklist (`TokenService.is_token_blocklisted`) and a per-user `tokens_invalidated_at` cutoff that invalidates all tokens issued before it.
- Authorization is centralized in `app/services/permission.py`. The hierarchy: superuser → resource creator → org admin → org read → per-user capability flags (`can_create_ca`, `can_create_cert`, `can_revoke_cert`, `can_export_private_key`, `can_delete_ca`). Roles: `SUPERUSER`, `ADMIN`, `USER`. **All reads/writes must be scoped to the caller's organization** (see commit `a975df5`).

**Encryption at rest.** Private keys are encrypted with a Fernet key from `PRIVATE_KEY_ENCRYPTION_KEY` via `app/services/encryption.py`. If unset, keys are stored plaintext (dev only).

**Config (`app/core/config.py`).** Settings load from env / `.env`. Notable vars: `DATABASE_URL`, `SECRET_KEY` (must be ≥32 chars and not the default in prod), `PRIVATE_KEY_ENCRYPTION_KEY`, `ENABLE_DOCS` (default `False` — `/docs`, `/redoc`, `openapi.json` are off unless enabled), `BACKEND_CORS_ORIGINS`, `AUTH_RATE_LIMIT`, `ALLOW_UNAUTHENTICATED_REGISTRATION`.

## Development Workflow

Work happens in a uv-managed virtualenv. Tests/lint will fail outside it (missing deps).

```bash
# Install dev dependencies (server + tooling). Uses dependency-groups, NOT extras.
uv sync

# Format
ruff format app cli tests

# Lint + type-check (include cli — see note below)
ruff check app cli tests
mypy app cli

# Run the server
uvicorn app.main:app --reload

# Run the full test suite (coverage is configured in pyproject addopts)
pytest

# Run a single test file / test
pytest tests/test_api/test_ca.py -v
pytest tests/test_api/test_ca.py::TestCreateCA::test_create_ca -v
```

The `Makefile` wraps these in `uv run` (`make install|format|lint|test|test-cov|run|docs|docs-serve`), so the targets work without manually activating the venv; `make docs` builds with `zensical`.

## Database Migrations (Alembic)

Models are SQLModel tables in `app/db/models.py`. The server auto-creates tables on startup, but schema changes ship as Alembic migrations.

```bash
# Generate a migration after changing models
alembic revision --autogenerate -m "describe change"

# Apply / roll back
alembic upgrade head
alembic downgrade -1
```

## Type Checking Guidelines

- Annotate every function; mypy runs in strict-ish mode (`disallow_untyped_defs`, `disallow_any_generics`, etc.).
- Use `X | None`, not `Optional[X]`.
- Follow SQLModel typing conventions for models.

## Testing

- TDD: Red → Green → Refactor. Write the failing test first.
- pytest with `asyncio_mode = "auto"`; coverage target >80% (currently ~90%).
- `tests/conftest.py` provides shared fixtures (app, async client, DB session).

## Git Commits

- Imperative mood, conventional commits (`fix:`, `feat:`, `docs:`, …).
- **Never** mention Claude/AI or add `Co-Authored-By` / attribution lines.

### PR Titles & Release Notes

PR titles drive autolabeling and release notes via Release Drafter (`.github/release-drafter.yml`).

- Prefix → label → CHANGELOG section: `feat:`→`feature`→**Added**; `fix:`→`fix`→**Fixed**; `security:`→`security`→**Security**; `docs:`→**Documentation**; `refactor:`/`perf:`→**Changed**; `chore:`/`ci:`/`build:`/`test:`→**Maintenance**.
- `feat!:`/`fix!:` or `BREAKING CHANGE:` in body → `breaking` → major bump. Override the bump with a `major`/`minor`/`patch` label; exclude a PR with `skip-changelog`.
- Publishing the drafted GitHub release triggers the `release` event in `ci.yml`, which builds the Docker image and publishes to PyPI.

## Release Checklist

- [ ] `pytest` passes
- [ ] `ruff check app cli tests && mypy app cli` passes
- [ ] **CLI conforms to the API** — every endpoint in `app/api/` has a matching CLI command in `cli/`. Adding/changing/removing an endpoint means updating the CLI.
- [ ] **Docs updated** — reflect new/changed features in `docs/` (esp. `reference/api.md`, `reference/configuration.md`, `reference/models.md`, `guides/`, `security/authentication.md`) and `zensical.toml` nav.
- [ ] Docs build: `zensical build`
- [ ] Version bumped: `make bump-patch|bump-minor|bump-major` (wraps `bump-my-version bump`). The version lives only in `[project].version` (PEP 621) — bump-my-version reads and updates it directly, then commits and tags `vX.Y.Z`. Requires a clean working tree; preview with `bump-my-version bump --dry-run --verbose <part>`.
