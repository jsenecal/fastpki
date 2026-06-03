# FastPKI - Claude Guide

This file contains important information for Claude when working with this codebase.

## Project Overview

FastPKI is an API-based PKI management system that provides an easier alternative to Easy-RSA. It allows you to create and manage Certificate Authorities, issue certificates, and revoke them through a RESTful API.

## Directory Structure

- `/app`: Main application code (web API)
  - `/api`: API endpoints
  - `/core`: Core configuration
  - `/db`: Database models and session management
  - `/schemas`: Pydantic schemas for API requests/responses
  - `/services`: Business logic services
- `/cli`: CLI tool (`pip install fastpki[cli]`)
- `/tests`: Test suite
- `/docs`: Documentation (built with Zensical)
- `/docker`: Docker configuration
- `/data`: SQLite database files and other persistent data

## Development Workflow

### Essential Commands

Always run these commands before confirming code changes:

```bash
# Format code
ruff format app tests

# Check code quality and types
ruff check app tests
mypy app

# Run tests
pytest
```

### Using Make

You can use the Makefile for common operations:

```bash
# Format code
make format

# Lint and type check code
make lint

# Run tests
make test

# Run tests with coverage
make test-cov
```

## Docker Workflow

For development with Docker:

```bash
# Development mode with SQLite and code reloading
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d

# Production mode with PostgreSQL
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d
```

## Type Checking Guidelines

- All function definitions should have proper type annotations
- Use `X | None` syntax instead of `Optional[X]` (Python 3.10+)
- All class attributes should be properly typed
- Use SQLModel typing conventions for database models

## Dependencies

- FastAPI for the web framework
- SQLModel for the ORM (built on SQLAlchemy)
- Cryptography for PKI operations
- Pydantic for data validation
- UV for package management
- Ruff for linting and formatting
- Mypy for type checking

## Database Configuration

The application supports both SQLite and PostgreSQL:
- Default: SQLite (`sqlite+aiosqlite:///./data/fastpki.db`)
- PostgreSQL: Configure via environment variable (`DATABASE_URL=postgresql+asyncpg://user:pass@host:port/db`)

## Testing

- We follow Test-Driven Development (TDD) practices
- Always create a failing test first, then implement the feature to make it pass
- Follow the Red-Green-Refactor cycle:
  1. Red: Write a failing test for the new functionality
  2. Green: Implement the minimal code needed to make the test pass
  3. Refactor: Clean up the code while ensuring tests still pass
- Use pytest for all tests
- Test coverage should be maintained above 80%
- Tests are in the `/tests` directory
- Tests need to be run within the virtual environment or otherwise will fail due to missing dependencies

## Git Commits

- Write concise, meaningful commit messages
- Use the imperative mood ("Add feature" not "Added feature")
- Follow the conventional commits format (fix:, feat:, docs:, etc.)
- NEVER mention Claude, AI, LLMs, or include any AI-related signatures in commit messages
- Do not include any "Co-Authored-By" statements

### PR Titles and Release Notes

PR titles should follow conventional commits — they drive automated labeling and release notes via Release Drafter (`.github/release-drafter.yml`, `.github/workflows/release-drafter.yml`).

- Title prefix → autolabel → CHANGELOG section:
  - `feat:` → `feature` → **Added**
  - `fix:` → `fix` → **Fixed**
  - `security:` → `security` → **Security**
  - `docs:` → `documentation` → **Documentation**
  - `refactor:` / `perf:` → `refactor` / `performance` → **Changed**
  - `chore:` / `chore(deps):` / `ci:` / `build:` / `test:` → **Maintenance**
  - `feat!:` / `fix!:` / `BREAKING CHANGE:` in body → `breaking` → forces major bump
- Override the semver bump by adding a `major`, `minor`, or `patch` label to the PR.
- Apply `skip-changelog` to exclude a PR from the drafted release notes (e.g. internal-only changes).
- The draft release is updated on every push to `master`; publishing it triggers the `release` event in `ci.yml`, which builds the Docker image and publishes to PyPI.

### Test Workflow

```bash
# Create a new failing test
# Example: Create a test for a new feature
pytest tests/test_new_feature.py -v

# Implement the feature to make the test pass
pytest tests/test_new_feature.py -v

# Refactor as needed while keeping tests passing
pytest -v
```

## Release Checklist

Before tagging a release, verify the following:

- [ ] All tests pass (`pytest`)
- [ ] Linting and type checks pass (`ruff check app cli tests && mypy app cli`)
- [ ] **CLI conforms to the API** — every API endpoint in `app/api/` has a corresponding CLI command in `cli/`. If an endpoint was added, modified, or removed, update the CLI to match.
- [ ] **Documentation is up-to-date** — any new or changed features, endpoints, configuration options, or CLI commands are reflected in `docs/`. Key files to check:
  - `docs/reference/api.md` — API endpoint reference
  - `docs/reference/configuration.md` — environment variables
  - `docs/reference/models.md` — data model fields
  - `docs/guides/` — user-facing guides for new features
  - `docs/security/authentication.md` — auth-related changes
  - `zensical.toml` — navigation updated if new pages were added
- [ ] Documentation builds successfully (`zensical build`)
- [ ] Version bumped (`bumpver update --patch|--minor|--major`) — updates `version` in `pyproject.toml` and the `current_version` in `[tool.bumpver]`, commits and tags
