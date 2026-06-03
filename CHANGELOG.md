# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- Disable the OpenAPI schema, Swagger UI, and ReDoc endpoints by default. `/docs`, `/redoc`, and `/api/v1/openapi.json` now return 404 unless `ENABLE_DOCS=true` is set, preventing reconnaissance via the published schema in production deployments. ([#5])
- Scope `GET /api/v1/users/{user_id}` to the caller's organization. ADMIN users can now only view users within their own organization; only SUPERUSER retains cross-organization visibility. Previously an ADMIN in one organization could read details (email, role, capabilities, organization membership) of users in any other organization by guessing user IDs. ([#9])

### Changed

- Refactor `app/main.py` into a `create_app()` factory so the FastAPI wiring is testable with arbitrary settings.

### Fixed

- Correct the Swagger UI and ReDoc URLs in the README and installation guide (`/docs` and `/redoc`, not the previously documented `/api/v1/docs` and `/api/v1/redoc`).

## [0.3.5] - 2026-04-14

Earlier releases (`v0.1.0` through `v0.3.5`) are tracked via git tags and the
[GitHub releases page](https://github.com/jsenecal/fastpki/releases). Per-change
notes start being recorded here from this changelog forward.

[Unreleased]: https://github.com/jsenecal/fastpki/compare/v0.3.5...HEAD
[0.3.5]: https://github.com/jsenecal/fastpki/releases/tag/v0.3.5
[#5]: https://github.com/jsenecal/fastpki/issues/5
[#9]: https://github.com/jsenecal/fastpki/issues/9
