# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-06-09

### Added

- Service accounts as a first-class principal: a non-human identity that owns API tokens independent of the `users` table. Tokens use a NetBox-v2-style scheme (an HMAC-SHA256 digest with a server-side pepper is stored; the plaintext is shown only once), capability flags mirror users, and the account is org-scoped with full CRUD and token-lifecycle endpoints, a `fastpki service-account` CLI, and audit attribution. ([#28])
- Issuance policies for service accounts: a deny-by-default allowlist (CN and SAN-DNS globs, SAN IP CIDRs, email domains, allowed CA ids, certificate types, and a maximum validity) attached 1:1 to a service account and enforced on `POST /certificates/` and `/certificates/sign-csr`. A service account with no policy cannot issue; user-bound tokens bypass policy. ([#29])
- Certificate renewal endpoint `POST /certificates/{id}/renew` for both server-key and CSR-origin certificates. It inherits the predecessor's subject, SANs, CA, type, and validity duration, records `renewed_from_id`/`renewed_to_ids` lineage, and re-evaluates the issuance policy for service-account callers. ([#30])

### Changed

- Refactor `app/main.py` into a `create_app()` factory so the FastAPI wiring is testable with arbitrary settings.
- Switch version management from bumpver to [bump-my-version](https://github.com/callowayproject/bump-my-version), sourcing the version solely from `[project].version` (PEP 621) so there is a single source of truth. ([#43])
- Align the `Makefile`, `scripts/lint.sh`, and README with the project's actual tooling: `uv sync` for installs, `zensical` for docs, and an `app cli tests` lint scope. ([#44])

### Security

- Disable the OpenAPI schema, Swagger UI, and ReDoc endpoints by default. `/docs`, `/redoc`, and `/api/v1/openapi.json` now return 404 unless `ENABLE_DOCS=true` is set, preventing reconnaissance via the published schema in production deployments. ([#5])
- Scope `GET /api/v1/users/{user_id}` to the caller's organization. ADMIN users can now only view users within their own organization; only SUPERUSER retains cross-organization visibility. Previously an ADMIN in one organization could read details (email, role, capabilities, organization membership) of users in any other organization by guessing user IDs. ([#9])

### Fixed

- Correct the Swagger UI and ReDoc URLs in the README and installation guide (`/docs` and `/redoc`, not the previously documented `/api/v1/docs` and `/api/v1/redoc`).

## [0.3.5] - 2026-04-14

Earlier releases (`v0.1.0` through `v0.3.5`) are tracked via git tags and the
[GitHub releases page](https://github.com/jsenecal/fastpki/releases). Per-change
notes start being recorded here from this changelog forward.

[Unreleased]: https://github.com/jsenecal/fastpki/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/jsenecal/fastpki/compare/v0.3.5...v0.4.0
[0.3.5]: https://github.com/jsenecal/fastpki/releases/tag/v0.3.5
[#5]: https://github.com/jsenecal/fastpki/issues/5
[#9]: https://github.com/jsenecal/fastpki/issues/9
[#28]: https://github.com/jsenecal/fastpki/issues/28
[#29]: https://github.com/jsenecal/fastpki/issues/29
[#30]: https://github.com/jsenecal/fastpki/issues/30
[#43]: https://github.com/jsenecal/fastpki/issues/43
[#44]: https://github.com/jsenecal/fastpki/issues/44
