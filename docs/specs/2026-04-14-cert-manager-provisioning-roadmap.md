# cert-manager Automated Provisioning — Roadmap

**Date:** 2026-04-14
**Status:** Design

## Motivation

FastPKI's issuance model today is entirely admin-initiated: an authenticated human (or a human-bound API token) calls `POST /certificates/` or `POST /certificates/sign-csr` and receives a cert. There is no path for automated workloads — notably cert-manager — to obtain certificates without borrowing a human's credentials, and no way to constrain what any given credential can issue.

This roadmap closes that gap for one concrete driver — **cert-manager-based internal TLS automation** — and deliberately leaves other provisioner-class features (ACME, OIDC, cloud IID, one-shot enrollment tokens) out of scope.

## Non-goals

- **Full ACME (RFC 8555) server.** cert-manager supports non-ACME issuers via the External Issuer contract; we don't need to implement ACME to serve cert-manager.
- **OIDC / SSO provisioner.** Human-SSO-driven issuance is a separate driver with a separate roadmap.
- **Cloud metadata provisioners** (AWS IID, GCP metadata, Azure IMDS). Revisit if a concrete workload-identity driver appears.
- **One-shot enrollment tokens.** May fall out naturally from Phase 2 (a policy-scoped short-TTL token); revisit after Phase 3.
- **Reusable named policies.** Start with policy attached 1:1 to a service account; revisit if duplication across service accounts becomes a real pain point.

## Architecture overview

Three in-repo features plus one out-of-repo component:

1. **Service Accounts** — a new non-human principal type that owns tokens, separate from `User`.
2. **Issuance Policies** — deny-by-default allowlists attached to a service account, enforced in the issuance path.
3. **Renewal endpoint** — `POST /certificates/{id}/renew` for both server-key and CSR-origin certs, with lineage and policy re-evaluation.
4. **`fastpki-issuer`** — a separate Go repo implementing cert-manager's External Issuer contract, consuming the above.

Dependencies: Phase 1 → Phase 2 → Phase 2.5 → Phase 3. Phase 3 could technically ship after Phase 1, but deploying workload-held tokens without policy enforcement means a compromised token has org-wide blast radius, so we gate Phase 3 on Phase 2 landing.

## Phase 1 — Service Accounts

### Goal

Introduce a non-human principal type that can hold API tokens and be authorized for certificate issuance, independent of the `User` table.

### Design

- New `ServiceAccount` entity: `id`, `name` (unique within org), `organization_id`, `created_by_user_id`, `created_at`, `disabled_at`, `description`.
- Tokens become polymorphic: a token belongs to **either** a user **or** a service account, never both. The existing bearer-token auth path resolves to a `Principal` union type (`UserPrincipal | ServicePrincipal`); downstream authorization checks operate on the principal.
- Existing permissions (`can_create_cert`, `can_revoke_cert`, etc.) apply to service accounts the same way they apply to users. No new permission names in Phase 1.
- Token lifecycle: create (returns token once, stores hash), rotate, revoke, list. Reuses the Phase-0 token-revocation work.
- Audit log records the service account's id/name as the actor on every action it performs.

### API surface (new)

- `POST /service-accounts/` — create
- `GET /service-accounts/` — list (org-scoped)
- `GET /service-accounts/{id}` — read
- `PATCH /service-accounts/{id}` — rename / disable / enable / edit description
- `DELETE /service-accounts/{id}` — delete (cascade revoke tokens)
- `POST /service-accounts/{id}/tokens` — mint token
- `GET /service-accounts/{id}/tokens` — list tokens (metadata only)
- `DELETE /service-accounts/{id}/tokens/{token_id}` — revoke

### CLI surface

Mirror the API:
- `fastpki service-account create|list|show|update|delete`
- `fastpki service-account token create|list|revoke`

### Out of scope for Phase 1

- No policy enforcement — a service account with `can_create_cert` can issue anything its org allows. This is intentional: Phase 1 is foundational and must land cleanly before constraints arrive.

### Acceptance criteria

- Service account CRUD + token lifecycle endpoints with tests.
- Auth layer resolves a bearer token to a `Principal` union; existing user-token behavior unchanged.
- CLI commands for every endpoint.
- Audit entries record service-account actors.
- Docs updated: `docs/reference/api.md`, `docs/reference/models.md`, new guide `docs/guides/service-accounts.md`.

## Phase 2 — Issuance Policies

### Goal

Constrain what a given service account is allowed to issue, enforced at the issuance path. Deny-by-default: a service account without a policy cannot issue certificates.

### Design

- New `IssuancePolicy` entity, 1:1 with `ServiceAccount` (embedded or separate table — decide during spec). Fields:
  - `cn_patterns: list[str]` — glob patterns matched against the requested CN
  - `san_dns_patterns: list[str]` — glob patterns for each DNS SAN
  - `san_ip_cidrs: list[str]` — CIDRs; each IP SAN must fall inside one
  - `san_email_domains: list[str]` — each email SAN must match `*@<domain>`
  - `allowed_ca_ids: list[int]` — empty means "none"; policies must explicitly list CAs
  - `allowed_certificate_types: list[CertificateType]` — server / client
  - `max_validity_days: int` — caps the requested validity
- Enforcement runs in both `POST /certificates/` and `POST /certificates/sign-csr` when the principal is a service account:
  1. If no policy exists → 403 `service_account_has_no_policy`
  2. Each field is evaluated independently; first failure short-circuits with a structured error naming the field and the offending value
  3. User-bound tokens bypass policy entirely (current behavior preserved; no risk of breaking existing deployments)
- Policy updates take effect on the next issuance request — no caching beyond normal request scope.

### API surface (new)

- `PUT /service-accounts/{id}/policy` — create-or-replace the policy
- `GET /service-accounts/{id}/policy` — read
- `DELETE /service-accounts/{id}/policy` — delete (reverts SA to "can't issue")

### CLI surface

- `fastpki service-account policy set|show|clear`
- Flags for each constraint field with repeatable values for lists.

### Acceptance criteria

- Policy CRUD endpoints with tests.
- Enforcement tests covering each constraint field: positive, negative, and boundary (e.g., CIDR edges).
- SA with no policy cannot issue via either issuance endpoint.
- User-bound tokens continue to issue without policy checks.
- Docs updated: new guide `docs/guides/issuance-policies.md`, policy fields in `docs/reference/models.md`, error codes in `docs/reference/api.md`.

## Phase 2.5 — Renewal endpoint

### Goal

Provide a single endpoint for "issue a new cert based on an existing one" that handles both the server-key and CSR-origin cases, with lineage and policy re-evaluation.

### Design

- New endpoint `POST /certificates/{id}/renew`.
- Request body:
  - `{}` — permitted only when the original cert was issued with a server-generated key. The server mints a fresh key pair and returns the new cert plus the new private key.
  - `{ "csr": "<PEM>" }` — permitted only when the original was issued via `/sign-csr`. The server verifies the CSR and uses its public key.
  - Mismatches return `400` with a clear message (`csr_required_for_csr_origin_cert` / `csr_not_allowed_for_server_key_cert`).
- The new cert inherits subject, SANs, CA, certificate type, and original validity *duration* (not expiry date — the new cert starts now) from the predecessor. Any subject information in a supplied CSR is ignored to prevent drift.
- Policy re-evaluation: if the caller is a service account, the current policy is applied to the inherited parameters. If the policy has tightened and the original parameters no longer comply, renewal is rejected with the same structured error as Phase 2 enforcement.
- Lineage:
  - New column on `Certificate`: `renewed_from_id: int | None`
  - The predecessor is marked superseded but not revoked — revocation remains a separate explicit action.
  - `GET /certificates/{id}` response includes `renewed_from_id` and a `renewed_to_ids` reverse list.

### CLI surface

- `fastpki cert renew <id>` — server-key case
- `fastpki cert renew <id> --csr <path>` — CSR-origin case

### Acceptance criteria

- Endpoint + CLI with tests for both modes and both mismatch errors.
- Lineage fields populated and returned via the read endpoint.
- Policy re-evaluation tests: renewal succeeds under current policy, fails under a tightened policy.
- Docs updated: `docs/reference/api.md`, `docs/guides/renewing-certificates.md`.

## Phase 3 — cert-manager External Issuer

### Goal

A cert-manager-compatible issuer that signs `CertificateRequest` resources by forwarding the embedded CSR to FastPKI's `/sign-csr`, authenticated by a service-account token.

### Design

- New repository: **`fastpki-issuer`** (Go, separate from the FastPKI repo). Cross-linked from FastPKI docs.
- Implements cert-manager's External Issuer contract: watches `CertificateRequest` resources referencing a `FastPKIIssuer` or `FastPKIClusterIssuer` CRD, signs or fails them.
- CRD fields:
  - `url` — FastPKI API base URL
  - `caBundle` — optional CA bundle to trust FastPKI's TLS (for self-signed deployments)
  - `caRef` — FastPKI CA identifier (id or name) that should sign requests
  - `authSecretRef` — reference to a `Secret` containing the service-account bearer token
- Controller flow:
  1. Receive `CertificateRequest` with CSR + requested duration
  2. POST to `/certificates/sign-csr` with CSR, CA selector, duration
  3. On success, populate `CertificateRequest.status.certificate` and `status.ca`
  4. On policy rejection, set a `Denied` condition with the FastPKI error message (cert-manager surfaces this on the parent `Certificate`)
- Packaging: Helm chart in the same repo, example manifests, `kubectl apply`-able all-in-one YAML.

### Testing

- Unit tests on the controller's reconcile logic (mock FastPKI API).
- Integration test in CI: spin up kind, install cert-manager, install fastpki-issuer, run a live FastPKI (via docker-compose), create an Issuer + Certificate, assert the cert is signed and the chain verifies.

### Docs

- New page in FastPKI docs: `docs/guides/cert-manager-integration.md` covering the SA + policy + issuer install flow end-to-end.
- README in the `fastpki-issuer` repo mirroring the install guide.

### Acceptance criteria

- Controller, CRDs, Helm chart shipped in `fastpki-issuer`.
- Integration test passes in CI.
- Documented end-to-end install path from the FastPKI docs site.

## Open questions deferred to per-phase specs

- Phase 1: exact permission naming — do we rename `can_create_cert` to be clearer about service-account applicability, or leave it?
- Phase 2: is the policy a column on `ServiceAccount` or its own table? Decide during spec based on query patterns.
- Phase 2.5: do we mark the predecessor cert superseded automatically on renewal, and if so, does it appear differently in `GET /certificates/`? Decide during spec.
- Phase 3: Helm chart packaging (OCI vs. classic chart repo) and release cadence coupling with FastPKI versions.
