# Phase 1 — Service Accounts (per-phase spec)

**Date:** 2026-06-08
**Status:** Design — resolves the Phase 1 open questions in
`2026-04-14-cert-manager-provisioning-roadmap.md`.
**Implements:** GitHub #28.

## Resolved decisions

| Roadmap open question | Decision |
| --- | --- |
| Permission naming — rename `can_create_cert`? | **Keep existing names.** Service accounts reuse the same five capability flags as users. No new permission names in Phase 1. |
| How tokens map to a principal | **`Principal` union** resolved in `deps.py`; authz operates on the principal. |
| Token format / storage | **NetBox-v2-style** opaque tokens: `fpki_sa_<public_id>.<secret>`, stored as an HMAC-SHA256 digest with a server-side pepper; plaintext shown once. |
| Authorization model for SAs | **Mirror capability flags** + `organization_id`; SA is treated as a `USER`-role principal scoped to its org. |
| Ownership of SA-created resources | **Add `created_by_service_account_id`** to `certificate_authorities` and `certificates`. |

## Data model

### `ServiceAccount` (`service_accounts`)

```
id: int PK
name: str                      # unique within organization
description: str | None
organization_id: int FK organizations.id   # required; SAs are always org-scoped
created_by_user_id: int | None FK users.id
created_at: datetime (tz)
updated_at: datetime (tz)
disabled_at: datetime | None   # None => enabled
# capability flags (mirror User)
can_create_ca: bool = False
can_create_cert: bool = False
can_revoke_cert: bool = False
can_export_private_key: bool = False
can_delete_ca: bool = False
```

Uniqueness: `(organization_id, name)` unique constraint.

### `ServiceAccountToken` (`service_account_tokens`)

```
id: int PK
service_account_id: int FK service_accounts.id (index)
public_id: str                 # unique, indexed — the lookup handle
digest: str                    # hex HMAC-SHA256(secret, pepper)
pepper_version: int = 1        # forward-compat for pepper rotation
name: str | None               # human label for the token
created_at: datetime (tz)
last_used_at: datetime | None
expires_at: datetime | None    # None => non-expiring
revoked: bool = False
```

Field naming note: the public lookup handle is `public_id`, not `key` — "key"
denotes a cryptographic private key everywhere else in this PKI codebase.

### Ownership columns (existing tables)

- `certificate_authorities.created_by_service_account_id: int | None FK service_accounts.id`
- `certificates.created_by_service_account_id: int | None FK service_accounts.id`

Exactly one of `created_by_user_id` / `created_by_service_account_id` is set on
resources created after this change. Legacy rows keep `created_by_user_id`.

### Audit actor columns (`audit_logs`)

- `service_account_id: int | None FK service_accounts.id`
- `service_account_name: str | None`

`AuditAction` gains: `SERVICE_ACCOUNT_CREATE`, `SERVICE_ACCOUNT_UPDATE`,
`SERVICE_ACCOUNT_DELETE`, `SERVICE_ACCOUNT_TOKEN_CREATE`,
`SERVICE_ACCOUNT_TOKEN_REVOKE`.

## Token scheme (NetBox v2)

**String:** `fpki_sa_<public_id>.<secret>`
- `public_id`: ~16 chars, `secrets.token_urlsafe`, stored and indexed.
- `secret`: ~32 bytes, `secrets.token_urlsafe`, never stored.

**Mint:**
1. Generate `public_id` + `secret`.
2. `digest = hmac_sha256(key=pepper, msg=secret).hexdigest()`.
3. Persist `public_id`, `digest`, `pepper_version`.
4. Return the full `fpki_sa_<public_id>.<secret>` string **once**.

**Verify (in `deps.py`):**
1. Bearer starts with `fpki_sa_` → SA path; else fall through to JWT path (unchanged).
2. Strip prefix, split on the first `.` → `(public_id, secret)`. Malformed → 401.
3. Look up token by `public_id`. Missing / `revoked` / past `expires_at` → 401.
4. Recompute digest with `peppers[pepper_version]`; `hmac.compare_digest` vs stored. Mismatch → 401.
5. Load `ServiceAccount`; `disabled_at` set → 401.
6. Best-effort `last_used_at = now()`.
7. Return a `ServicePrincipal`.

**Pepper:** `settings.SERVICE_ACCOUNT_TOKEN_PEPPER` (new). If unset, fall back to
`SECRET_KEY` (same dev-degradation pattern as `PRIVATE_KEY_ENCRYPTION_KEY`).
Stored as a single current pepper at `pepper_version = 1`; the version column
leaves room for rotation without a schema change.

## Principal abstraction

A small value object both human and machine callers resolve to:

```python
@dataclass
class Principal:
    kind: Literal["user", "service_account"]
    id: int
    organization_id: int | None
    role: UserRole                 # SAs synthesize UserRole.USER
    is_active: bool
    display_name: str              # username or service-account name
    can_create_ca: bool
    can_create_cert: bool
    can_revoke_cert: bool
    can_export_private_key: bool
    can_delete_ca: bool
```

- `Principal.from_user(user)` and `Principal.from_service_account(sa)` constructors.
- New dep `get_current_principal` resolves user-JWT **or** SA-token bearer.
- `permission.py` switches to `Principal`. The creator rule becomes dual:
  `kind == "user"` matches `created_by_user_id`; `kind == "service_account"`
  matches `created_by_service_account_id`. Superuser/org-admin/capability rules
  are unchanged (an SA is never SUPERUSER/ADMIN, so it only ever passes via
  creator-match or org-scoped capability flags).

## Endpoint scope (which dep guards what)

- **Issuance/CA/cert/export endpoints** adopt `get_current_principal` so SAs can
  issue. They pass a `Principal` into `permission.py`.
- **`/service-accounts` management, `/users`, `/organizations`, `/auth`** stay
  **human-only** (`get_current_active_user` / admin deps). A service account
  cannot manage users or other service accounts in Phase 1.
- Creating an SA requires org-admin (or superuser); the SA's capability flags
  cannot exceed what an admin may grant.

## API surface

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| POST | `/service-accounts/` | admin | create (name unique in org, capability flags) |
| GET | `/service-accounts/` | user | list, org-scoped |
| GET | `/service-accounts/{id}` | user | read, org-scoped |
| PATCH | `/service-accounts/{id}` | admin | rename / enable / disable / description / flags |
| DELETE | `/service-accounts/{id}` | admin | delete; cascade-revoke tokens |
| POST | `/service-accounts/{id}/tokens` | admin | mint; returns plaintext once |
| GET | `/service-accounts/{id}/tokens` | user | list metadata only (never digest/plaintext) |
| DELETE | `/service-accounts/{id}/tokens/{token_id}` | admin | revoke |

All paths are org-scoped: callers can only see/act on SAs in their own org
(superuser unrestricted), matching commit `a975df5`.

## CLI surface

`cli/service_account.py`, wired into `cli/__init__.py` as `service-account`:
- `create | list | show | update | delete`
- `token create | list | revoke`

Token `create` prints the plaintext once with a clear "store this now" warning.

## Out of scope (Phase 1)

No issuance-policy enforcement (Phase 2 / #29). An SA with `can_create_cert`
can issue anything its org allows. No refresh-token flow for SAs (their tokens
are long-lived and rotated explicitly).

## Test plan (TDD)

- **Service:** SA CRUD org-scoping; name uniqueness; token mint returns plaintext
  once and stores only a digest; revoke; expired/disabled rejection.
- **Auth:** `fpki_sa_` token resolves to a `ServicePrincipal`; malformed/revoked/
  expired/disabled → 401; user JWT path unchanged (regression).
- **Permission:** SA creator-match on `created_by_service_account_id`; SA capability
  flags enforced; cross-org denial; SA never gets admin/superuser powers.
- **API:** every endpoint, including human-only guards (SA token rejected on
  management endpoints) and audit entries recording the SA actor.
