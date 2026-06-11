# API Reference

All endpoints are prefixed with `/api/v1` (configurable via `API_V1_STR`).

Base URL for examples: `http://localhost:8000/api/v1`

---

## Authentication

### `POST /auth/token`

Obtain a JWT access token.

- **Auth required:** No
- **Content-Type:** `application/x-www-form-urlencoded`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `username` | `string` | Yes | Username |
| `password` | `string` | Yes | Password |

**Response** `200`:

```json
{
    "access_token": "eyJ...",
    "token_type": "bearer",
    "refresh_token": "dGhpcyBpcyBhIHJlZnJlc2ggdG9rZW4..."
}
```

**Errors:** `401` — incorrect credentials.

---

### `POST /auth/refresh`

Exchange a refresh token for a new access token and refresh token. The submitted refresh token is immediately invalidated (token rotation).

- **Auth required:** No (refresh token passed in request body)
- **Content-Type:** `application/json`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `refresh_token` | `string` | Yes | A valid, non-expired refresh token |

**Response** `200`:

```json
{
    "access_token": "eyJ...",
    "token_type": "bearer",
    "refresh_token": "dGhpcyBpcyBhIHJlZnJlc2ggdG9rZW4..."
}
```

**Errors:** `401` — token missing, expired, or already revoked.

---

### `POST /auth/logout`

Invalidate the refresh token and access token associated with the current session.

- **Auth required:** Yes (Bearer token)

**Response** `204`: No content.

**Errors:** `401` — not authenticated.

---

### `POST /auth/invalidate`

Invalidate all refresh tokens for the authenticated user, signing out every active session.

- **Auth required:** Yes (Bearer token)

**Response** `204`: No content.

**Errors:** `401` — not authenticated.

---

## Users

### `POST /users/`

Create a new user.

- **Auth required:** Optional (required for admin/superuser roles after first user)
- **Content-Type:** `application/json`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `username` | `string` | Yes | — | Unique username |
| `email` | `string` | Yes | — | Unique email |
| `password` | `string` | Yes | — | Min 8 characters |
| `role` | `string` | No | `user` | `superuser`, `admin`, `user` |
| `organization_id` | `int` | No | `null` | Organization to assign |
| `can_create_ca` | `bool` | No | `false` | Capability flag |
| `can_create_cert` | `bool` | No | `false` | Capability flag |
| `can_revoke_cert` | `bool` | No | `false` | Capability flag |
| `can_export_private_key` | `bool` | No | `false` | Capability flag |
| `can_delete_ca` | `bool` | No | `false` | Capability flag |

**Response** `201`: User object.

**Errors:** `400` — username/email already exists. `403` — insufficient permissions for role.

### `GET /users/`

List all users.

- **Auth required:** Superuser

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip` | `int` | `0` | Offset |
| `limit` | `int` | `100` | Max results |

**Response** `200`: Array of user objects.

### `GET /users/me`

Get the current authenticated user.

- **Auth required:** Any active user

**Response** `200`: User object.

### `GET /users/{user_id}`

Get a user by ID.

- **Auth required:** Self, Admin, or Superuser

**Response** `200`: User object. **Errors:** `403`, `404`.

### `PATCH /users/{user_id}`

Update a user.

- **Auth required:** Self (limited fields), Admin (same org), or Superuser
- **Content-Type:** `application/json`

| Field | Type | Who can set |
|-------|------|-------------|
| `email` | `string` | Self, Admin (same org), Superuser |
| `password` | `string` | Self, Admin (same org), Superuser |
| `role` | `string` | Superuser only |
| `is_active` | `bool` | Superuser only |
| `organization_id` | `int` | Superuser only |
| `can_create_ca` | `bool` | Admin (same org), Superuser |
| `can_create_cert` | `bool` | Admin (same org), Superuser |
| `can_revoke_cert` | `bool` | Admin (same org), Superuser |
| `can_export_private_key` | `bool` | Admin (same org), Superuser |
| `can_delete_ca` | `bool` | Admin (same org), Superuser |

**Response** `200`: Updated user object. **Errors:** `403`, `404`.

### `DELETE /users/{user_id}`

Delete a user.

- **Auth required:** Superuser (cannot delete self)

**Response** `204`. **Errors:** `400` (self-delete), `404`.

---

## Organizations

### `POST /organizations/`

Create an organization.

- **Auth required:** Admin or Superuser
- **Content-Type:** `application/json`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `string` | Yes | Unique name (non-empty) |
| `description` | `string` | No | Description |

**Response** `201`: Organization object. **Errors:** `400` — name already exists.

### `GET /organizations/`

List organizations.

- **Auth required:** Any active user
- Superusers see all. Others see only their own organization.

**Response** `200`: Array of organization objects.

### `GET /organizations/{organization_id}`

Get an organization by ID.

- **Auth required:** Member of the organization, or Superuser

**Response** `200`: Organization object. **Errors:** `403`, `404`.

### `PUT /organizations/{organization_id}`

Update an organization.

- **Auth required:** Admin in the organization, or Superuser
- **Content-Type:** `application/json`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | New name (non-empty) |
| `description` | `string` | New description |

**Response** `200`: Updated organization object. **Errors:** `400`, `403`, `404`.

### `DELETE /organizations/{organization_id}`

Delete an organization.

- **Auth required:** Superuser only
- Organizations with users cannot be deleted.

**Response** `204`. **Errors:** `400` (has users), `404`.

### `POST /organizations/{organization_id}/users/{user_id}`

Add a user to an organization.

- **Auth required:** Admin in the organization, or Superuser

**Response** `200`: Updated user object. **Errors:** `403`, `404`.

### `DELETE /organizations/{organization_id}/users/{user_id}`

Remove a user from an organization.

- **Auth required:** Admin in the organization, or Superuser

**Response** `200`: Updated user object. **Errors:** `403`, `404`.

### `GET /organizations/{organization_id}/users`

List users in an organization.

- **Auth required:** Member of the organization, or Superuser

**Response** `200`: Array of user objects. **Errors:** `403`, `404`.

---

## Certificate Authorities

### `POST /cas/`

Create a new CA.

- **Auth required:** `create_ca` capability, Admin (same org), or Superuser
- **Content-Type:** `application/json`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `string` | Yes | — | CA name |
| `subject_dn` | `string` | Yes | — | X.509 distinguished name |
| `description` | `string` | No | `null` | Description |
| `key_size` | `int` | No | `CA_KEY_SIZE` | RSA key size |
| `valid_days` | `int` | No | `CA_CERT_DAYS` | Validity in days |
| `parent_ca_id` | `int` | No | `null` | Parent CA ID (creates an intermediate) |
| `path_length` | `int` | No | `null` | BasicConstraints path length |
| `allow_leaf_certs` | `bool` | No | `null` | Override leaf cert policy (auto-managed if `null`) |

**Response** `201`: CA detail (includes private key, `is_root`, `allow_leaf_certs`). **Errors:** `400`, `403`.

### `GET /cas/`

List all CAs visible to the current user.

- **Auth required:** Any active user
- Superusers see all. Others see only their organization's CAs.

**Response** `200`: Array of CA objects (without private keys).

### `GET /cas/{ca_id}`

Get a CA by ID.

- **Auth required:** Read access to the CA

**Response** `200`: CA object (without private key). **Errors:** `403`, `404`.

### `GET /cas/{ca_id}/private-key`

Get a CA including its private key.

- **Auth required:** `export_private_key` capability, Admin (same org), or Superuser
- **Audit-logged**

**Response** `200`: CA detail (includes private key). **Errors:** `403`, `404`.

### `GET /cas/{ca_id}/chain`

Get the certificate chain from a CA up to the root.

- **Auth required:** Read access to the CA

**Response** `200`: Array of CA objects ordered from the specified CA to the root. **Errors:** `403`, `404`.

### `GET /cas/{ca_id}/children`

Get direct child CAs of the specified CA.

- **Auth required:** Read access to the CA

**Response** `200`: Array of child CA objects. **Errors:** `403`, `404`.

### `PATCH /cas/{ca_id}`

Assign a CA to an organization. Intended for adopting resources created on
pre-organization instances (where `organization_id` is null) so that
organization-scoped users and service accounts can use them.

- **Auth required:** Superuser
- **Audit-logged**

**Request body:**

```json
{
  "organization_id": 1,
  "cascade": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `organization_id` | integer | Yes | Organization to assign the CA to |
| `cascade` | boolean | No | Also adopt descendant CAs and issued certificates that are org-less (default `false`). Resources owned by a different organization are left untouched, and traversal stops at such CAs |

**Response** `200`: Updated CA object. **Errors:** `400` (unknown organization), `403`, `404`.

### `DELETE /cas/{ca_id}`

Delete a CA and all its certificates. CAs with child CAs cannot be deleted.

- **Auth required:** `delete_ca` capability, Admin (same org), or Superuser
- **Audit-logged**

**Response** `204`. **Errors:** `403`, `404`, `409` (has child CAs).

---

## Certificates

### `POST /certificates/?ca_id={ca_id}`

Issue a new certificate under the specified CA.

- **Auth required:** `create_cert` capability on the CA, Admin (same org), or Superuser
- **Content-Type:** `application/json`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `common_name` | `string` | Yes | — | Common Name |
| `subject_dn` | `string` | Yes | — | Distinguished name |
| `certificate_type` | `string` | Yes | — | `server`, `client`, or `ca` |
| `key_size` | `int` | No | `CERT_KEY_SIZE` | RSA key size |
| `valid_days` | `int` | No | `CERT_DAYS` | Validity in days |
| `include_private_key` | `bool` | No | `true` | Generate a private key |
| `san_dns_names` | `string[]` | No | — | DNS SAN entries (server/dual-purpose only) |
| `san_ip_addresses` | `string[]` | No | — | IP SAN entries (server/dual-purpose only) |
| `san_email_addresses` | `string[]` | No | — | Email SAN entries (client/dual-purpose only) |

For `server` and `dual_purpose` certificates, if no DNS SANs are supplied the Common Name is auto-added as a DNS SAN. For `client` certificates, if no email SANs are supplied and the Common Name parses as an email address, it is auto-added. SAN type restrictions are enforced: servers reject email SANs, clients reject DNS and IP SANs.

**Response** `201`: Certificate detail (includes private key if generated). **Errors:** `400` (includes when CA has `allow_leaf_certs=false`, or when SAN types violate the certificate-type restrictions), `403`, `404`.

### `POST /certificates/sign-csr`

Sign an externally-generated Certificate Signing Request. The submitted CSR's subject, SANs, and public key are used as defaults; any explicit fields in the request body override them. The CSR's signature is verified before signing.

- **Auth required:** `create_cert` capability on the resolved CA, Admin (same org), or Superuser
- **Content-Type:** `application/json`
- **Audit-logged**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `csr` | `string` | Yes | — | PEM-encoded CSR |
| `ca_id` | `int` | Conditional | — | Issuing CA by ID (required if `ca_name` omitted) |
| `ca_name` | `string` | Conditional | — | Issuing CA by name, scoped to the caller's org (or globally for superusers) |
| `certificate_type` | `string` | Yes | — | `server`, `client`, or `dual_purpose` |
| `valid_days` | `int` | No | `CERT_DAYS` | Validity in days |
| `common_name` | `string` | No | From CSR | Override the Common Name |
| `subject_dn` | `string` | No | From CSR | Override the full DN |
| `san_dns_names` | `string[]` | No | From CSR | Override DNS SANs |
| `san_ip_addresses` | `string[]` | No | From CSR | Override IP SANs |
| `san_email_addresses` | `string[]` | No | From CSR | Override email SANs |

Exactly one of `ca_id` or `ca_name` must be provided. The `certificate` service never returns a private key for this endpoint — the client retains the key it generated alongside the CSR.

**Response** `201`: Certificate object (no private key). **Errors:** `400` (malformed CSR, invalid signature, SAN-type violation, or CA has `allow_leaf_certs=false`), `403`, `404` (CA not found).

### `GET /certificates/`

List certificates visible to the current user.

- **Auth required:** Any active user

| Parameter | Type | Description |
|-----------|------|-------------|
| `ca_id` | `int` | Filter by issuing CA |

**Response** `200`: Array of certificate objects (without private keys).

### `GET /certificates/{cert_id}`

Get a certificate by ID.

- **Auth required:** Read access to the certificate

**Response** `200`: Certificate object (without private key), including lineage
fields `renewed_from_id` and `renewed_to_ids`. **Errors:** `403`, `404`.

### `GET /certificates/{cert_id}/private-key`

Get a certificate including its private key.

- **Auth required:** `export_private_key` capability, Admin (same org), or Superuser
- **Audit-logged**

**Response** `200`: Certificate detail (includes private key). **Errors:** `403`, `404`.

### `POST /certificates/{cert_id}/revoke`

Revoke a certificate.

- **Auth required:** `revoke_cert` capability, Admin (same org), or Superuser
- **Content-Type:** `application/json`
- **Audit-logged**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `reason` | `string` | No | Revocation reason |

**Response** `200`: Updated certificate object (status `revoked`). **Errors:** `403`, `404`, `409` (already revoked).

### `POST /certificates/{cert_id}/renew`

Issue a new certificate based on an existing one, inheriting subject, SANs, CA,
type, and validity duration. See the
[renewal guide](../guides/renewing-certificates.md).

- **Auth required:** `create_cert` capability, Admin (same org), or Superuser
- **Content-Type:** `application/json`
- **Audit-logged**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `csr` | `string` | For CSR-origin certs | PEM CSR; its subject is ignored |

Send an empty body (`{}`) for server-key certificates; the response includes a
freshly minted private key. For service-account callers, the current issuance
policy is re-evaluated against the inherited parameters.

**Response** `201`: New certificate (with `renewed_from_id` set). **Errors:**
`400` (`csr_required_for_csr_origin_cert` / `csr_not_allowed_for_server_key_cert`
/ `policy_violation`), `403` (`service_account_has_no_policy`), `404`.

---

## Export

All export endpoints return PEM files with `Content-Disposition: attachment` headers.

### `GET /export/ca/{ca_id}/certificate`

Download a CA certificate as PEM.

- **Auth required:** Read access to the CA

### `GET /export/ca/{ca_id}/private-key`

Download a CA private key as PEM.

- **Auth required:** `export_private_key`
- **Audit-logged**

### `GET /export/certificate/{cert_id}`

Download a certificate as PEM.

- **Auth required:** Read access to the certificate

### `GET /export/certificate/{cert_id}/private-key`

Download a certificate private key as PEM.

- **Auth required:** `export_private_key`
- **Audit-logged**
- Returns `404` if no private key exists.

### `GET /export/certificate/{cert_id}/chain`

Download a certificate with its full chain as PEM.

- **Auth required:** Read access to the certificate

---

## Audit Logs

### `GET /audit-logs/`

Query audit logs.

- **Auth required:** Admin (org-scoped) or Superuser (all)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | `AuditAction` | — | Filter by action type |
| `user_id` | `int` | — | Filter by acting user |
| `resource_type` | `string` | — | Filter by resource type |
| `resource_id` | `int` | — | Filter by resource ID |
| `since` | `datetime` | — | Start of time range |
| `until` | `datetime` | — | End of time range |
| `skip` | `int` | `0` | Offset |
| `limit` | `int` | `100` | Max results (1–1000) |

**Response** `200`: Array of audit log objects. **Errors:** `403`.

---

## Service Accounts

Non-human principals that hold API tokens. See the
[Service Accounts guide](../guides/service-accounts.md). All endpoints are
org-scoped; out-of-org accounts return `404`. Management requires an admin user
session — a service-account token cannot call these endpoints.

### `POST /api/v1/service-accounts/`

Create a service account in the caller's organization.

- **Auth required:** Admin (or superuser)
- **Body:** `name` (required, unique in org), `description`, capability flags
  (`can_create_ca`, `can_create_cert`, `can_revoke_cert`,
  `can_export_private_key`, `can_delete_ca`), and `organization_id` (superusers
  only).
- **Response** `201`: service account object. **Errors:** `400` (duplicate name).

### `GET /api/v1/service-accounts/`

List service accounts in the caller's organization. **Response** `200`.

### `GET /api/v1/service-accounts/{id}`

Read one service account. **Response** `200`. **Errors:** `404`.

### `PATCH /api/v1/service-accounts/{id}`

Update name / description / capability flags, or `disabled` (`true`/`false`) to
disable/enable. **Auth:** admin. **Response** `200`. **Errors:** `400`, `404`.

### `DELETE /api/v1/service-accounts/{id}`

Delete the account and cascade-revoke its tokens. **Auth:** admin.
**Response** `204`. **Errors:** `404`.

### `POST /api/v1/service-accounts/{id}/tokens`

Mint a token. **Auth:** admin. **Body:** `name`, `expires_at` (optional).
**Response** `201`: token metadata **plus** a one-time `token` field of the form
`fpki_sa_<public_id>.<secret>` — never returned again. **Errors:** `404`.

### `GET /api/v1/service-accounts/{id}/tokens`

List token metadata (no plaintext, no digest). **Response** `200`. **Errors:** `404`.

### `DELETE /api/v1/service-accounts/{id}/tokens/{token_id}`

Revoke a token. **Auth:** admin. **Response** `204`. **Errors:** `404`.

### `PUT /api/v1/service-accounts/{id}/policy`

Create or replace the [issuance policy](../guides/issuance-policies.md).
**Auth:** admin. **Body:** `cn_patterns`, `san_dns_patterns`, `san_ip_cidrs`,
`san_email_domains`, `allowed_ca_ids`, `allowed_certificate_types`,
`max_validity_days`. **Response** `200`. **Errors:** `404`.

### `GET /api/v1/service-accounts/{id}/policy`

Read the policy. **Response** `200`. **Errors:** `404` (no policy / out of org).

### `DELETE /api/v1/service-accounts/{id}/policy`

Delete the policy (reverts the account to deny-all). **Auth:** admin.
**Response** `204`. **Errors:** `404`.

### Issuance error codes (service accounts)

When the caller is a service account, `POST /api/v1/certificates/` and
`POST /api/v1/certificates/sign-csr` enforce the account's policy. Errors carry
a structured `detail`:

| Status | `detail.code` | When |
|--------|---------------|------|
| `403` | `service_account_has_no_policy` | The account has no policy attached |
| `400` | `policy_violation` | A constraint failed; `detail.field` and `detail.value` name it |

User-bound tokens are not subject to policy and never produce these errors.

---

## Public PKI Endpoints

These endpoints are mounted at the application root (not under `/api/v1/`) and require no authentication. The `{slug}` format is `{name-slug}-{id}` (e.g. `my-root-ca-3`). Both the name prefix and ID are validated.

### `GET /crl/{slug}`

Download the CRL for a CA in DER format.

- **Auth required:** None
- **Content-Type:** `application/pkix-crl`

### `GET /crl/{slug}.pem`

Download the CRL for a CA in PEM format.

- **Auth required:** None
- **Content-Type:** `application/x-pem-file`

### `GET /ca/{slug}.crt`

Download a CA certificate in DER format.

- **Auth required:** None
- **Content-Type:** `application/pkix-cert`

### `GET /ca/{slug}.pem`

Download a CA certificate in PEM format.

- **Auth required:** None
- **Content-Type:** `application/x-pem-file`
