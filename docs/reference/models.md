# Models Reference

This page documents the database models and enumerations used by FastPKI.

## Enumerations

### `UserRole`

| Value | Description |
|-------|-------------|
| `superuser` | Global admin — full access to everything |
| `admin` | Organization admin — full access within their organization |
| `user` | Regular user — read access plus capability-gated write actions |

### `CertificateType`

| Value | Description |
|-------|-------------|
| `ca` | CA certificate |
| `server` | Server / TLS certificate |
| `client` | Client certificate |

### `CertificateStatus`

| Value | Description |
|-------|-------------|
| `valid` | Active certificate |
| `revoked` | Certificate has been revoked |
| `expired` | Certificate has passed its `not_after` date |

### `PermissionAction`

| Value | Description |
|-------|-------------|
| `read` | View a resource |
| `create_ca` | Create a Certificate Authority |
| `create_cert` | Issue a certificate |
| `revoke_cert` | Revoke a certificate |
| `export_private_key` | View or download a private key |
| `delete_ca` | Delete a Certificate Authority |

### `AuditAction`

| Value | Description |
|-------|-------------|
| `ca_create` | CA created |
| `ca_delete` | CA deleted |
| `ca_export_private_key` | CA private key viewed / exported |
| `cert_create` | Certificate issued |
| `cert_revoke` | Certificate revoked |
| `cert_export_private_key` | Certificate private key viewed / exported |
| `login_success` | Successful login |
| `login_failure` | Failed login attempt |
| `user_create` | User created |
| `user_update` | User updated |
| `org_create` | Organization created |
| `org_delete` | Organization deleted |
| `org_add_user` | User added to organization |
| `org_remove_user` | User removed from organization |

## Database Models

### `Organization`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `int` | Primary key | Auto-increment ID |
| `name` | `str` | Unique, indexed | Organization name |
| `description` | `str` | Nullable | Optional description |
| `created_at` | `datetime` | — | Creation timestamp (UTC) |
| `updated_at` | `datetime` | — | Last update timestamp (UTC) |

### `User`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `int` | Primary key | Auto-increment ID |
| `username` | `str` | Unique, indexed | Login username |
| `email` | `str` | Unique, indexed | Email address |
| `hashed_password` | `str` | — | bcrypt password hash |
| `role` | `UserRole` | — | User role |
| `is_active` | `bool` | Default `true` | Whether the user can authenticate |
| `can_create_ca` | `bool` | Default `false` | Capability flag |
| `can_create_cert` | `bool` | Default `false` | Capability flag |
| `can_revoke_cert` | `bool` | Default `false` | Capability flag |
| `can_export_private_key` | `bool` | Default `false` | Capability flag |
| `can_delete_ca` | `bool` | Default `false` | Capability flag |
| `organization_id` | `int` | FK → `organizations.id`, nullable | Organization membership |
| `created_at` | `datetime` | — | Creation timestamp (UTC) |
| `updated_at` | `datetime` | — | Last update timestamp (UTC) |

### `CertificateAuthority`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `int` | Primary key | Auto-increment ID |
| `name` | `str` | Indexed | CA name |
| `description` | `str` | Nullable | Optional description |
| `subject_dn` | `str` | — | X.509 distinguished name |
| `key_size` | `int` | — | RSA key size |
| `valid_days` | `int` | — | Certificate validity period |
| `private_key` | `str` | — | PEM-encoded private key (may be Fernet-encrypted) |
| `certificate` | `str` | — | PEM-encoded certificate |
| `organization_id` | `int` | FK → `organizations.id`, nullable | Owning organization |
| `created_by_user_id` | `int` | FK → `users.id`, nullable | Creating user |
| `parent_ca_id` | `int` | FK → `certificate_authorities.id`, nullable | Parent CA (null for root CAs) |
| `path_length` | `int` | Nullable | BasicConstraints path length constraint |
| `allow_leaf_certs` | `bool` | Default `true` | Whether this CA can issue leaf certificates |
| `crl_base_url` | `str` | Nullable | Override base URL for CDP/AIA extensions in issued certificates |
| `created_at` | `datetime` | — | Creation timestamp (UTC) |
| `updated_at` | `datetime` | — | Last update timestamp (UTC) |

**Relationships:** A CA can have one `parent_ca` and many `child_cas`, forming a hierarchy.

### `Certificate`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `int` | Primary key | Auto-increment ID |
| `common_name` | `str` | Indexed | Certificate common name |
| `subject_dn` | `str` | — | Full distinguished name |
| `certificate_type` | `CertificateType` | — | `server`, `client`, or `ca` |
| `key_size` | `int` | — | RSA key size |
| `valid_days` | `int` | — | Validity period |
| `status` | `CertificateStatus` | Default `valid` | Current status |
| `private_key` | `str` | Nullable | PEM-encoded private key (may be encrypted) |
| `certificate` | `str` | — | PEM-encoded certificate |
| `serial_number` | `str` | Indexed | Certificate serial number |
| `not_before` | `datetime` | — | Validity start |
| `not_after` | `datetime` | — | Validity end |
| `revoked_at` | `datetime` | Nullable | Revocation timestamp |
| `issuer_id` | `int` | FK → `certificate_authorities.id`, nullable | Issuing CA |
| `organization_id` | `int` | FK → `organizations.id`, nullable | Owning organization |
| `created_by_user_id` | `int` | FK → `users.id`, nullable | Creating user |
| `created_at` | `datetime` | — | Creation timestamp (UTC) |
| `updated_at` | `datetime` | — | Last update timestamp (UTC) |

### `CRLEntry`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `int` | Primary key | Auto-increment ID |
| `serial_number` | `str` | Indexed | Revoked certificate serial number |
| `revocation_date` | `datetime` | — | When the certificate was revoked |
| `reason` | `str` | Nullable | Revocation reason |
| `ca_id` | `int` | FK → `certificate_authorities.id` | CA that issued the revoked certificate |
| `created_at` | `datetime` | — | Entry creation timestamp (UTC) |

### `AuditLog`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `int` | Primary key | Auto-increment ID |
| `created_at` | `datetime` | Indexed | Event timestamp (UTC) |
| `action` | `AuditAction` | Indexed | Type of action |
| `user_id` | `int` | FK → `users.id`, indexed, nullable | User who performed the action |
| `username` | `str` | Nullable | Username at the time of the action |
| `organization_id` | `int` | FK → `organizations.id`, indexed, nullable | Organization context |
| `resource_type` | `str` | Nullable | Type of affected resource |
| `resource_id` | `int` | Nullable | ID of affected resource |
| `detail` | `str` | Nullable | Human-readable description |

### `ServiceAccount`

A non-human principal that holds API tokens, scoped to one organization. See the
[Service Accounts guide](../guides/service-accounts.md).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `int` | Primary key | Auto-increment ID |
| `name` | `str` | Indexed; unique per org | Service account name |
| `description` | `str` | Nullable | Free text |
| `organization_id` | `int` | FK → `organizations.id`, indexed | Owning organization |
| `created_by_user_id` | `int` | FK → `users.id`, nullable | User who created it |
| `created_at` / `updated_at` | `datetime` | — | Timestamps (UTC) |
| `disabled_at` | `datetime` | Nullable | When disabled (`null` = enabled) |
| `can_create_ca` | `bool` | Default `false` | Capability flag |
| `can_create_cert` | `bool` | Default `false` | Capability flag |
| `can_revoke_cert` | `bool` | Default `false` | Capability flag |
| `can_export_private_key` | `bool` | Default `false` | Capability flag |
| `can_delete_ca` | `bool` | Default `false` | Capability flag |

A unique constraint on `(organization_id, name)` enforces per-org name uniqueness.

### `ServiceAccountToken`

A bearer credential for a service account. Only a salted HMAC-SHA256 digest is
stored; the plaintext (`fpki_sa_<public_id>.<secret>`) is shown once at creation.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `int` | Primary key | Auto-increment ID |
| `service_account_id` | `int` | FK → `service_accounts.id`, indexed | Owning account |
| `public_id` | `str` | Indexed, unique | Lookup handle (the `<public_id>` part) |
| `digest` | `str` | — | Hex HMAC-SHA256 of the secret, peppered |
| `pepper_version` | `int` | Default `1` | Pepper version (for rotation) |
| `name` | `str` | Nullable | Human label |
| `created_at` | `datetime` | — | Creation timestamp (UTC) |
| `last_used_at` | `datetime` | Nullable | Last successful auth |
| `expires_at` | `datetime` | Nullable | Expiry (`null` = non-expiring) |
| `revoked` | `bool` | Default `false` | Whether revoked |

!!! note "Resource ownership"
    `CertificateAuthority` and `Certificate` carry a nullable
    `created_by_service_account_id` (FK → `service_accounts.id`) alongside
    `created_by_user_id`, so a resource created by a service account is owned by
    it for authorization purposes.

### `IssuancePolicy`

A deny-by-default allowlist attached 1:1 to a service account, enforced on the
issuance path. See the [Issuance Policies guide](../guides/issuance-policies.md).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `int` | Primary key | Auto-increment ID |
| `service_account_id` | `int` | FK → `service_accounts.id`, unique | Owning account (1:1) |
| `cn_patterns` | `list[str]` | JSON | Glob patterns the requested CN must match |
| `san_dns_patterns` | `list[str]` | JSON | Glob patterns each DNS SAN must match |
| `san_ip_cidrs` | `list[str]` | JSON | CIDRs each IP SAN must fall within |
| `san_email_domains` | `list[str]` | JSON | Allowed domains for email SANs |
| `allowed_ca_ids` | `list[int]` | JSON | Issuing CA ids the account may use |
| `allowed_certificate_types` | `list[str]` | JSON | `server` / `client` |
| `max_validity_days` | `int` | — | Caps requested (and default) validity |
| `created_at` / `updated_at` | `datetime` | — | Timestamps (UTC) |

An empty list denies that dimension entirely (deny-by-default).
