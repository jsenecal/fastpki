# Service Accounts

Service accounts are **non-human principals** that hold API tokens independently
of the `users` table. They let automated workloads (CI pipelines, the
[cert-manager integration](https://github.com/jsenecal/fastpki), schedulers)
authenticate to FastPKI without borrowing a person's credentials.

A service account:

- belongs to exactly one organization and is always org-scoped;
- carries the same capability flags as a user (`can_create_ca`,
  `can_create_cert`, `can_revoke_cert`, `can_export_private_key`,
  `can_delete_ca`);
- owns one or more **tokens** used as bearer credentials;
- is never an admin or superuser — it authorizes purely via its organization,
  the resources it created, and its capability flags.

!!! note "Management is human-only"
    Creating, updating, deleting service accounts and minting/revoking their
    tokens requires an **admin** (or superuser) **user** session. A service
    account token cannot manage users or other service accounts.

## Create a service account

Requires an admin user. The account is created in the admin's organization.

```bash
curl -s -X POST http://localhost:8000/api/v1/service-accounts/ \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ci-runner",
    "description": "Issues TLS certs for CI",
    "can_create_cert": true
  }' | python -m json.tool
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `name` | yes | — | Unique within the organization |
| `description` | no | `null` | Free text |
| `organization_id` | no | caller's org | Only honored for superusers |
| `can_create_ca` | no | `false` | Capability flag |
| `can_create_cert` | no | `false` | Capability flag |
| `can_revoke_cert` | no | `false` | Capability flag |
| `can_export_private_key` | no | `false` | Capability flag |
| `can_delete_ca` | no | `false` | Capability flag |

With the CLI:

```bash
fastpki service-account create --name ci-runner --can-create-cert
```

## Mint a token

Tokens use a NetBox-v2-style scheme: the server stores only a salted
HMAC-SHA256 **digest** of the secret (peppered with
`SERVICE_ACCOUNT_TOKEN_PEPPER`), never the plaintext. The full token is
returned **once**, at creation — store it immediately, it cannot be recovered.

```bash
curl -s -X POST http://localhost:8000/api/v1/service-accounts/1/tokens \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "ci-token", "expires_at": "2027-01-01T00:00:00Z"}'
```

```json
{
  "id": 1,
  "public_id": "Yx3k…",
  "name": "ci-token",
  "expires_at": "2027-01-01T00:00:00Z",
  "revoked": false,
  "token": "fpki_sa_Yx3k….<secret>"
}
```

The token format is `fpki_sa_<public_id>.<secret>`. Use it as a bearer token on
the issuance endpoints — for example, to issue a certificate (the account needs
`can_create_cert`):

```bash
curl -s -X POST "http://localhost:8000/api/v1/certificates/?ca_id=1" \
  -H "Authorization: Bearer fpki_sa_Yx3k….<secret>" \
  -H "Content-Type: application/json" \
  -d '{
    "common_name": "svc.example.com",
    "subject_dn": "CN=svc.example.com",
    "certificate_type": "server"
  }'
```

Certificates and CAs created by a service account are owned by it
(`created_by_service_account_id`), and every action is attributed to the account
in the audit log.

CLI:

```bash
fastpki service-account token create 1 --name ci-token
```

## List, rotate, and revoke tokens

Listing returns metadata only — never the plaintext or the digest.

```bash
fastpki service-account token list 1
fastpki service-account token revoke 1 <token_id>
```

To **rotate**, mint a new token, roll it out, then revoke the old one.

## Disable or delete

Disabling keeps the account and its history but rejects all of its tokens at
authentication time. Deleting cascades — all of the account's tokens are
removed.

```bash
fastpki service-account update 1 --disable
fastpki service-account update 1 --enable
fastpki service-account delete 1
```

## Authorization model

A service-account token resolves to a principal scoped to its organization.
Authorization then follows the same rules as a user (see
[Authorization](../security/authorization.md)):

- it can read and act on resources in its own organization subject to its
  capability flags;
- it has full access to resources **it created** (tracked via
  `created_by_service_account_id`);
- it can never reach resources in another organization or unowned resources.

!!! tip "Pre-organization CAs"
    On instances that predate organizations, existing CAs have no
    organization and are therefore out of reach for service accounts. A
    superuser can adopt them into an organization — see
    [Adopting Pre-Organization Resources](organizations.md#adopting-pre-organization-resources).

!!! info "Issuance policies (next phase)"
    Capability flags are coarse — `can_create_cert` lets an account issue any
    certificate its organization allows. Fine-grained, deny-by-default
    constraints (allowed CNs, SANs, CAs, validity) are introduced as
    **issuance policies** in a later phase.

## Auditing

Every management action on a service account is recorded in the
[audit log](audit-logs.md): `service_account_create`, `service_account_update`,
`service_account_delete`, `service_account_token_create`, and
`service_account_token_revoke`.
