# Issuance Policies

An **issuance policy** is a deny-by-default allowlist attached 1:1 to a
[service account](service-accounts.md). It bounds what that account may issue,
so a compromised service-account token has a limited blast radius.

!!! warning "Deny-by-default"
    A service account **with no policy cannot issue certificates at all** — the
    issuance endpoints return `403 service_account_has_no_policy`. Each
    constraint is also deny-by-default: an empty allowlist (e.g. `cn_patterns:
    []`) rejects everything for that dimension. You must explicitly allow what
    the account needs.

Policies apply **only to service-account principals**. User-bound tokens bypass
policy entirely, preserving existing behavior.

## Policy fields

| Field | Type | Meaning |
|-------|------|---------|
| `cn_patterns` | `list[str]` | Glob patterns; the requested CN must match at least one |
| `san_dns_patterns` | `list[str]` | Glob patterns; **each** DNS SAN must match at least one |
| `san_ip_cidrs` | `list[str]` | CIDRs; **each** IP SAN must fall inside one |
| `san_email_domains` | `list[str]` | **Each** email SAN's domain must be listed (`a@<domain>`) |
| `allowed_ca_ids` | `list[int]` | The issuing CA's id must be listed |
| `allowed_certificate_types` | `list[str]` | `server` / `client` |
| `max_validity_days` | `int` | Caps the requested validity (and the server default) |

Globs use shell-style matching (`*`, `?`). For example `*.svc.example.com`
matches `api.svc.example.com` but not `svc.example.com`.

## Set a policy

Requires an admin user. `PUT` is create-or-replace.

```bash
curl -s -X PUT http://localhost:8000/api/v1/service-accounts/1/policy \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "cn_patterns": ["*.svc.example.com"],
    "san_dns_patterns": ["*.svc.example.com"],
    "san_ip_cidrs": ["10.0.0.0/8"],
    "san_email_domains": [],
    "allowed_ca_ids": [1],
    "allowed_certificate_types": ["server"],
    "max_validity_days": 90
  }'
```

CLI (list flags are repeatable):

```bash
fastpki service-account policy set 1 \
  --cn-pattern '*.svc.example.com' \
  --san-dns-pattern '*.svc.example.com' \
  --san-ip-cidr 10.0.0.0/8 \
  --allowed-ca-id 1 \
  --cert-type server \
  --max-validity-days 90
```

## Read and clear

```bash
fastpki service-account policy show 1
fastpki service-account policy clear 1   # reverts to deny-all
```

## Enforcement

When a service account calls `POST /certificates/` or
`POST /certificates/sign-csr`, the policy is evaluated against the **effective**
issuance parameters (for `sign-csr`, the CN/SANs resolved from the CSR after any
overrides). Constraints are checked independently and the **first** failure
short-circuits with a structured error:

```json
{ "detail": { "code": "policy_violation", "field": "san_dns_patterns", "value": "db.evil.com" } }
```

| Outcome | Status | Body `detail.code` |
|---------|--------|--------------------|
| No policy attached | `403` | `service_account_has_no_policy` |
| A constraint failed | `400` | `policy_violation` (with `field` and `value`) |

Policy changes take effect on the next issuance request. Because
`max_validity_days` also caps the server's default validity, a request that
omits `valid_days` is evaluated against that default — set `max_validity_days`
with the default in mind, or have the client request an explicit duration.
