# Certificates

Certificates are issued by a Certificate Authority and can be of type `server`, `client`, or `ca`.

## Issue a Certificate

Certificates are always created under a specific CA, identified by the `ca_id` query parameter.

```bash
curl -s -X POST "http://localhost:8000/api/v1/certificates/?ca_id=1" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "common_name": "api.example.com",
    "subject_dn": "CN=api.example.com,O=Acme Corp,C=US",
    "certificate_type": "server",
    "key_size": 2048,
    "valid_days": 365,
    "include_private_key": true
  }' | python -m json.tool
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `common_name` | Yes | — | Common Name for the certificate |
| `subject_dn` | Yes | — | Full distinguished name |
| `certificate_type` | Yes | — | `server`, `client`, or `ca` |
| `key_size` | No | `CERT_KEY_SIZE` (2048) | RSA key size in bits |
| `valid_days` | No | `CERT_DAYS` (365) | Validity period in days |
| `include_private_key` | No | `true` | Whether to generate and store a private key |
| `san_dns_names` | No | — | DNS Subject Alternative Names (server/dual-purpose) |
| `san_ip_addresses` | No | — | IP Subject Alternative Names (server/dual-purpose) |
| `san_email_addresses` | No | — | Email Subject Alternative Names (client/dual-purpose) |

**Required permission:** `create_cert` capability on the target CA, Admin role in the same org, or Superuser.

### Subject Alternative Names

Modern TLS clients ignore the Common Name and rely on SANs. FastPKI auto-populates sensible defaults:

- **Server / dual-purpose** certificates get a DNS SAN equal to the Common Name when **no SANs of any kind** are provided (no `san_dns_names`, `san_ip_addresses`, or `san_email_addresses`). The Common Name is only used if it is a valid DNS name (letters, digits, `-`, `_`, optional leading `*.` wildcard); a CN such as `My Web Server` or an IP literal such as `10.0.0.1` is never written into a DNS SAN.
- **Client** certificates whose Common Name parses as an email get an email SAN when no `san_email_addresses` are provided.

!!! note "Explicit SANs are taken verbatim"
    If you supply **any** SAN list, it is used exactly as given and the Common
    Name is **not** added on top. For example, a server certificate requested
    with only `san_ip_addresses: ["10.0.0.42"]` carries an IP SAN and no DNS
    SAN. If you also want the Common Name as a DNS SAN, include it in
    `san_dns_names`. (Before this change, the Common Name was added as a DNS SAN
    whenever `san_dns_names` was omitted, even if IP SANs were given.)

The same rules apply to `sign-csr` (after CSR values and overrides are merged)
and to renewal. [Issuance policies](issuance-policies.md) are evaluated against
these *effective* SANs — exactly what ends up in the certificate.

SAN types are constrained by certificate type: servers reject email SANs; clients reject DNS and IP SANs. Violations return `400`.

```bash
curl -s -X POST "http://localhost:8000/api/v1/certificates/?ca_id=1" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "common_name": "api.example.com",
    "subject_dn": "CN=api.example.com,O=Acme Corp,C=US",
    "certificate_type": "server",
    "san_dns_names": ["api.example.com", "api-v2.example.com"],
    "san_ip_addresses": ["10.0.0.42"]
  }' | python -m json.tool
```

### Client Certificate Example

```bash
curl -s -X POST "http://localhost:8000/api/v1/certificates/?ca_id=1" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "common_name": "alice",
    "subject_dn": "CN=alice,O=Acme Corp,C=US",
    "certificate_type": "client"
  }' | python -m json.tool
```

## Sign a Certificate Signing Request

When the private key must stay on the requesting host — for example, a cert-manager controller, a load balancer provisioning its own keys, or a compliance workflow that forbids the CA from ever seeing the key — submit a CSR instead of calling the create endpoint.

```bash
# Generate a key and CSR locally
openssl req -new -newkey rsa:2048 -nodes \
  -keyout api.example.com.key \
  -out api.example.com.csr \
  -subj "/CN=api.example.com/O=Acme Corp/C=US"

# Submit the CSR for signing
curl -s -X POST http://localhost:8000/api/v1/certificates/sign-csr \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "$(jq -Rs --arg type server --arg ca "Acme Issuing CA" \
        '{csr: ., certificate_type: $type, ca_name: $ca}' \
        < api.example.com.csr)" | python -m json.tool
```

The CA is selected by `ca_id` or `ca_name` (one required). `ca_name` is scoped to the caller's organization; superusers may resolve any CA by name. Subject, SANs, and the public key are extracted from the CSR; pass explicit `common_name`, `subject_dn`, or `san_*` fields to override.

The response contains only the signed certificate — never a private key, since the requester already holds it.

**Required permission:** `create_cert` capability on the resolved CA, Admin in the same org, or Superuser. Audit-logged.

## List Certificates

```bash
# All certificates visible to the current user
curl -s http://localhost:8000/api/v1/certificates/ \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool

# Filter by issuing CA
curl -s "http://localhost:8000/api/v1/certificates/?ca_id=1" \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

Superusers see all certificates. Other users see only certificates belonging to their organization.

## View a Certificate

```bash
curl -s http://localhost:8000/api/v1/certificates/1 \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

## View a Certificate with Private Key

```bash
curl -s http://localhost:8000/api/v1/certificates/1/private-key \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

**Required permission:** `export_private_key` capability, Admin in the same org, or Superuser. Audit-logged.

## Revoke a Certificate

```bash
curl -s -X POST http://localhost:8000/api/v1/certificates/1/revoke \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "reason": "Key compromise"
  }' | python -m json.tool
```

| Field | Required | Description |
|-------|----------|-------------|
| `reason` | No | Human-readable revocation reason |

The certificate status changes to `revoked` and a CRL entry is created. Revoking an already-revoked certificate returns `409 Conflict`.

**Required permission:** `revoke_cert` capability, Admin in the same org, or Superuser. Audit-logged.
