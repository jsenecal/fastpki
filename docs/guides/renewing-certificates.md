# Renewing Certificates

`POST /api/v1/certificates/{id}/renew` issues a **new** certificate based on an
existing one. The subject, SANs, issuing CA, certificate type, and the original
validity *duration* are inherited; the new certificate starts now. The
predecessor is **not** revoked — revocation remains a separate, explicit action.

## Two modes

The mode is determined by how the original certificate was issued:

| Original issued via | Renew with | Result |
|---------------------|-----------|--------|
| Server-generated key (`POST /certificates/`) | **empty body** `{}` | Server mints a fresh key pair and returns the new cert **and** its private key |
| A CSR (`POST /certificates/sign-csr`) | `{ "csr": "<PEM>" }` | Server uses the new CSR's public key; the CSR's **subject is ignored** (inherited from the predecessor) |

Mismatches are rejected with `400` and a structured `detail.code`:

| Situation | `detail.code` |
|-----------|---------------|
| Empty body for a CSR-origin certificate | `csr_required_for_csr_origin_cert` |
| CSR supplied for a server-key certificate | `csr_not_allowed_for_server_key_cert` |

## Examples

Server-key certificate (returns a new private key):

```bash
curl -s -X POST http://localhost:8000/api/v1/certificates/42/renew \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{}'
```

CSR-origin certificate:

```bash
curl -s -X POST http://localhost:8000/api/v1/certificates/42/renew \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"csr\": \"$(awk '{printf "%s\\n", $0}' new.csr)\"}"
```

CLI:

```bash
fastpki cert renew 42              # server-key
fastpki cert renew 42 --csr new.csr   # CSR-origin
```

## Lineage

The new certificate records `renewed_from_id`. Reading the predecessor exposes
the reverse list:

```bash
fastpki cert show 42    # renewed_to_ids: [57]
```

`GET /api/v1/certificates/{id}` returns both `renewed_from_id` and
`renewed_to_ids`. A certificate is effectively *superseded* once it has entries
in `renewed_to_ids`, but it stays valid (and usable) until it is revoked or
expires.

## Policy re-evaluation

When the caller is a [service account](service-accounts.md), the **current**
[issuance policy](issuance-policies.md) is applied to the inherited parameters.
If the policy has tightened so the original parameters no longer comply, renewal
is rejected with the same structured `policy_violation` error as first issuance.
This means a policy change retroactively governs renewals.
