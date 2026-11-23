# First Steps

This walkthrough takes you from a fresh install to a signed certificate. All examples use `curl` against `http://localhost:8000`.

## 1. Create the First Superuser

The very first user created in the system can be assigned any role — this is the bootstrap mechanism.

```bash
curl -s -X POST http://localhost:8000/api/v1/users/ \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "email": "admin@example.com",
    "password": "tangerine-orbit-velvet-42",
    "role": "superuser"
  }' | python -m json.tool
```

??? example "Response"

    ```json
    {
        "id": 1,
        "username": "admin",
        "email": "admin@example.com",
        "role": "superuser",
        "is_active": true,
        "organization_id": null,
        "can_create_ca": false,
        "can_create_cert": false,
        "can_revoke_cert": false,
        "can_export_private_key": false,
        "can_delete_ca": false,
        "created_at": "2025-01-30T12:00:00",
        "updated_at": "2025-01-30T12:00:00"
    }
    ```

## 2. Log In and Get a Token

FastPKI uses OAuth2 password flow. The token endpoint expects form-encoded data.

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=tangerine-orbit-velvet-42" | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo $TOKEN
```

From here on every authenticated request sends the token as a Bearer header:

```
-H "Authorization: Bearer $TOKEN"
```

## 3. Create a Certificate Authority

```bash
curl -s -X POST http://localhost:8000/api/v1/cas/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Root CA",
    "subject_dn": "CN=My Root CA,O=Example Inc,C=US",
    "description": "Primary root CA"
  }' | python -m json.tool
```

??? example "Response"

    ```json
    {
        "id": 1,
        "name": "My Root CA",
        "description": "Primary root CA",
        "subject_dn": "CN=My Root CA,O=Example Inc,C=US",
        "key_size": 4096,
        "valid_days": 3650,
        "created_at": "2025-01-30T12:01:00",
        "updated_at": "2025-01-30T12:01:00",
        "certificate": "-----BEGIN CERTIFICATE-----\n...",
        "private_key": "-----BEGIN RSA PRIVATE KEY-----\n...",
        "organization_id": null,
        "created_by_user_id": 1
    }
    ```

The response includes the private key because the `POST` endpoint returns a `CADetailResponse`. Subsequent `GET /cas/{id}` calls return only the certificate (without the private key) unless you explicitly request the private key endpoint.

## 4. Issue a Certificate

```bash
curl -s -X POST "http://localhost:8000/api/v1/certificates/?ca_id=1" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "common_name": "web.example.com",
    "subject_dn": "CN=web.example.com,O=Example Inc,C=US",
    "certificate_type": "server",
    "include_private_key": true
  }' | python -m json.tool
```

??? example "Response"

    ```json
    {
        "id": 1,
        "common_name": "web.example.com",
        "subject_dn": "CN=web.example.com,O=Example Inc,C=US",
        "certificate_type": "server",
        "key_size": 2048,
        "valid_days": 365,
        "status": "valid",
        "certificate": "-----BEGIN CERTIFICATE-----\n...",
        "private_key": "-----BEGIN RSA PRIVATE KEY-----\n...",
        "serial_number": "ABC123...",
        "not_before": "2025-01-30T12:02:00",
        "not_after": "2026-01-30T12:02:00",
        "revoked_at": null,
        "created_at": "2025-01-30T12:02:00",
        "updated_at": "2025-01-30T12:02:00",
        "issuer_id": 1,
        "organization_id": null,
        "created_by_user_id": 1
    }
    ```

## 5. Export the Certificate as a PEM File

```bash
curl -s -OJ http://localhost:8000/api/v1/export/certificate/1 \
  -H "Authorization: Bearer $TOKEN"
```

This downloads the certificate as `certificate_1.pem`. You can also export the private key and the full chain — see [Exporting](../guides/exporting.md).

## Next Steps

- [Certificate Authorities guide](../guides/certificate-authorities.md) — managing CAs in depth
- [CRL & Public PKI](../guides/crl.md) — CRL generation and public download endpoints
- [Organizations guide](../guides/organizations.md) — setting up multi-tenant access
- [CLI Tool](../guides/cli.md) — manage PKI from the command line instead of curl
- [Security overview](../security/authentication.md) — understanding tokens, roles, and permissions
