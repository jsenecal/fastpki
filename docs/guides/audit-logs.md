# Audit Logs

FastPKI maintains an immutable audit log for every security-sensitive action. Audit log entries are append-only — they cannot be modified or deleted through the API.

## Delivery Guarantee

Auditing is **fail-closed**. An audited action and its audit entry are written in the same database transaction, committed once before the response is sent: either both are stored or neither is. If the audit entry cannot be written, the action is rolled back and the client receives an error (HTTP 500), so a retry starts from a clean state. A success response therefore always means the audit entry exists. Audited reads, such as private key exports, also return the key only after their audit entry has been committed.

Failure events (`login_failure`) are committed before the error response is returned, so they are recorded even though the request itself fails.

## Query Audit Logs

```bash
curl -s "http://localhost:8000/api/v1/audit-logs/" \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

**Required role:** Admin or Superuser. Regular users (`USER` role) cannot access audit logs.

- **Superusers** see all audit logs across the system.
- **Admins** see only logs for their own organization.

## Filtering

All filters are optional query parameters and can be combined.

```bash
# Filter by action type
curl -s "http://localhost:8000/api/v1/audit-logs/?action=ca_create" \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool

# Filter by user
curl -s "http://localhost:8000/api/v1/audit-logs/?user_id=1" \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool

# Filter by resource
curl -s "http://localhost:8000/api/v1/audit-logs/?resource_type=certificate&resource_id=5" \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool

# Filter by date range
curl -s "http://localhost:8000/api/v1/audit-logs/?since=2025-01-01T00:00:00&until=2025-02-01T00:00:00" \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool

# Pagination
curl -s "http://localhost:8000/api/v1/audit-logs/?skip=0&limit=50" \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `action` | `AuditAction` | Filter by action type (see table below) |
| `user_id` | `int` | Filter by the user who performed the action |
| `resource_type` | `str` | Filter by resource type (`ca`, `certificate`, `user`, `organization`) |
| `resource_id` | `int` | Filter by resource ID |
| `since` | `datetime` | Only logs created at or after this time |
| `until` | `datetime` | Only logs created at or before this time |
| `skip` | `int` | Offset for pagination (default `0`) |
| `limit` | `int` | Max results, 1–1000 (default `100`) |

## Audit Event Types

| Action | Trigger |
|--------|---------|
| `ca_create` | A Certificate Authority is created |
| `ca_delete` | A Certificate Authority is deleted |
| `ca_export_private_key` | A CA private key is viewed or exported |
| `cert_create` | A certificate is issued |
| `cert_revoke` | A certificate is revoked |
| `cert_export_private_key` | A certificate private key is viewed or exported |
| `login_success` | Successful authentication |
| `login_failure` | Failed authentication attempt |
| `user_create` | A new user is created |
| `user_update` | A user is updated |
| `org_create` | An organization is created |
| `org_delete` | An organization is deleted |
| `org_add_user` | A user is added to an organization |
| `org_remove_user` | A user is removed from an organization |

## Response Format

```json
{
    "id": 42,
    "created_at": "2025-01-30T14:22:00",
    "action": "cert_create",
    "user_id": 1,
    "username": "admin",
    "organization_id": 1,
    "resource_type": "certificate",
    "resource_id": 7,
    "detail": "Created certificate 'web.example.com'"
}
```
