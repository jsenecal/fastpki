# Users

## Create a User

### First User (Bootstrap)

The very first user in the system can be created without authentication and assigned any role, including `superuser`. This is the bootstrap mechanism.

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

### Subsequent Users

After the first user exists, creating users with `admin` or `superuser` roles requires authentication as a superuser. Regular users can be created by anyone.

```bash
curl -s -X POST http://localhost:8000/api/v1/users/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "alice",
    "email": "alice@example.com",
    "password": "sunflower-harbor-velvet-7",
    "role": "user",
    "organization_id": 1,
    "can_create_cert": true,
    "can_revoke_cert": true
  }' | python -m json.tool
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `username` | Yes | — | Unique username |
| `email` | Yes | — | Unique email address |
| `password` | Yes | — | Must satisfy the [password policy](../security/authentication.md#password-policy) (12–72 bytes, not common, not containing the username/email) |
| `role` | No | `user` | `superuser`, `admin`, or `user` |
| `organization_id` | No | `null` | Assign to an organization on creation |
| `can_create_ca` | No | `false` | Allow creating CAs |
| `can_create_cert` | No | `false` | Allow creating certificates |
| `can_revoke_cert` | No | `false` | Allow revoking certificates |
| `can_export_private_key` | No | `false` | Allow exporting private keys |
| `can_delete_ca` | No | `false` | Allow deleting CAs |

## List Users

```bash
curl -s http://localhost:8000/api/v1/users/ \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

**Required role:** Superuser only.

Supports `skip` and `limit` query parameters for pagination.

## Get Current User

```bash
curl -s http://localhost:8000/api/v1/users/me \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

## Get a User by ID

```bash
curl -s http://localhost:8000/api/v1/users/2 \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

Regular users can only view their own profile. Admins and superusers can view any user.

## Update a User

```bash
curl -s -X PATCH http://localhost:8000/api/v1/users/2 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "can_create_cert": true,
    "can_export_private_key": true
  }' | python -m json.tool
```

| Field | Description | Who can change it |
|-------|-------------|-------------------|
| `email` | New email address | Self, Admin (same org), Superuser |
| `password` | New password; must satisfy the [password policy](../security/authentication.md#password-policy) | Self, Admin (same org), Superuser |
| `role` | Change role | Superuser only |
| `is_active` | Activate / deactivate | Superuser only |
| `organization_id` | Change organization | Superuser only |
| `can_create_ca` | Capability flag | Admin (same org) or Superuser |
| `can_create_cert` | Capability flag | Admin (same org) or Superuser |
| `can_revoke_cert` | Capability flag | Admin (same org) or Superuser |
| `can_export_private_key` | Capability flag | Admin (same org) or Superuser |
| `can_delete_ca` | Capability flag | Admin (same org) or Superuser |

## Delete a User

```bash
curl -s -X DELETE http://localhost:8000/api/v1/users/2 \
  -H "Authorization: Bearer $TOKEN"
```

Returns `204 No Content`. Superusers cannot delete themselves.

**Required role:** Superuser only.
