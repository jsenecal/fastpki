# Authorization

FastPKI combines role-based access control (RBAC) with per-user capability flags to provide flexible, fine-grained permissions.

## Roles

| Role | Scope | Description |
|------|-------|-------------|
| **Superuser** | Global | Full access to all resources across all organizations |
| **Admin** | Organization | Full access to all resources within their organization |
| **User** | Organization | Read access within their organization; write actions require capability flags |

## Permission Hierarchy

When a user performs an action on a resource (CA or certificate), the permission check follows this order:

1. **Superuser** — always allowed
2. **Unowned resource** (no `organization_id`) — only superusers can access
3. **Different organization** — denied
4. **Admin in same organization** — full access
5. **User in same organization, read action** — allowed
6. **User in same organization, write action** — check capability flags

!!! note "Creating a resource grants no extra rights"
    `created_by_user_id` / `created_by_service_account_id` record who created a
    CA or certificate for attribution and auditing only. They are **not**
    consulted during authorization. Access is always evaluated against the
    caller's *current* role, organization, and capability flags, so demoting a
    user, revoking a capability flag, or moving a user or service account to
    another organization immediately removes the corresponding access to
    resources they created earlier — including private key export.

## Capability Flags

Regular users can be granted specific write permissions through boolean flags on their user profile. These only apply to users with the `user` role — superusers and admins already have the corresponding access.

| Flag | Grants |
|------|--------|
| `can_create_ca` | Create Certificate Authorities |
| `can_create_cert` | Issue certificates |
| `can_revoke_cert` | Revoke certificates |
| `can_export_private_key` | View / export private keys |
| `can_delete_ca` | Delete Certificate Authorities |

### Setting Capability Flags

Superusers and admins (within the same organization) can set capability flags when creating or updating a user:

```bash
# Grant cert creation and revocation to user 3
curl -s -X PATCH http://localhost:8000/api/v1/users/3 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "can_create_cert": true,
    "can_revoke_cert": true
  }' | python -m json.tool
```

## Permission Matrix

The table below shows which roles and capabilities allow each action.

| Action | Superuser | Admin (same org) | User (same org) | Required flag |
|--------|:---------:|:-----------------:|:----------------:|:-------------:|
| List CAs | All | Org-scoped | Org-scoped | — |
| View CA | Yes | Yes | Yes | — |
| Create CA | Yes | Yes | Flag | `can_create_ca` |
| Delete CA | Yes | Yes | Flag | `can_delete_ca` |
| Export CA private key | Yes | Yes | Flag | `can_export_private_key` |
| List certificates | All | Org-scoped | Org-scoped | — |
| View certificate | Yes | Yes | Yes | — |
| Create certificate | Yes | Yes | Flag | `can_create_cert` |
| Revoke certificate | Yes | Yes | Flag | `can_revoke_cert` |
| Export cert private key | Yes | Yes | Flag | `can_export_private_key` |
| Create users | Any role, any org | `user` role, own org | — | — |
| Update users | Yes | Non-superusers, own org | Self only | — |
| Delete users | Yes | — | — | — |
| Create organizations | Yes | Yes | — | — |
| Delete organizations | Yes | — | — | — |
| Manage org membership | Yes | Yes | — | — |
| View audit logs | Yes | Org-scoped | — | — |

## Endpoint-Level Auth Requirements

| Dependency | Meaning |
|------------|---------|
| `get_current_active_user` | Any authenticated, active user |
| `get_current_active_admin_user` | Admin or Superuser |
| `get_current_active_superuser` | Superuser only |
