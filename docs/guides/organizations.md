# Organizations

Organizations provide multi-tenant isolation. CAs and certificates created by a user are scoped to that user's organization. Users within the same organization can see each other's resources according to the [permission hierarchy](../security/authorization.md).

## Create an Organization

```bash
curl -s -X POST http://localhost:8000/api/v1/organizations/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Engineering",
    "description": "Engineering department"
  }' | python -m json.tool
```

**Required role:** Admin or Superuser.

??? example "Response"

    ```json
    {
        "id": 1,
        "name": "Engineering",
        "description": "Engineering department",
        "created_at": "2025-01-30T12:00:00",
        "updated_at": "2025-01-30T12:00:00"
    }
    ```

## List Organizations

```bash
curl -s http://localhost:8000/api/v1/organizations/ \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

- **Superusers** see all organizations.
- **Other users** see only their own organization (or an empty list if unassigned).

## View an Organization

```bash
curl -s http://localhost:8000/api/v1/organizations/1 \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

Users can only view organizations they belong to. Superusers can view any organization.

## Update an Organization

```bash
curl -s -X PUT http://localhost:8000/api/v1/organizations/1 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Engineering Team",
    "description": "Updated description"
  }' | python -m json.tool
```

**Required role:** Superuser, or Admin in the same organization.

## Delete an Organization

```bash
curl -s -X DELETE http://localhost:8000/api/v1/organizations/1 \
  -H "Authorization: Bearer $TOKEN"
```

Returns `204 No Content`. Organizations that still have users cannot be deleted — remove all users first.

**Required role:** Superuser only. Audit-logged.

## Add a User to an Organization

```bash
curl -s -X POST http://localhost:8000/api/v1/organizations/1/users/2 \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

**Required role:** Superuser (any org), or Admin in the target organization.

## Remove a User from an Organization

```bash
curl -s -X DELETE http://localhost:8000/api/v1/organizations/1/users/2 \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

**Required role:** Superuser (any org), or Admin in the target organization.

## Adopting Pre-Organization Resources

Instances that predate organizations (or resources created by a superuser with
no organization) have CAs and certificates with `organization_id: null`. Such
resources are accessible to superusers only, so organization-scoped users and
[service accounts](service-accounts.md) cannot use them — for example, a
service account with an issuance policy cannot mint certificates under an
org-less CA.

To adopt an existing CA into an organization, assign it with
`PATCH /cas/{ca_id}`:

```bash
curl -s -X PATCH http://localhost:8000/api/v1/cas/2 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "organization_id": 1,
    "cascade": true
  }' | python -m json.tool
```

Or with the CLI:

```bash
fastpki ca assign-org 2 --org 1 --cascade
```

With `"cascade": true`, all descendant CAs and the certificates issued by the
affected CAs are adopted into the organization as well — use this to migrate
an entire pre-organization hierarchy in one call. Without it, only the
specified CA is reassigned; previously issued certificates keep their current
organization (and org-less ones remain superuser-only).

**Required role:** Superuser only. Audit-logged.

## List Users in an Organization

```bash
curl -s http://localhost:8000/api/v1/organizations/1/users \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

Members of the organization can list its users. Superusers can list users in any organization.
