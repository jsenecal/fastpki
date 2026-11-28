import pytest

from app.core.config import settings
from app.db.models import AuditAction
from app.services.audit import AuditService


@pytest.mark.asyncio
async def test_unauthenticated_returns_401(client):
    response = await client.get(f"{settings.API_V1_STR}/audit-logs/")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_normal_user_returns_403(normal_user_client):
    response = await normal_user_client.get(f"{settings.API_V1_STR}/audit-logs/")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_admin_can_access(admin_client, db):
    audit = AuditService(db)
    await audit.log_action(action=AuditAction.CA_CREATE, organization_id=1)

    response = await admin_client.get(f"{settings.API_V1_STR}/audit-logs/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_superuser_can_access(superuser_client, db):
    audit = AuditService(db)
    await audit.log_action(action=AuditAction.CA_CREATE)

    response = await superuser_client.get(f"{settings.API_V1_STR}/audit-logs/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_admin_only_sees_own_org_logs(admin_client, admin_user, db):
    audit = AuditService(db)
    await audit.log_action(
        action=AuditAction.CA_CREATE,
        organization_id=admin_user.organization_id,
    )
    await audit.log_action(
        action=AuditAction.CA_CREATE,
        organization_id=9999,
    )

    response = await admin_client.get(f"{settings.API_V1_STR}/audit-logs/")
    assert response.status_code == 200
    logs = response.json()
    assert all(log["organization_id"] == admin_user.organization_id for log in logs)
    assert len(logs) == 1


@pytest.mark.asyncio
async def test_superuser_sees_all_logs(superuser_client, db):
    audit = AuditService(db)
    await audit.log_action(action=AuditAction.CA_CREATE, organization_id=1)
    await audit.log_action(action=AuditAction.CERT_CREATE, organization_id=2)
    await audit.log_action(action=AuditAction.LOGIN_SUCCESS, organization_id=None)

    response = await superuser_client.get(f"{settings.API_V1_STR}/audit-logs/")
    assert response.status_code == 200
    logs = response.json()
    assert len(logs) == 3


@pytest.mark.asyncio
async def test_filter_by_action(superuser_client, db):
    audit = AuditService(db)
    await audit.log_action(action=AuditAction.CA_CREATE)
    await audit.log_action(action=AuditAction.CERT_CREATE)

    response = await superuser_client.get(
        f"{settings.API_V1_STR}/audit-logs/",
        params={"action": "ca_create"},
    )
    assert response.status_code == 200
    logs = response.json()
    assert len(logs) == 1
    assert logs[0]["action"] == "ca_create"


@pytest.mark.asyncio
async def test_filter_by_resource_type(superuser_client, db):
    audit = AuditService(db)
    await audit.log_action(action=AuditAction.CA_CREATE, resource_type="ca")
    await audit.log_action(action=AuditAction.CERT_CREATE, resource_type="certificate")

    response = await superuser_client.get(
        f"{settings.API_V1_STR}/audit-logs/",
        params={"resource_type": "ca"},
    )
    assert response.status_code == 200
    logs = response.json()
    assert len(logs) == 1
    assert logs[0]["resource_type"] == "ca"


@pytest.mark.asyncio
async def test_pagination(superuser_client, db):
    audit = AuditService(db)
    for _ in range(5):
        await audit.log_action(action=AuditAction.CA_CREATE)

    response = await superuser_client.get(
        f"{settings.API_V1_STR}/audit-logs/",
        params={"skip": 0, "limit": 2},
    )
    assert response.status_code == 200
    logs = response.json()
    assert len(logs) == 2


@pytest.mark.asyncio
async def test_ca_create_generates_audit_log(superuser_client, db):
    ca_data = {
        "name": "Audit Test CA",
        "subject_dn": "CN=Audit Test CA,O=Test,C=US",
        "key_size": 2048,
        "valid_days": 365,
    }
    response = await superuser_client.post(f"{settings.API_V1_STR}/cas/", json=ca_data)
    assert response.status_code == 201

    audit = AuditService(db)
    logs = await audit.list_audit_logs(action=AuditAction.CA_CREATE)
    assert len(logs) >= 1
    assert logs[0].resource_type == "ca"


@pytest.mark.asyncio
async def test_ca_delete_generates_audit_log(superuser_client, db):
    ca_data = {
        "name": "CA to Delete",
        "subject_dn": "CN=CA to Delete,O=Test,C=US",
    }
    resp = await superuser_client.post(f"{settings.API_V1_STR}/cas/", json=ca_data)
    ca_id = resp.json()["id"]

    response = await superuser_client.delete(f"{settings.API_V1_STR}/cas/{ca_id}")
    assert response.status_code == 204

    audit = AuditService(db)
    logs = await audit.list_audit_logs(action=AuditAction.CA_DELETE)
    assert len(logs) >= 1
    assert logs[0].resource_id == ca_id


@pytest.mark.asyncio
async def test_cert_create_generates_audit_log(superuser_client, db):
    ca_data = {
        "name": "Cert Audit CA",
        "subject_dn": "CN=Cert Audit CA,O=Test,C=US",
    }
    ca_resp = await superuser_client.post(f"{settings.API_V1_STR}/cas/", json=ca_data)
    ca_id = ca_resp.json()["id"]

    cert_data = {
        "common_name": "test.example.com",
        "subject_dn": "CN=test.example.com,O=Test,C=US",
        "certificate_type": "server",
        "key_size": 2048,
        "valid_days": 365,
    }
    response = await superuser_client.post(
        f"{settings.API_V1_STR}/certificates/",
        json=cert_data,
        params={"ca_id": ca_id},
    )
    assert response.status_code == 201

    audit = AuditService(db)
    logs = await audit.list_audit_logs(action=AuditAction.CERT_CREATE)
    assert len(logs) >= 1
    assert logs[0].resource_type == "certificate"


@pytest.mark.asyncio
async def test_cert_revoke_generates_audit_log(superuser_client, db):
    ca_data = {
        "name": "Revoke Audit CA",
        "subject_dn": "CN=Revoke Audit CA,O=Test,C=US",
    }
    ca_resp = await superuser_client.post(f"{settings.API_V1_STR}/cas/", json=ca_data)
    ca_id = ca_resp.json()["id"]

    cert_data = {
        "common_name": "revoke.example.com",
        "subject_dn": "CN=revoke.example.com,O=Test,C=US",
        "certificate_type": "server",
        "key_size": 2048,
        "valid_days": 365,
    }
    cert_resp = await superuser_client.post(
        f"{settings.API_V1_STR}/certificates/",
        json=cert_data,
        params={"ca_id": ca_id},
    )
    cert_id = cert_resp.json()["id"]

    response = await superuser_client.post(
        f"{settings.API_V1_STR}/certificates/{cert_id}/revoke",
        json={"reason": "testing"},
    )
    assert response.status_code == 200

    audit = AuditService(db)
    logs = await audit.list_audit_logs(action=AuditAction.CERT_REVOKE)
    assert len(logs) >= 1
    assert logs[0].resource_id == cert_id


@pytest.mark.asyncio
async def test_login_success_generates_audit_log(client, db):
    user_data = {
        "username": "audit_login_user",
        "email": "auditlogin@example.com",
        "password": "Kestrel-Meadow-4821",
    }
    await client.post(f"{settings.API_V1_STR}/users/", json=user_data)

    response = await client.post(
        f"{settings.API_V1_STR}/auth/token",
        data={"username": "audit_login_user", "password": "Kestrel-Meadow-4821"},
    )
    assert response.status_code == 200

    audit = AuditService(db)
    logs = await audit.list_audit_logs(action=AuditAction.LOGIN_SUCCESS)
    assert len(logs) >= 1
    assert logs[0].username == "audit_login_user"


@pytest.mark.asyncio
async def test_login_failure_generates_audit_log(client, db):
    response = await client.post(
        f"{settings.API_V1_STR}/auth/token",
        data={"username": "nonexistent_audit_user", "password": "wrongpass"},
    )
    assert response.status_code == 401

    audit = AuditService(db)
    logs = await audit.list_audit_logs(action=AuditAction.LOGIN_FAILURE)
    assert len(logs) >= 1
    assert logs[0].username == "nonexistent_audit_user"
    assert logs[0].user_id is None
