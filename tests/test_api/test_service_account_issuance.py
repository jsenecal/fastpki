import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_principal
from app.db.models import AuditAction, CertificateType
from app.db.session import get_session
from app.services.audit import AuditService
from app.services.ca import CAService
from app.services.cert import CertificateService
from app.services.issuance_policy import IssuancePolicyService
from app.services.organization import OrganizationService
from app.services.principal import Principal
from app.services.service_account import ServiceAccountService
from tests.conftest import create_test_app, get_test_session


def _app_for_principal(principal: Principal) -> FastAPI:
    app = create_test_app()
    app.dependency_overrides[get_session] = get_test_session
    app.dependency_overrides[get_current_principal] = lambda: principal
    return app


@pytest_asyncio.fixture
async def org(db):
    return await OrganizationService(db).create_organization(name="IssueOrg")


@pytest_asyncio.fixture
async def ca(db, org):
    return await CAService(db).create_ca(
        name="Issue CA", subject_dn="CN=Issue CA", organization_id=org.id
    )


async def _sa(db, org, **flags):
    return await ServiceAccountService(db).create_service_account(
        name=flags.pop("name", "issuer"), organization_id=org.id, **flags
    )


@pytest.mark.asyncio
async def test_service_account_can_create_certificate(db, org, ca):
    sa = await _sa(db, org, can_create_cert=True)
    # A service account needs a policy to issue (deny-by-default, see #29).
    await IssuancePolicyService(db).set_policy(
        sa.id,
        cn_patterns=["*.example.com"],
        san_dns_patterns=[],
        san_ip_cidrs=[],
        san_email_domains=[],
        allowed_ca_ids=[ca.id],
        allowed_certificate_types=[CertificateType.SERVER],
        max_validity_days=3650,
    )
    app = _app_for_principal(Principal.from_service_account(sa))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            f"/api/v1/certificates/?ca_id={ca.id}",
            json={
                "common_name": "svc.example.com",
                "subject_dn": "CN=svc.example.com",
                "certificate_type": CertificateType.SERVER.value,
            },
        )
    assert resp.status_code == 201, resp.text

    # The new cert must be owned by the service account, not a user.
    certs = await CertificateService(db).list_certificates(organization_id=org.id)
    assert len(certs) == 1
    assert certs[0].created_by_service_account_id == sa.id
    assert certs[0].created_by_user_id is None

    # Audit records the service account as the actor.
    logs = await AuditService(db).list_audit_logs(action=AuditAction.CERT_CREATE)
    assert len(logs) == 1
    assert logs[0].service_account_id == sa.id
    assert logs[0].service_account_name == sa.name
    assert logs[0].user_id is None


@pytest.mark.asyncio
async def test_service_account_without_capability_denied(db, org, ca):
    sa = await _sa(db, org, name="readonly", can_create_cert=False)
    app = _app_for_principal(Principal.from_service_account(sa))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            f"/api/v1/certificates/?ca_id={ca.id}",
            json={
                "common_name": "svc.example.com",
                "subject_dn": "CN=svc.example.com",
                "certificate_type": CertificateType.SERVER.value,
            },
        )
    assert resp.status_code == 403, resp.text
