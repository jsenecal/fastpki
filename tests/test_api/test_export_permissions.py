import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_current_active_admin_user,
    get_current_active_superuser,
    get_current_active_user,
    get_current_principal,
    get_current_user,
)
from app.core.config import settings
from app.db.models import CertificateType, User, UserRole
from app.db.session import get_session
from app.services.ca import CAService
from app.services.cert import CertificateService
from app.services.organization import OrganizationService
from app.services.user import UserService
from tests.conftest import (
    TestAuth,
    TestPrincipalAuth,
    create_test_app,
    get_test_session,
)


def _app_for(user: User) -> FastAPI:
    app = create_test_app()
    app.dependency_overrides[get_session] = get_test_session
    app.dependency_overrides[get_current_user] = TestAuth(user)
    app.dependency_overrides[get_current_active_user] = TestAuth(user)
    app.dependency_overrides[get_current_active_superuser] = TestAuth(user)
    app.dependency_overrides[get_current_active_admin_user] = TestAuth(user)
    app.dependency_overrides[get_current_principal] = TestPrincipalAuth(user)
    return app


@pytest_asyncio.fixture
async def other_org(db: AsyncSession):
    org_service = OrganizationService(db)
    return await org_service.create_organization(
        name="OtherExpOrg", description="Other"
    )


@pytest_asyncio.fixture
async def user_other_org(db: AsyncSession, other_org):
    user_service = UserService(db)
    return await user_service.create_user(
        username="otherorgexp",
        email="otherorgexp@example.com",
        password="password123",
        role=UserRole.USER,
        organization_id=other_org.id,
    )


@pytest_asyncio.fixture
async def user_with_export(db: AsyncSession, test_org):
    user_service = UserService(db)
    return await user_service.create_user(
        username="canexport",
        email="canexport@example.com",
        password="password123",
        role=UserRole.USER,
        organization_id=test_org.id,
        can_export_private_key=True,
    )


@pytest_asyncio.fixture
async def org_ca(db: AsyncSession, test_org, superuser):
    ca_service = CAService(db)
    return await ca_service.create_ca(
        name="ExportPermCA",
        subject_dn="CN=ExportPermCA",
        organization_id=test_org.id,
        created_by_user_id=superuser.id,
    )


@pytest_asyncio.fixture
async def org_cert(db: AsyncSession, org_ca, test_org, admin_user):
    cert_service = CertificateService(db)
    return await cert_service.create_certificate(
        ca_id=org_ca.id,
        common_name="export.example.com",
        subject_dn="CN=export.example.com",
        certificate_type=CertificateType.SERVER,
        include_private_key=True,
        organization_id=test_org.id,
        created_by_user_id=admin_user.id,
    )


@pytest.mark.asyncio
async def test_user_in_org_can_export_ca_cert(setup_db, normal_user, org_ca):
    app = _app_for(normal_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/export/ca/{org_ca.id}/certificate")
        assert resp.status_code == 200
        assert "BEGIN CERTIFICATE" in resp.text


@pytest.mark.asyncio
async def test_other_org_cannot_export_ca_cert(setup_db, user_other_org, org_ca):
    app = _app_for(user_other_org)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/export/ca/{org_ca.id}/certificate")
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_user_without_cap_cannot_export_ca_private_key(
    setup_db, normal_user, org_ca
):
    app = _app_for(normal_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/export/ca/{org_ca.id}/private-key")
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_user_with_export_cap_can_export_ca_private_key(
    setup_db, user_with_export, org_ca
):
    app = _app_for(user_with_export)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/export/ca/{org_ca.id}/private-key")
        assert resp.status_code == 200
        assert "BEGIN PRIVATE KEY" in resp.text


@pytest.mark.asyncio
async def test_admin_can_export_ca_private_key(setup_db, admin_user, org_ca):
    app = _app_for(admin_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/export/ca/{org_ca.id}/private-key")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_user_in_org_can_export_cert(setup_db, normal_user, org_cert):
    app = _app_for(normal_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/export/certificate/{org_cert.id}")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_other_org_cannot_export_cert(setup_db, user_other_org, org_cert):
    app = _app_for(user_other_org)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/export/certificate/{org_cert.id}")
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_user_without_cap_cannot_export_cert_key(setup_db, normal_user, org_cert):
    app = _app_for(normal_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(
            f"{settings.API_V1_STR}/export/certificate/{org_cert.id}/private-key"
        )
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_user_with_export_cap_can_export_cert_key(
    setup_db, user_with_export, org_cert
):
    app = _app_for(user_with_export)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(
            f"{settings.API_V1_STR}/export/certificate/{org_cert.id}/private-key"
        )
        assert resp.status_code == 200
        assert "BEGIN PRIVATE KEY" in resp.text


@pytest.mark.asyncio
async def test_user_in_org_can_export_chain(setup_db, normal_user, org_cert):
    app = _app_for(normal_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(
            f"{settings.API_V1_STR}/export/certificate/{org_cert.id}/chain"
        )
        assert resp.status_code == 200
        assert resp.text.count("BEGIN CERTIFICATE") >= 2


@pytest.mark.asyncio
async def test_other_org_cannot_export_chain(setup_db, user_other_org, org_cert):
    app = _app_for(user_other_org)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(
            f"{settings.API_V1_STR}/export/certificate/{org_cert.id}/chain"
        )
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_export_nonexistent_ca_cert_returns_404(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/export/ca/99999/certificate")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_export_nonexistent_ca_key_returns_404(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/export/ca/99999/private-key")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_export_nonexistent_cert_returns_404(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/export/certificate/99999")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_export_nonexistent_cert_key_returns_404(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(
            f"{settings.API_V1_STR}/export/certificate/99999/private-key"
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_export_cert_key_when_no_key_returns_404(setup_db, superuser, org_ca, db):
    cert_service = CertificateService(db)
    cert = await cert_service.create_certificate(
        ca_id=org_ca.id,
        common_name="nokey.example.com",
        subject_dn="CN=nokey.example.com",
        certificate_type=CertificateType.SERVER,
        include_private_key=False,
        organization_id=org_ca.organization_id,
        created_by_user_id=superuser.id,
    )
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(
            f"{settings.API_V1_STR}/export/certificate/{cert.id}/private-key"
        )
        assert resp.status_code == 404
        assert "does not have a private key" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_export_nonexistent_cert_chain_returns_404(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/export/certificate/99999/chain")
        assert resp.status_code == 404
