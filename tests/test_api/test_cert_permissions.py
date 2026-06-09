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
        name="OtherCertOrg", description="Other"
    )


@pytest_asyncio.fixture
async def user_other_org(db: AsyncSession, other_org):
    user_service = UserService(db)
    return await user_service.create_user(
        username="otherorgcert",
        email="otherorgcert@example.com",
        password="password123",
        role=UserRole.USER,
        organization_id=other_org.id,
    )


@pytest_asyncio.fixture
async def user_with_create_cert(db: AsyncSession, test_org):
    user_service = UserService(db)
    return await user_service.create_user(
        username="cancreate_cert",
        email="cancreate_cert@example.com",
        password="password123",
        role=UserRole.USER,
        organization_id=test_org.id,
        can_create_cert=True,
    )


@pytest_asyncio.fixture
async def user_with_revoke(db: AsyncSession, test_org):
    user_service = UserService(db)
    return await user_service.create_user(
        username="canrevoke",
        email="canrevoke@example.com",
        password="password123",
        role=UserRole.USER,
        organization_id=test_org.id,
        can_revoke_cert=True,
    )


@pytest_asyncio.fixture
async def org_ca(db: AsyncSession, test_org, superuser):
    ca_service = CAService(db)
    return await ca_service.create_ca(
        name="CertPermCA",
        subject_dn="CN=CertPermCA",
        organization_id=test_org.id,
        created_by_user_id=superuser.id,
    )


@pytest_asyncio.fixture
async def org_cert(db: AsyncSession, org_ca, test_org, admin_user):
    cert_service = CertificateService(db)
    return await cert_service.create_certificate(
        ca_id=org_ca.id,
        common_name="perm.example.com",
        subject_dn="CN=perm.example.com",
        certificate_type=CertificateType.SERVER,
        organization_id=test_org.id,
        created_by_user_id=admin_user.id,
    )


@pytest.mark.asyncio
async def test_superuser_can_create_cert(setup_db, superuser, org_ca):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            f"{settings.API_V1_STR}/certificates/?ca_id={org_ca.id}",
            json={
                "common_name": "su.example.com",
                "subject_dn": "CN=su.example.com",
                "certificate_type": "server",
            },
        )
        assert resp.status_code == 201


@pytest.mark.asyncio
async def test_admin_can_create_cert(setup_db, admin_user, org_ca):
    app = _app_for(admin_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            f"{settings.API_V1_STR}/certificates/?ca_id={org_ca.id}",
            json={
                "common_name": "admin.example.com",
                "subject_dn": "CN=admin.example.com",
                "certificate_type": "server",
            },
        )
        assert resp.status_code == 201


@pytest.mark.asyncio
async def test_user_with_capability_can_create_cert(
    setup_db, user_with_create_cert, org_ca
):
    app = _app_for(user_with_create_cert)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            f"{settings.API_V1_STR}/certificates/?ca_id={org_ca.id}",
            json={
                "common_name": "cap.example.com",
                "subject_dn": "CN=cap.example.com",
                "certificate_type": "server",
            },
        )
        assert resp.status_code == 201


@pytest.mark.asyncio
async def test_user_without_capability_cannot_create_cert(
    setup_db, normal_user, org_ca
):
    app = _app_for(normal_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            f"{settings.API_V1_STR}/certificates/?ca_id={org_ca.id}",
            json={
                "common_name": "noperm.example.com",
                "subject_dn": "CN=noperm.example.com",
                "certificate_type": "server",
            },
        )
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_other_org_cannot_create_cert(setup_db, user_other_org, org_ca):
    app = _app_for(user_other_org)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            f"{settings.API_V1_STR}/certificates/?ca_id={org_ca.id}",
            json={
                "common_name": "other.example.com",
                "subject_dn": "CN=other.example.com",
                "certificate_type": "server",
            },
        )
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_user_in_org_can_read_cert(setup_db, normal_user, org_cert):
    app = _app_for(normal_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/certificates/{org_cert.id}")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_other_org_cannot_read_cert(setup_db, user_other_org, org_cert):
    app = _app_for(user_other_org)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/certificates/{org_cert.id}")
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_user_without_cap_cannot_get_private_key(setup_db, normal_user, org_cert):
    app = _app_for(normal_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(
            f"{settings.API_V1_STR}/certificates/{org_cert.id}/private-key"
        )
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_user_without_cap_cannot_revoke(setup_db, normal_user, org_cert):
    app = _app_for(normal_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            f"{settings.API_V1_STR}/certificates/{org_cert.id}/revoke",
            json={"reason": "test"},
        )
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_user_with_revoke_cap_can_revoke(setup_db, user_with_revoke, org_cert):
    app = _app_for(user_with_revoke)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            f"{settings.API_V1_STR}/certificates/{org_cert.id}/revoke",
            json={"reason": "test"},
        )
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_list_certs_scoped_to_org(setup_db, normal_user, org_cert):
    app = _app_for(normal_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/certificates/")
        assert resp.status_code == 200
        certs = resp.json()
        for cert in certs:
            assert cert["organization_id"] == normal_user.organization_id


@pytest.mark.asyncio
async def test_create_cert_nonexistent_ca_returns_404(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            f"{settings.API_V1_STR}/certificates/?ca_id=99999",
            json={
                "common_name": "ghost.example.com",
                "subject_dn": "CN=ghost.example.com",
                "certificate_type": "server",
            },
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_nonexistent_cert_returns_404(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/certificates/99999")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_nonexistent_cert_private_key_returns_404(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/certificates/99999/private-key")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_revoke_nonexistent_cert_returns_404(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            f"{settings.API_V1_STR}/certificates/99999/revoke",
            json={"reason": "test"},
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_revoke_already_revoked_cert_returns_409(setup_db, superuser, org_cert):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        # First revoke should succeed
        resp = await c.post(
            f"{settings.API_V1_STR}/certificates/{org_cert.id}/revoke",
            json={"reason": "first"},
        )
        assert resp.status_code == 200
        # Second revoke should conflict
        resp = await c.post(
            f"{settings.API_V1_STR}/certificates/{org_cert.id}/revoke",
            json={"reason": "second"},
        )
        assert resp.status_code == 409


@pytest.mark.asyncio
async def test_superuser_can_list_certs(setup_db, superuser, org_cert):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/certificates/")
        assert resp.status_code == 200
        certs = resp.json()
        assert len(certs) >= 1
