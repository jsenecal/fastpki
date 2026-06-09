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
from app.db.models import User, UserRole
from app.db.session import get_session
from app.services.ca import CAService
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
        name="OtherPermOrg", description="Other org"
    )


@pytest_asyncio.fixture
async def user_other_org(db: AsyncSession, other_org):
    user_service = UserService(db)
    return await user_service.create_user(
        username="otherorguserCA",
        email="otherorguserCA@example.com",
        password="password123",
        role=UserRole.USER,
        organization_id=other_org.id,
    )


@pytest_asyncio.fixture
async def user_with_create_ca(db: AsyncSession, test_org):
    user_service = UserService(db)
    return await user_service.create_user(
        username="cancreate_ca",
        email="cancreate_ca@example.com",
        password="password123",
        role=UserRole.USER,
        organization_id=test_org.id,
        can_create_ca=True,
    )


@pytest_asyncio.fixture
async def org_ca(db: AsyncSession, test_org, superuser):
    ca_service = CAService(db)
    return await ca_service.create_ca(
        name="OrgCA",
        subject_dn="CN=OrgCA",
        organization_id=test_org.id,
        created_by_user_id=superuser.id,
    )


@pytest.mark.asyncio
async def test_superuser_can_create_ca(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            f"{settings.API_V1_STR}/cas/",
            json={"name": "SU CA", "subject_dn": "CN=SU CA"},
        )
        assert resp.status_code == 201


@pytest.mark.asyncio
async def test_admin_in_org_can_create_ca(setup_db, admin_user):
    app = _app_for(admin_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            f"{settings.API_V1_STR}/cas/",
            json={"name": "Admin CA", "subject_dn": "CN=Admin CA"},
        )
        assert resp.status_code == 201


@pytest.mark.asyncio
async def test_user_with_capability_can_create_ca(setup_db, user_with_create_ca):
    app = _app_for(user_with_create_ca)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            f"{settings.API_V1_STR}/cas/",
            json={"name": "Cap CA", "subject_dn": "CN=Cap CA"},
        )
        assert resp.status_code == 201


@pytest.mark.asyncio
async def test_user_without_capability_cannot_create_ca(setup_db, normal_user):
    app = _app_for(normal_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            f"{settings.API_V1_STR}/cas/",
            json={"name": "NoPerms CA", "subject_dn": "CN=NoPerms CA"},
        )
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_superuser_sees_all_cas(setup_db, superuser, org_ca):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/cas/")
        assert resp.status_code == 200
        cas = resp.json()
        assert len(cas) >= 1


@pytest.mark.asyncio
async def test_user_sees_only_own_org_cas(setup_db, normal_user, org_ca):
    app = _app_for(normal_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/cas/")
        assert resp.status_code == 200
        cas = resp.json()
        for ca in cas:
            assert ca["organization_id"] == normal_user.organization_id


@pytest.mark.asyncio
async def test_user_other_org_cannot_read_ca(setup_db, user_other_org, org_ca):
    app = _app_for(user_other_org)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/cas/{org_ca.id}")
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_admin_in_org_can_read_ca(setup_db, admin_user, org_ca):
    app = _app_for(admin_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/cas/{org_ca.id}")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_user_in_org_can_read_ca(setup_db, normal_user, org_ca):
    app = _app_for(normal_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/cas/{org_ca.id}")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_user_without_capability_cannot_get_private_key(
    setup_db, normal_user, org_ca
):
    app = _app_for(normal_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/cas/{org_ca.id}/private-key")
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_admin_can_get_private_key(setup_db, admin_user, org_ca):
    app = _app_for(admin_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/cas/{org_ca.id}/private-key")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_user_without_capability_cannot_delete_ca(setup_db, normal_user, org_ca):
    app = _app_for(normal_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.delete(f"{settings.API_V1_STR}/cas/{org_ca.id}")
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_admin_can_delete_ca(setup_db, admin_user, org_ca):
    app = _app_for(admin_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.delete(f"{settings.API_V1_STR}/cas/{org_ca.id}")
        assert resp.status_code == 204


@pytest.mark.asyncio
async def test_get_nonexistent_ca_returns_404(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/cas/99999")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_nonexistent_ca_private_key_returns_404(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/cas/99999/private-key")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_ca_returns_404(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.delete(f"{settings.API_V1_STR}/cas/99999")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_ca_service_error_returns_400(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            f"{settings.API_V1_STR}/cas/",
            json={"name": "BadCA", "subject_dn": "invalid"},
        )
        assert resp.status_code == 400
        assert "Failed to create CA" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_unauthenticated_gets_401(client):
    resp = await client.get(f"{settings.API_V1_STR}/cas/")
    assert resp.status_code == 401
