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
        name="OtherCapOrg", description="Other org"
    )


@pytest_asyncio.fixture
async def target_user(db: AsyncSession, test_org):
    user_service = UserService(db)
    return await user_service.create_user(
        username="targetuser",
        email="targetuser@example.com",
        password="password123",
        role=UserRole.USER,
        organization_id=test_org.id,
    )


@pytest_asyncio.fixture
async def admin_other_org(db: AsyncSession, other_org):
    user_service = UserService(db)
    return await user_service.create_user(
        username="adminother",
        email="adminother@example.com",
        password="password123",
        role=UserRole.ADMIN,
        organization_id=other_org.id,
    )


@pytest.mark.asyncio
async def test_admin_can_set_user_capabilities(
    setup_db,
    admin_user,
    target_user,
):
    app = _app_for(admin_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.patch(
            f"{settings.API_V1_STR}/users/{target_user.id}",
            json={
                "can_create_ca": True,
                "can_create_cert": True,
                "can_revoke_cert": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["can_create_ca"] is True
        assert data["can_create_cert"] is True
        assert data["can_revoke_cert"] is False
        assert data["can_export_private_key"] is False
        assert data["can_delete_ca"] is False


@pytest.mark.asyncio
async def test_superuser_can_set_capabilities(
    setup_db,
    superuser,
    target_user,
):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.patch(
            f"{settings.API_V1_STR}/users/{target_user.id}",
            json={
                "can_export_private_key": True,
                "can_delete_ca": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["can_export_private_key"] is True
        assert data["can_delete_ca"] is True


@pytest.mark.asyncio
async def test_regular_user_cannot_set_capabilities(
    setup_db,
    normal_user,
    target_user,
):
    app = _app_for(normal_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.patch(
            f"{settings.API_V1_STR}/users/{target_user.id}",
            json={"can_create_ca": True},
        )
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_admin_cannot_set_capabilities_for_other_org(
    setup_db,
    admin_other_org,
    target_user,
):
    app = _app_for(admin_other_org)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.patch(
            f"{settings.API_V1_STR}/users/{target_user.id}",
            json={"can_create_ca": True},
        )
        assert resp.status_code == 403
