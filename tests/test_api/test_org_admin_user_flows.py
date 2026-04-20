"""Org-admin (non-superuser) branches of add/remove user endpoints.

The superuser branch is covered by test_organizations.py; this file exercises
the branch that routes through user_can_*_organization helpers and asserts
cross-org boundaries are enforced.
"""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi import FastAPI, status
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import api_router
from app.api.deps import (
    get_current_active_admin_user,
    get_current_active_superuser,
    get_current_active_user,
    get_current_user,
)
from app.core.config import settings
from app.db.models import Organization, User, UserRole
from app.db.session import get_session
from app.services.organization import OrganizationService
from app.services.user import UserService
from tests.conftest import TestAuth, get_test_session


def _make_app(user: User) -> FastAPI:
    app = FastAPI()
    app.include_router(api_router, prefix=settings.API_V1_STR)
    app.dependency_overrides[get_session] = get_test_session
    app.dependency_overrides[get_current_user] = TestAuth(user)
    app.dependency_overrides[get_current_active_user] = TestAuth(user)
    app.dependency_overrides[get_current_active_admin_user] = TestAuth(user)
    app.dependency_overrides[get_current_active_superuser] = TestAuth(user)
    return app


@pytest_asyncio.fixture
async def other_org(db: AsyncSession) -> Organization:
    return await OrganizationService(db).create_organization(
        name="OtherOrg", description="Separate tenant"
    )


@pytest_asyncio.fixture
async def unassigned_user(db: AsyncSession) -> User:
    return await UserService(db).create_user(
        username="unassigned_member",
        email="unassigned@example.com",
        password="password123",
        role=UserRole.USER,
    )


@pytest_asyncio.fixture
async def other_org_member(db: AsyncSession, other_org: Organization) -> User:
    return await UserService(db).create_user(
        username="other_org_user",
        email="otheruser@example.com",
        password="password123",
        role=UserRole.USER,
        organization_id=other_org.id,
    )


@pytest_asyncio.fixture
async def admin_client_for(
    setup_db, admin_user: User
) -> AsyncGenerator[AsyncClient, None]:
    app = _make_app(admin_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_org_admin_adds_user_to_own_org(
    admin_client_for: AsyncClient,
    admin_user: User,
    unassigned_user: User,
):
    response = await admin_client_for.post(
        f"{settings.API_V1_STR}/organizations/"
        f"{admin_user.organization_id}/users/{unassigned_user.id}"
    )

    assert response.status_code == status.HTTP_200_OK, response.text
    body = response.json()
    assert body["id"] == unassigned_user.id
    assert body["organization_id"] == admin_user.organization_id


@pytest.mark.asyncio
async def test_org_admin_cannot_add_user_to_other_org(
    admin_client_for: AsyncClient,
    unassigned_user: User,
    other_org: Organization,
):
    response = await admin_client_for.post(
        f"{settings.API_V1_STR}/organizations/{other_org.id}/users/{unassigned_user.id}"
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_org_admin_removes_user_from_own_org(
    admin_client_for: AsyncClient,
    admin_user: User,
    normal_user: User,
):
    assert normal_user.organization_id == admin_user.organization_id

    response = await admin_client_for.delete(
        f"{settings.API_V1_STR}/organizations/"
        f"{admin_user.organization_id}/users/{normal_user.id}"
    )

    assert response.status_code == status.HTTP_200_OK, response.text
    body = response.json()
    assert body["id"] == normal_user.id
    assert body["organization_id"] is None


@pytest.mark.asyncio
async def test_org_admin_cannot_remove_user_from_other_org(
    admin_client_for: AsyncClient,
    other_org: Organization,
    other_org_member: User,
):
    response = await admin_client_for.delete(
        f"{settings.API_V1_STR}/organizations/"
        f"{other_org.id}/users/{other_org_member.id}"
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN
