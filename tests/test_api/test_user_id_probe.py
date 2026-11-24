"""User-ID enumeration: endpoints taking a ``user_id`` must answer the same way
for a non-existent account and an account the caller is not allowed to see.

Out-of-scope accounts are reported as ``404 User not found`` (the convention
already used for service accounts), so an org admin cannot map user IDs that
belong to other organizations.
"""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy.ext.asyncio import AsyncSession

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
from tests.conftest import TestAuth, create_test_app, get_test_session

MISSING_ID = 99999
NOT_FOUND = {"detail": "User not found"}
PASSWORD = "Pewter-Lagoon-3087"


def _app_for(user: User) -> FastAPI:
    app = create_test_app()
    app.dependency_overrides[get_session] = get_test_session
    app.dependency_overrides[get_current_user] = TestAuth(user)
    app.dependency_overrides[get_current_active_user] = TestAuth(user)
    app.dependency_overrides[get_current_active_admin_user] = TestAuth(user)
    app.dependency_overrides[get_current_active_superuser] = TestAuth(user)
    return app


async def _client(user: User) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=_app_for(user))
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def org_b(db: AsyncSession) -> Organization:
    return await OrganizationService(db).create_organization(
        name="ProbeOrgB", description="Another tenant"
    )


@pytest_asyncio.fixture
async def other_org_user(db: AsyncSession, org_b: Organization) -> User:
    return await UserService(db).create_user(
        username="probe_other",
        email="probe_other@example.com",
        password=PASSWORD,
        organization_id=org_b.id,
    )


@pytest_asyncio.fixture
async def unassigned_superuser(db: AsyncSession) -> User:
    return await UserService(db).create_user(
        username="probe_su",
        email="probe_su@example.com",
        password=PASSWORD,
        role=UserRole.SUPERUSER,
    )


@pytest_asyncio.fixture
async def admin_client(setup_db, admin_user: User) -> AsyncGenerator[AsyncClient, None]:
    async for c in _client(admin_user):
        yield c


@pytest_asyncio.fixture
async def user_client(setup_db, normal_user: User) -> AsyncGenerator[AsyncClient, None]:
    async for c in _client(normal_user):
        yield c


def _same(a: Response, b: Response) -> None:
    assert a.status_code == b.status_code == 404, (a.text, b.text)
    assert a.json() == b.json() == NOT_FOUND


# --- GET /users/{id} -------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_get_other_org_user_looks_missing(
    admin_client: AsyncClient, other_org_user: User
):
    url = f"{settings.API_V1_STR}/users"
    _same(
        await admin_client.get(f"{url}/{other_org_user.id}"),
        await admin_client.get(f"{url}/{MISSING_ID}"),
    )


@pytest.mark.asyncio
async def test_user_get_peer_looks_missing(user_client: AsyncClient, admin_user: User):
    """A regular user may only see themselves, even within their org."""
    url = f"{settings.API_V1_STR}/users"
    _same(
        await user_client.get(f"{url}/{admin_user.id}"),
        await user_client.get(f"{url}/{MISSING_ID}"),
    )


@pytest.mark.asyncio
async def test_admin_without_org_get_user_looks_missing(
    setup_db, db: AsyncSession, other_org_user: User
):
    orphan = await UserService(db).create_user(
        username="probe_orphan",
        email="probe_orphan@example.com",
        password=PASSWORD,
        role=UserRole.ADMIN,
    )
    url = f"{settings.API_V1_STR}/users"
    async for c in _client(orphan):
        _same(
            await c.get(f"{url}/{other_org_user.id}"),
            await c.get(f"{url}/{MISSING_ID}"),
        )


# --- PATCH /users/{id} -----------------------------------------------------


@pytest.mark.asyncio
async def test_admin_patch_other_org_user_looks_missing(
    admin_client: AsyncClient, other_org_user: User
):
    url = f"{settings.API_V1_STR}/users"
    body = {"email": "probe@example.com"}
    _same(
        await admin_client.patch(f"{url}/{other_org_user.id}", json=body),
        await admin_client.patch(f"{url}/{MISSING_ID}", json=body),
    )


@pytest.mark.asyncio
async def test_user_patch_peer_looks_missing(
    user_client: AsyncClient, admin_user: User
):
    url = f"{settings.API_V1_STR}/users"
    body = {"can_create_ca": True}
    _same(
        await user_client.patch(f"{url}/{admin_user.id}", json=body),
        await user_client.patch(f"{url}/{MISSING_ID}", json=body),
    )


@pytest.mark.asyncio
async def test_admin_patch_visible_superuser_is_still_forbidden(
    admin_client: AsyncClient, admin_user: User, db: AsyncSession
):
    """A visible-but-unmanageable account keeps its 403; nothing is leaked
    since the admin can already read it."""
    su = await UserService(db).create_user(
        username="probe_su_in_org",
        email="probe_su_in_org@example.com",
        password=PASSWORD,
        role=UserRole.SUPERUSER,
        organization_id=admin_user.organization_id,
    )
    resp = await admin_client.patch(
        f"{settings.API_V1_STR}/users/{su.id}", json={"email": "x@example.com"}
    )
    assert resp.status_code == 403


# --- POST /organizations/{org}/users/{id} ----------------------------------


@pytest.mark.asyncio
async def test_admin_add_other_org_user_looks_missing(
    admin_client: AsyncClient, admin_user: User, other_org_user: User
):
    url = f"{settings.API_V1_STR}/organizations/{admin_user.organization_id}/users"
    _same(
        await admin_client.post(f"{url}/{other_org_user.id}"),
        await admin_client.post(f"{url}/{MISSING_ID}"),
    )


@pytest.mark.asyncio
async def test_admin_add_superuser_looks_missing(
    admin_client: AsyncClient, admin_user: User, unassigned_superuser: User
):
    url = f"{settings.API_V1_STR}/organizations/{admin_user.organization_id}/users"
    _same(
        await admin_client.post(f"{url}/{unassigned_superuser.id}"),
        await admin_client.post(f"{url}/{MISSING_ID}"),
    )


@pytest.mark.asyncio
async def test_admin_add_to_foreign_org_does_not_depend_on_user_id(
    admin_client: AsyncClient, org_b: Organization, other_org_user: User
):
    url = f"{settings.API_V1_STR}/organizations/{org_b.id}/users"
    existing = await admin_client.post(f"{url}/{other_org_user.id}")
    missing = await admin_client.post(f"{url}/{MISSING_ID}")
    assert existing.status_code == missing.status_code == 403
    assert existing.json() == missing.json()


@pytest.mark.asyncio
async def test_superuser_add_missing_user_is_404(
    setup_db, superuser: User, test_org: Organization
):
    async for c in _client(superuser):
        resp = await c.post(
            f"{settings.API_V1_STR}/organizations/{test_org.id}/users/{MISSING_ID}"
        )
    assert resp.status_code == 404
    assert resp.json() == NOT_FOUND


# --- DELETE /organizations/{org}/users/{id} --------------------------------


@pytest.mark.asyncio
async def test_admin_remove_other_org_user_looks_missing(
    admin_client: AsyncClient, admin_user: User, other_org_user: User
):
    url = f"{settings.API_V1_STR}/organizations/{admin_user.organization_id}/users"
    _same(
        await admin_client.delete(f"{url}/{other_org_user.id}"),
        await admin_client.delete(f"{url}/{MISSING_ID}"),
    )


@pytest.mark.asyncio
async def test_admin_remove_from_foreign_org_does_not_depend_on_user_id(
    admin_client: AsyncClient, org_b: Organization, other_org_user: User
):
    url = f"{settings.API_V1_STR}/organizations/{org_b.id}/users"
    existing = await admin_client.delete(f"{url}/{other_org_user.id}")
    missing = await admin_client.delete(f"{url}/{MISSING_ID}")
    assert existing.status_code == missing.status_code == 403
    assert existing.json() == missing.json()
