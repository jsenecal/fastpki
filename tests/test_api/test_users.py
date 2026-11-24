from datetime import timedelta

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_current_active_admin_user,
    get_current_active_superuser,
    get_current_active_user,
    get_current_user,
)
from app.core.config import settings
from app.db.models import User, UserRole
from app.db.session import get_session
from app.services.organization import OrganizationService
from app.services.user import UserService
from tests.conftest import TestAuth, create_test_app, get_test_session


def _app_for(user: User):
    app = create_test_app()
    app.dependency_overrides[get_session] = get_test_session
    app.dependency_overrides[get_current_user] = TestAuth(user)
    app.dependency_overrides[get_current_active_user] = TestAuth(user)
    app.dependency_overrides[get_current_active_superuser] = TestAuth(user)
    app.dependency_overrides[get_current_active_admin_user] = TestAuth(user)
    return app


@pytest_asyncio.fixture
async def target_user(db: AsyncSession, test_org):
    user_service = UserService(db)
    return await user_service.create_user(
        username="usertarget",
        email="usertarget@example.com",
        password="Pewter-Lagoon-3087",
        role=UserRole.USER,
        organization_id=test_org.id,
    )


@pytest.mark.asyncio
async def test_create_user(client):
    # Test creating a new user
    user_data = {
        "username": "newuser",
        "email": "new@example.com",
        "password": "Kestrel-Meadow-4821",
    }

    response = await client.post("/api/v1/users/", json=user_data)

    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "newuser"
    assert data["email"] == "new@example.com"
    assert data["role"] == UserRole.USER.value
    assert data["is_active"] is True
    assert "hashed_password" not in data  # Password should not be returned


@pytest.mark.asyncio
async def test_create_user_duplicate_username(client):
    # Bootstrap a superuser (first user) and authenticate so that we exercise
    # the duplicate-username branch rather than the unauthenticated-registration
    # policy gate.
    superuser_data = {
        "username": "superdup",
        "email": "superdup@example.com",
        "password": "Orchid-Basalt-6214",
        "role": UserRole.SUPERUSER.value,
    }
    await client.post("/api/v1/users/", json=superuser_data)
    login_response = await client.post(
        "/api/v1/auth/token",
        data={"username": "superdup", "password": "Orchid-Basalt-6214"},
    )
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    user_data = {
        "username": "duplicate",
        "email": "dup1@example.com",
        "password": "Kestrel-Meadow-4821",
    }
    await client.post("/api/v1/users/", json=user_data, headers=headers)

    duplicate_data = {
        "username": "duplicate",
        "email": "dup2@example.com",
        "password": "Lantern-Copper-9375",
    }
    response = await client.post("/api/v1/users/", json=duplicate_data, headers=headers)

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "Username already registered"


@pytest.mark.asyncio
async def test_create_user_duplicate_email(client):
    superuser_data = {
        "username": "superdupemail",
        "email": "superdupemail@example.com",
        "password": "Orchid-Basalt-6214",
        "role": UserRole.SUPERUSER.value,
    }
    await client.post("/api/v1/users/", json=superuser_data)
    login_response = await client.post(
        "/api/v1/auth/token",
        data={"username": "superdupemail", "password": "Orchid-Basalt-6214"},
    )
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    user_data = {
        "username": "emailuser1",
        "email": "same@example.com",
        "password": "Kestrel-Meadow-4821",
    }
    await client.post("/api/v1/users/", json=user_data, headers=headers)

    duplicate_data = {
        "username": "emailuser2",
        "email": "same@example.com",
        "password": "Lantern-Copper-9375",
    }
    response = await client.post("/api/v1/users/", json=duplicate_data, headers=headers)

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "Email already registered"


@pytest.mark.asyncio
async def test_unauthenticated_registration_does_not_leak_username_existence(client):
    """Issue #8: unauthenticated registration must return the same generic
    error regardless of whether the supplied username/email already exists,
    so attackers cannot enumerate accounts via the registration endpoint.
    """
    # Bootstrap a superuser (first user, no policy gate) so a known account exists.
    superuser_data = {
        "username": "enumadmin",
        "email": "enumadmin@example.com",
        "password": "Orchid-Basalt-6214",
        "role": UserRole.SUPERUSER.value,
    }
    await client.post("/api/v1/users/", json=superuser_data)

    # Unauthenticated probe using an existing username.
    existing_probe = await client.post(
        "/api/v1/users/",
        json={
            "username": "enumadmin",
            "email": "attacker@example.com",
            "password": "Pewter-Lagoon-3087",
        },
    )
    # Unauthenticated probe using a fresh username.
    fresh_probe = await client.post(
        "/api/v1/users/",
        json={
            "username": "doesnotexist",
            "email": "fresh@example.com",
            "password": "Pewter-Lagoon-3087",
        },
    )

    assert existing_probe.status_code == fresh_probe.status_code
    assert existing_probe.json()["detail"] == fresh_probe.json()["detail"]
    # Sanity-check that we hit the registration-policy gate, not a stale 400.
    assert existing_probe.status_code == 403


@pytest.mark.asyncio
async def test_unauthenticated_registration_does_not_leak_email_existence(client):
    """Issue #8: same as username — email existence must not leak either."""
    superuser_data = {
        "username": "emailenum",
        "email": "emailenum@example.com",
        "password": "Orchid-Basalt-6214",
        "role": UserRole.SUPERUSER.value,
    }
    await client.post("/api/v1/users/", json=superuser_data)

    existing_probe = await client.post(
        "/api/v1/users/",
        json={
            "username": "freshname1",
            "email": "emailenum@example.com",
            "password": "Pewter-Lagoon-3087",
        },
    )
    fresh_probe = await client.post(
        "/api/v1/users/",
        json={
            "username": "freshname2",
            "email": "neveruse@example.com",
            "password": "Pewter-Lagoon-3087",
        },
    )

    assert existing_probe.status_code == fresh_probe.status_code
    assert existing_probe.json()["detail"] == fresh_probe.json()["detail"]
    assert existing_probe.status_code == 403


@pytest.mark.asyncio
async def test_unauthenticated_elevated_role_probe_does_not_leak_existence(client):
    """Issue #8: probing with an elevated role must not leak existence either."""
    await client.post(
        "/api/v1/users/",
        json={
            "username": "roleenum",
            "email": "roleenum@example.com",
            "password": "Orchid-Basalt-6214",
            "role": UserRole.SUPERUSER.value,
        },
    )

    existing_probe = await client.post(
        "/api/v1/users/",
        json={
            "username": "roleenum",
            "email": "roleenum@example.com",
            "password": "Pewter-Lagoon-3087",
            "role": UserRole.ADMIN.value,
        },
    )
    fresh_probe = await client.post(
        "/api/v1/users/",
        json={
            "username": "rolefresh",
            "email": "rolefresh@example.com",
            "password": "Pewter-Lagoon-3087",
            "role": UserRole.ADMIN.value,
        },
    )

    assert existing_probe.status_code == fresh_probe.status_code == 403
    assert existing_probe.json()["detail"] == fresh_probe.json()["detail"]


@pytest.mark.asyncio
async def test_open_registration_still_reports_duplicates(client, monkeypatch):
    """With ALLOW_UNAUTHENTICATED_REGISTRATION enabled, duplicate errors are
    still returned: open self-registration inherently has to reject taken
    usernames, so this is a documented trade-off rather than a leak."""
    monkeypatch.setattr(settings, "ALLOW_UNAUTHENTICATED_REGISTRATION", True)
    await client.post(
        "/api/v1/users/",
        json={
            "username": "openfirst",
            "email": "openfirst@example.com",
            "password": "Orchid-Basalt-6214",
            "role": UserRole.SUPERUSER.value,
        },
    )

    response = await client.post(
        "/api/v1/users/",
        json={
            "username": "openfirst",
            "email": "other@example.com",
            "password": "Pewter-Lagoon-3087",
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Username already registered"


@pytest.mark.asyncio
async def test_get_users(client, db):
    # First create a superuser (will be the first user so has permission)
    superuser_data = {
        "username": "superadmin",
        "email": "super@example.com",
        "password": "Granite-Willow-5739",
        "role": UserRole.SUPERUSER.value,
    }

    await client.post("/api/v1/users/", json=superuser_data)

    # Login as superuser
    login_response = await client.post(
        "/api/v1/auth/token",
        data={"username": "superadmin", "password": "Granite-Willow-5739"},
    )
    token = login_response.json()["access_token"]

    # Create multiple users using the superuser token
    users_data = [
        {
            "username": "user1",
            "email": "user1@example.com",
            "password": "Juniper-Falcon-1111",
        },
        {
            "username": "user2",
            "email": "user2@example.com",
            "password": "Juniper-Falcon-2222",
        },
        {
            "username": "user3",
            "email": "user3@example.com",
            "password": "Juniper-Falcon-3333",
        },
    ]

    headers = {"Authorization": f"Bearer {token}"}
    for user_data in users_data:
        await client.post("/api/v1/users/", json=user_data, headers=headers)

    # Get list of users
    headers = {"Authorization": f"Bearer {token}"}
    response = await client.get("/api/v1/users/", headers=headers)

    assert response.status_code == 200
    data = response.json()

    # Should have at least 4 users (3 regular + 1 superuser)
    assert len(data) >= 4

    # Check if our created users are in the list
    usernames = [user["username"] for user in data]
    assert "user1" in usernames
    assert "user2" in usernames
    assert "user3" in usernames


@pytest.mark.asyncio
async def test_get_user_by_id(client, db):
    # First create a superuser (will be the first user so has permission)
    superuser_data = {
        "username": "superget",
        "email": "superget@example.com",
        "password": "Granite-Willow-5739",
        "role": UserRole.SUPERUSER.value,
    }

    await client.post("/api/v1/users/", json=superuser_data)

    # Login as superuser
    login_response = await client.post(
        "/api/v1/auth/token",
        data={"username": "superget", "password": "Granite-Willow-5739"},
    )
    token = login_response.json()["access_token"]

    # Create a test user with superuser token
    user_data = {
        "username": "getbyid",
        "email": "getbyid@example.com",
        "password": "Kestrel-Meadow-4821",
    }

    headers = {"Authorization": f"Bearer {token}"}
    create_response = await client.post(
        "/api/v1/users/", json=user_data, headers=headers
    )
    user_id = create_response.json()["id"]

    # Get user by ID
    headers = {"Authorization": f"Bearer {token}"}
    response = await client.get(f"/api/v1/users/{user_id}", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == user_id
    assert data["username"] == "getbyid"
    assert data["email"] == "getbyid@example.com"


@pytest.mark.asyncio
async def test_update_user(client, db):
    # First create a superuser (will be the first user so has permission)
    superuser_data = {
        "username": "superupdate",
        "email": "superupdate@example.com",
        "password": "Granite-Willow-5739",
        "role": UserRole.SUPERUSER.value,
    }

    await client.post("/api/v1/users/", json=superuser_data)

    # Login as superuser
    login_response = await client.post(
        "/api/v1/auth/token",
        data={"username": "superupdate", "password": "Granite-Willow-5739"},
    )
    token = login_response.json()["access_token"]

    # Create a test user with superuser token
    user_data = {
        "username": "updateme",
        "email": "update@example.com",
        "password": "Kestrel-Meadow-4821",
    }

    headers = {"Authorization": f"Bearer {token}"}
    create_response = await client.post(
        "/api/v1/users/", json=user_data, headers=headers
    )
    user_id = create_response.json()["id"]

    # Update user
    update_data = {
        "email": "updated@example.com",
        "role": UserRole.ADMIN.value,
        "is_active": False,
    }

    headers = {"Authorization": f"Bearer {token}"}
    response = await client.patch(
        f"/api/v1/users/{user_id}", json=update_data, headers=headers
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == user_id
    assert data["username"] == "updateme"  # Username should not change
    assert data["email"] == "updated@example.com"
    assert data["role"] == UserRole.ADMIN.value
    assert data["is_active"] is False


@pytest.mark.asyncio
async def test_delete_user(client, db):
    # First create a superuser (will be the first user so has permission)
    superuser_data = {
        "username": "superdelete",
        "email": "superdelete@example.com",
        "password": "Granite-Willow-5739",
        "role": UserRole.SUPERUSER.value,
    }

    await client.post("/api/v1/users/", json=superuser_data)

    # Login as superuser
    login_response = await client.post(
        "/api/v1/auth/token",
        data={"username": "superdelete", "password": "Granite-Willow-5739"},
    )
    token = login_response.json()["access_token"]

    # Create a test user with superuser token
    user_data = {
        "username": "deleteme",
        "email": "delete@example.com",
        "password": "Kestrel-Meadow-4821",
    }

    headers = {"Authorization": f"Bearer {token}"}
    create_response = await client.post(
        "/api/v1/users/", json=user_data, headers=headers
    )
    user_id = create_response.json()["id"]

    # Delete user
    headers = {"Authorization": f"Bearer {token}"}
    response = await client.delete(f"/api/v1/users/{user_id}", headers=headers)

    assert response.status_code == 204

    # Verify user is deleted
    get_response = await client.get(f"/api/v1/users/{user_id}", headers=headers)
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_regular_user_cannot_access_other_users(client, db):
    # First create a superuser (will be the first user so has permission)
    superuser_data = {
        "username": "superaccess",
        "email": "superaccess@example.com",
        "password": "Granite-Willow-5739",
        "role": UserRole.SUPERUSER.value,
    }

    await client.post("/api/v1/users/", json=superuser_data)

    # Login as superuser
    login_response = await client.post(
        "/api/v1/auth/token",
        data={"username": "superaccess", "password": "Granite-Willow-5739"},
    )
    super_token = login_response.json()["access_token"]

    # Create two regular users with superuser token
    user1_data = {
        "username": "user1access",
        "email": "user1@example.com",
        "password": "Juniper-Falcon-1111",
    }

    user2_data = {
        "username": "user2access",
        "email": "user2@example.com",
        "password": "Juniper-Falcon-2222",
    }

    # We'll use _ for unused response variable
    headers = {"Authorization": f"Bearer {super_token}"}
    _ = await client.post("/api/v1/users/", json=user1_data, headers=headers)
    user2_response = await client.post(
        "/api/v1/users/", json=user2_data, headers=headers
    )

    user2_id = user2_response.json()["id"]

    # Login as user1
    login_response = await client.post(
        "/api/v1/auth/token",
        data={"username": "user1access", "password": "Juniper-Falcon-1111"},
    )
    token = login_response.json()["access_token"]

    # Try to get user2's details
    headers = {"Authorization": f"Bearer {token}"}
    response = await client.get(f"/api/v1/users/{user2_id}", headers=headers)

    # Should be forbidden
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_admin_cannot_read_user_in_other_organization(db: AsyncSession):
    """Issue #9: an ADMIN must only see users within their own organization.

    Previously the role-only check let an ADMIN in Org A read any user by ID,
    leaking emails, capabilities, and org membership across boundaries.
    """
    org_service = OrganizationService(db)
    org_a = await org_service.create_organization(name="OrgA9", description="A")
    org_b = await org_service.create_organization(name="OrgB9", description="B")

    user_service = UserService(db)
    admin_a = await user_service.create_user(
        username="admin_a_9",
        email="admin_a_9@example.com",
        password="Pewter-Lagoon-3087",
        role=UserRole.ADMIN,
        organization_id=org_a.id,
    )
    user_b = await user_service.create_user(
        username="user_b_9",
        email="user_b_9@example.com",
        password="Pewter-Lagoon-3087",
        role=UserRole.USER,
        organization_id=org_b.id,
    )

    app = _app_for(admin_a)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/users/{user_b.id}")
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_admin_can_read_user_in_own_organization(db: AsyncSession):
    """An ADMIN must still be able to read users within their own organization."""
    org_service = OrganizationService(db)
    org_a = await org_service.create_organization(name="OrgA9Own", description="A")

    user_service = UserService(db)
    admin_a = await user_service.create_user(
        username="admin_a_own_9",
        email="admin_a_own_9@example.com",
        password="Pewter-Lagoon-3087",
        role=UserRole.ADMIN,
        organization_id=org_a.id,
    )
    user_a = await user_service.create_user(
        username="user_a_own_9",
        email="user_a_own_9@example.com",
        password="Pewter-Lagoon-3087",
        role=UserRole.USER,
        organization_id=org_a.id,
    )

    app = _app_for(admin_a)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/users/{user_a.id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == user_a.id


@pytest.mark.asyncio
async def test_superuser_can_still_read_users_across_organizations(db: AsyncSession):
    """SUPERUSER retains cross-organization visibility."""
    org_service = OrganizationService(db)
    org_a = await org_service.create_organization(name="OrgA9Super", description="A")
    org_b = await org_service.create_organization(name="OrgB9Super", description="B")

    user_service = UserService(db)
    su = await user_service.create_user(
        username="su_9",
        email="su_9@example.com",
        password="Pewter-Lagoon-3087",
        role=UserRole.SUPERUSER,
        organization_id=org_a.id,
    )
    user_b = await user_service.create_user(
        username="user_b_super_9",
        email="user_b_super_9@example.com",
        password="Pewter-Lagoon-3087",
        role=UserRole.USER,
        organization_id=org_b.id,
    )

    app = _app_for(su)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/users/{user_b.id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == user_b.id


@pytest.mark.asyncio
async def test_admin_without_organization_cannot_read_other_users(db: AsyncSession):
    """An ADMIN with no organization assigned must not gain read access to
    arbitrary users (defense-in-depth for the cross-org check)."""
    org_service = OrganizationService(db)
    org_b = await org_service.create_organization(name="OrgB9NoOrg", description="B")

    user_service = UserService(db)
    orphan_admin = await user_service.create_user(
        username="orphan_admin_9",
        email="orphan_admin_9@example.com",
        password="Pewter-Lagoon-3087",
        role=UserRole.ADMIN,
        organization_id=None,
    )
    user_b = await user_service.create_user(
        username="user_b_noorg_9",
        email="user_b_noorg_9@example.com",
        password="Pewter-Lagoon-3087",
        role=UserRole.USER,
        organization_id=org_b.id,
    )

    app = _app_for(orphan_admin)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/users/{user_b.id}")
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_get_nonexistent_user_returns_404(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get(f"{settings.API_V1_STR}/users/99999")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_nonexistent_user_returns_404(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.patch(
            f"{settings.API_V1_STR}/users/99999",
            json={"email": "nope@example.com"},
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_non_superuser_cannot_change_role(setup_db, admin_user, target_user):
    app = _app_for(admin_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.patch(
            f"{settings.API_V1_STR}/users/{target_user.id}",
            json={"role": UserRole.ADMIN.value},
        )
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_non_superuser_cannot_change_is_active(setup_db, admin_user, target_user):
    app = _app_for(admin_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.patch(
            f"{settings.API_V1_STR}/users/{target_user.id}",
            json={"is_active": False},
        )
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_regular_user_can_update_own_email(setup_db, normal_user):
    app = _app_for(normal_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.patch(
            f"{settings.API_V1_STR}/users/{normal_user.id}",
            json={"email": "newemail@example.com"},
        )
        assert resp.status_code == 200
        assert resp.json()["email"] == "newemail@example.com"


@pytest.mark.asyncio
async def test_superuser_cannot_downgrade_self(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.patch(
            f"{settings.API_V1_STR}/users/{superuser.id}",
            json={"role": UserRole.USER.value},
        )
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_delete_nonexistent_user_returns_404(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.delete(f"{settings.API_V1_STR}/users/99999")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_superuser_cannot_delete_self(setup_db, superuser):
    app = _app_for(superuser)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.delete(f"{settings.API_V1_STR}/users/{superuser.id}")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Issue #69: user creation must be authorized and organization-scoped.
# ---------------------------------------------------------------------------

USERS_URL = f"{settings.API_V1_STR}/users/"


def _bearer(db: AsyncSession, user: User) -> dict[str, str]:
    token = UserService(db).create_access_token(
        data={"sub": user.username, "id": user.id, "role": user.role},
        expires_delta=timedelta(minutes=30),
    )
    return {"Authorization": f"Bearer {token}"}


def _new_user_payload(name: str, **extra: object) -> dict[str, object]:
    return {
        "username": name,
        "email": f"{name}@example.com",
        "password": "Pewter-Lagoon-3087",
        **extra,
    }


@pytest_asyncio.fixture
async def org_b(db: AsyncSession):
    return await OrganizationService(db).create_organization(
        name="OrgB69", description="Another tenant"
    )


@pytest.mark.asyncio
async def test_create_user_as_regular_user_forbidden(
    client, normal_user_token_headers, test_org
):
    resp = await client.post(
        USERS_URL,
        json=_new_user_payload("byuser69", organization_id=test_org.id),
        headers=normal_user_token_headers,
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Not enough permissions to create users"


@pytest.mark.asyncio
async def test_create_user_as_regular_user_does_not_leak_duplicates(
    client, normal_user, normal_user_token_headers
):
    """The USER-role gate runs before uniqueness checks (issue #8 ordering)."""
    resp = await client.post(
        USERS_URL,
        json={
            "username": normal_user.username,
            "email": normal_user.email,
            "password": "Pewter-Lagoon-3087",
        },
        headers=normal_user_token_headers,
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_admin_cannot_create_user_in_other_org(
    client, admin_token_headers, org_b
):
    resp = await client.post(
        USERS_URL,
        json=_new_user_payload("crossorg69", organization_id=org_b.id),
        headers=admin_token_headers,
    )
    assert resp.status_code == 403
    assert (
        resp.json()["detail"]
        == "Not enough permissions to create users in this organization"
    )


@pytest.mark.asyncio
async def test_admin_cross_org_create_does_not_leak_duplicates(
    client, admin_token_headers, org_b, superuser
):
    resp = await client.post(
        USERS_URL,
        json={
            "username": superuser.username,
            "email": superuser.email,
            "password": "Pewter-Lagoon-3087",
            "organization_id": org_b.id,
        },
        headers=admin_token_headers,
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_admin_creates_user_in_own_org(client, admin_token_headers, test_org):
    resp = await client.post(
        USERS_URL,
        json=_new_user_payload("ownorg69", organization_id=test_org.id),
        headers=admin_token_headers,
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["organization_id"] == test_org.id
    assert resp.json()["role"] == UserRole.USER.value


@pytest.mark.asyncio
async def test_admin_create_defaults_to_own_org(client, admin_token_headers, test_org):
    resp = await client.post(
        USERS_URL,
        json=_new_user_payload("defaultorg69"),
        headers=admin_token_headers,
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["organization_id"] == test_org.id


@pytest.mark.asyncio
async def test_admin_cannot_create_admin(client, admin_token_headers):
    resp = await client.post(
        USERS_URL,
        json=_new_user_payload("adminbyadmin69", role=UserRole.ADMIN.value),
        headers=admin_token_headers,
    )
    assert resp.status_code == 403
    assert (
        resp.json()["detail"]
        == "Not enough permissions to create admin or superuser accounts"
    )


@pytest.mark.asyncio
async def test_admin_without_org_cannot_create_users(client, db: AsyncSession):
    orphan = await UserService(db).create_user(
        username="orphanadmin69",
        email="orphanadmin69@example.com",
        password="Pewter-Lagoon-3087",
        role=UserRole.ADMIN,
    )
    resp = await client.post(
        USERS_URL, json=_new_user_payload("byorphan69"), headers=_bearer(db, orphan)
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_superuser_creates_user_in_any_org(
    client, superuser_token_headers, org_b
):
    resp = await client.post(
        USERS_URL,
        json=_new_user_payload(
            "superorg69", organization_id=org_b.id, role=UserRole.ADMIN.value
        ),
        headers=superuser_token_headers,
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["organization_id"] == org_b.id
    assert resp.json()["role"] == UserRole.ADMIN.value


@pytest.mark.asyncio
async def test_superuser_create_without_org_leaves_unassigned(
    client, superuser_token_headers
):
    resp = await client.post(
        USERS_URL,
        json=_new_user_payload("superunassigned69"),
        headers=superuser_token_headers,
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["organization_id"] is None


@pytest.mark.asyncio
async def test_anonymous_self_registration_cannot_join_an_org(
    client, monkeypatch, superuser, org_b
):
    monkeypatch.setattr(settings, "ALLOW_UNAUTHENTICATED_REGISTRATION", True)
    resp = await client.post(
        USERS_URL, json=_new_user_payload("selfreg69", organization_id=org_b.id)
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["organization_id"] is None


@pytest.mark.asyncio
async def test_invalid_token_is_rejected_not_treated_as_anonymous(
    client, monkeypatch, superuser
):
    """An explicitly supplied but invalid token must 401, even when open
    registration would otherwise accept an anonymous request."""
    monkeypatch.setattr(settings, "ALLOW_UNAUTHENTICATED_REGISTRATION", True)
    resp = await client.post(
        USERS_URL,
        json=_new_user_payload("badtoken69"),
        headers={"Authorization": "Bearer not-a-valid-token"},
    )
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


@pytest.mark.asyncio
async def test_inactive_caller_cannot_create_users(client, db: AsyncSession):
    inactive_su = await UserService(db).create_user(
        username="inactivesu69",
        email="inactivesu69@example.com",
        password="Pewter-Lagoon-3087",
        role=UserRole.SUPERUSER,
    )
    # Flip the flag directly: update_user() would also invalidate tokens (401),
    # whereas this exercises a still-valid token held by an inactive account.
    inactive_su.is_active = False
    db.add(inactive_su)
    await db.commit()
    resp = await client.post(
        USERS_URL,
        json=_new_user_payload("byinactive69"),
        headers=_bearer(db, inactive_su),
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Inactive user"


@pytest.mark.asyncio
async def test_null_role_is_treated_as_user(client, admin_token_headers, test_org):
    resp = await client.post(
        USERS_URL,
        json=_new_user_payload("nullrole69", role=None),
        headers=admin_token_headers,
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["role"] == UserRole.USER.value


# ---------------------------------------------------------------------------
# Issue #69 (related): PATCH /users/{id} must be organization-scoped for ADMINs.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_cannot_update_user_in_other_org(
    setup_db, admin_user, db: AsyncSession, org_b
):
    victim = await UserService(db).create_user(
        username="victim69",
        email="victim69@example.com",
        password="Pewter-Lagoon-3087",
        organization_id=org_b.id,
    )
    transport = ASGITransport(app=_app_for(admin_user))
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.patch(
            f"{settings.API_V1_STR}/users/{victim.id}",
            json={"password": "hijacked-password"},
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_admin_cannot_update_superuser(setup_db, admin_user, db: AsyncSession):
    su = await UserService(db).create_user(
        username="su_in_org69",
        email="su_in_org69@example.com",
        password="Pewter-Lagoon-3087",
        role=UserRole.SUPERUSER,
        organization_id=admin_user.organization_id,
    )
    transport = ASGITransport(app=_app_for(admin_user))
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.patch(
            f"{settings.API_V1_STR}/users/{su.id}",
            json={"email": "takeover@example.com"},
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_regular_user_cannot_grant_self_capabilities(setup_db, normal_user):
    transport = ASGITransport(app=_app_for(normal_user))
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.patch(
            f"{settings.API_V1_STR}/users/{normal_user.id}",
            json={"can_create_ca": True},
        )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Not enough permissions to change capabilities"


@pytest.mark.asyncio
async def test_admin_can_update_user_in_own_org(setup_db, admin_user, target_user):
    transport = ASGITransport(app=_app_for(admin_user))
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.patch(
            f"{settings.API_V1_STR}/users/{target_user.id}",
            json={"email": "changed69@example.com"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["email"] == "changed69@example.com"


# ---------------------------------------------------------------------------
# Capability flags on POST /users/ must be honored for privileged callers and
# refused (not silently dropped) for anonymous self-registration.
# ---------------------------------------------------------------------------

ALL_CAPABILITIES = {
    "can_create_ca": True,
    "can_create_cert": True,
    "can_revoke_cert": True,
    "can_export_private_key": True,
    "can_delete_ca": True,
}


@pytest.mark.asyncio
async def test_superuser_create_user_persists_capabilities(
    client, superuser_token_headers, db: AsyncSession
):
    resp = await client.post(
        USERS_URL,
        json=_new_user_payload("capsbysu", **ALL_CAPABILITIES),
        headers=superuser_token_headers,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    for flag in ALL_CAPABILITIES:
        assert body[flag] is True, flag
    stored = await UserService(db).get_user_by_id(body["id"])
    assert stored is not None
    await db.refresh(stored)
    assert stored.can_create_ca
    assert stored.can_delete_ca


@pytest.mark.asyncio
async def test_admin_create_user_in_own_org_persists_capabilities(
    client, admin_token_headers, test_org
):
    resp = await client.post(
        USERS_URL,
        json=_new_user_payload(
            "capsbyadmin", can_create_cert=True, can_revoke_cert=True
        ),
        headers=admin_token_headers,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["organization_id"] == test_org.id
    assert body["can_create_cert"] is True
    assert body["can_revoke_cert"] is True
    assert body["can_create_ca"] is False
    assert body["can_export_private_key"] is False
    assert body["can_delete_ca"] is False


@pytest.mark.asyncio
async def test_first_user_bootstrap_passes_capabilities_through(client):
    resp = await client.post(
        USERS_URL,
        json=_new_user_payload(
            "capsfirst", role=UserRole.SUPERUSER.value, can_create_ca=True
        ),
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["can_create_ca"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", sorted(ALL_CAPABILITIES))
async def test_anonymous_registration_cannot_set_capabilities(
    client, monkeypatch, superuser, flag
):
    monkeypatch.setattr(settings, "ALLOW_UNAUTHENTICATED_REGISTRATION", True)
    resp = await client.post(
        USERS_URL, json=_new_user_payload("anoncaps", **{flag: True})
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Not enough permissions to set capabilities"


@pytest.mark.asyncio
async def test_anonymous_capability_probe_does_not_leak_existence(
    client, monkeypatch, superuser
):
    """The capability gate runs before uniqueness checks (issue #8 ordering)."""
    monkeypatch.setattr(settings, "ALLOW_UNAUTHENTICATED_REGISTRATION", True)
    existing = await client.post(
        USERS_URL,
        json={
            "username": superuser.username,
            "email": superuser.email,
            "password": "Pewter-Lagoon-3087",
            "can_create_ca": True,
        },
    )
    fresh = await client.post(
        USERS_URL, json=_new_user_payload("anoncapsfresh", can_create_ca=True)
    )
    assert existing.status_code == fresh.status_code == 403
    assert existing.json()["detail"] == fresh.json()["detail"]


@pytest.mark.asyncio
async def test_anonymous_registration_with_false_capabilities_succeeds(
    client, monkeypatch, superuser
):
    """The CLI always sends every flag; explicit ``false`` must still register."""
    monkeypatch.setattr(settings, "ALLOW_UNAUTHENTICATED_REGISTRATION", True)
    resp = await client.post(
        USERS_URL,
        json=_new_user_payload(
            "anonfalsecaps", **dict.fromkeys(ALL_CAPABILITIES, False)
        ),
    )
    assert resp.status_code == 201, resp.text
    assert not any(resp.json()[flag] for flag in ALL_CAPABILITIES)
