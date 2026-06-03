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
        password="password123",
        role=UserRole.USER,
        organization_id=test_org.id,
    )


@pytest.mark.asyncio
async def test_create_user(client):
    # Test creating a new user
    user_data = {
        "username": "newuser",
        "email": "new@example.com",
        "password": "password123",
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
        "password": "superpass",
        "role": UserRole.SUPERUSER.value,
    }
    await client.post("/api/v1/users/", json=superuser_data)
    login_response = await client.post(
        "/api/v1/auth/token", data={"username": "superdup", "password": "superpass"}
    )
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    user_data = {
        "username": "duplicate",
        "email": "dup1@example.com",
        "password": "password123",
    }
    await client.post("/api/v1/users/", json=user_data, headers=headers)

    duplicate_data = {
        "username": "duplicate",
        "email": "dup2@example.com",
        "password": "password456",
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
        "password": "superpass",
        "role": UserRole.SUPERUSER.value,
    }
    await client.post("/api/v1/users/", json=superuser_data)
    login_response = await client.post(
        "/api/v1/auth/token",
        data={"username": "superdupemail", "password": "superpass"},
    )
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    user_data = {
        "username": "emailuser1",
        "email": "same@example.com",
        "password": "password123",
    }
    await client.post("/api/v1/users/", json=user_data, headers=headers)

    duplicate_data = {
        "username": "emailuser2",
        "email": "same@example.com",
        "password": "password456",
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
        "password": "superpass",
        "role": UserRole.SUPERUSER.value,
    }
    await client.post("/api/v1/users/", json=superuser_data)

    # Unauthenticated probe using an existing username.
    existing_probe = await client.post(
        "/api/v1/users/",
        json={
            "username": "enumadmin",
            "email": "attacker@example.com",
            "password": "password123",
        },
    )
    # Unauthenticated probe using a fresh username.
    fresh_probe = await client.post(
        "/api/v1/users/",
        json={
            "username": "doesnotexist",
            "email": "fresh@example.com",
            "password": "password123",
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
        "password": "superpass",
        "role": UserRole.SUPERUSER.value,
    }
    await client.post("/api/v1/users/", json=superuser_data)

    existing_probe = await client.post(
        "/api/v1/users/",
        json={
            "username": "freshname1",
            "email": "emailenum@example.com",
            "password": "password123",
        },
    )
    fresh_probe = await client.post(
        "/api/v1/users/",
        json={
            "username": "freshname2",
            "email": "neveruse@example.com",
            "password": "password123",
        },
    )

    assert existing_probe.status_code == fresh_probe.status_code
    assert existing_probe.json()["detail"] == fresh_probe.json()["detail"]
    assert existing_probe.status_code == 403


@pytest.mark.asyncio
async def test_get_users(client, db):
    # First create a superuser (will be the first user so has permission)
    superuser_data = {
        "username": "superadmin",
        "email": "super@example.com",
        "password": "superpass",
        "role": UserRole.SUPERUSER.value,
    }

    await client.post("/api/v1/users/", json=superuser_data)

    # Login as superuser
    login_response = await client.post(
        "/api/v1/auth/token", data={"username": "superadmin", "password": "superpass"}
    )
    token = login_response.json()["access_token"]

    # Create multiple users using the superuser token
    users_data = [
        {"username": "user1", "email": "user1@example.com", "password": "password1"},
        {"username": "user2", "email": "user2@example.com", "password": "password2"},
        {"username": "user3", "email": "user3@example.com", "password": "password3"},
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
        "password": "superpass",
        "role": UserRole.SUPERUSER.value,
    }

    await client.post("/api/v1/users/", json=superuser_data)

    # Login as superuser
    login_response = await client.post(
        "/api/v1/auth/token", data={"username": "superget", "password": "superpass"}
    )
    token = login_response.json()["access_token"]

    # Create a test user with superuser token
    user_data = {
        "username": "getbyid",
        "email": "getbyid@example.com",
        "password": "password123",
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
        "password": "superpass",
        "role": UserRole.SUPERUSER.value,
    }

    await client.post("/api/v1/users/", json=superuser_data)

    # Login as superuser
    login_response = await client.post(
        "/api/v1/auth/token", data={"username": "superupdate", "password": "superpass"}
    )
    token = login_response.json()["access_token"]

    # Create a test user with superuser token
    user_data = {
        "username": "updateme",
        "email": "update@example.com",
        "password": "password123",
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
        "password": "superpass",
        "role": UserRole.SUPERUSER.value,
    }

    await client.post("/api/v1/users/", json=superuser_data)

    # Login as superuser
    login_response = await client.post(
        "/api/v1/auth/token", data={"username": "superdelete", "password": "superpass"}
    )
    token = login_response.json()["access_token"]

    # Create a test user with superuser token
    user_data = {
        "username": "deleteme",
        "email": "delete@example.com",
        "password": "password123",
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
        "password": "superpass",
        "role": UserRole.SUPERUSER.value,
    }

    await client.post("/api/v1/users/", json=superuser_data)

    # Login as superuser
    login_response = await client.post(
        "/api/v1/auth/token", data={"username": "superaccess", "password": "superpass"}
    )
    super_token = login_response.json()["access_token"]

    # Create two regular users with superuser token
    user1_data = {
        "username": "user1access",
        "email": "user1@example.com",
        "password": "password1",
    }

    user2_data = {
        "username": "user2access",
        "email": "user2@example.com",
        "password": "password2",
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
        "/api/v1/auth/token", data={"username": "user1access", "password": "password1"}
    )
    token = login_response.json()["access_token"]

    # Try to get user2's details
    headers = {"Authorization": f"Bearer {token}"}
    response = await client.get(f"/api/v1/users/{user2_id}", headers=headers)

    # Should be forbidden
    assert response.status_code == 403


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
