import pytest

from app.core.config import settings
from app.db.models import UserRole


@pytest.mark.asyncio
async def test_first_user_bootstrap_as_superuser(client):
    """The very first user can be created as superuser without authentication."""
    response = await client.post(
        "/api/v1/users/",
        json={
            "username": "bootstrap_super",
            "email": "bootstrap@example.com",
            "password": "Kestrel-Meadow-4821",
            "role": UserRole.SUPERUSER.value,
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "bootstrap_super"
    assert data["role"] == UserRole.SUPERUSER.value
    assert data["is_active"] is True


@pytest.mark.asyncio
async def test_login_as_bootstrapped_superuser(client):
    """After bootstrapping a superuser, they can log in and get a token."""
    await client.post(
        "/api/v1/users/",
        json={
            "username": "loginsuper",
            "email": "loginsuper@example.com",
            "password": "Kestrel-Meadow-4821",
            "role": UserRole.SUPERUSER.value,
        },
    )

    response = await client.post(
        "/api/v1/auth/token",
        data={"username": "loginsuper", "password": "Kestrel-Meadow-4821"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_authenticated_superuser_creates_another_superuser(client):
    """An authenticated superuser can create another superuser."""
    await client.post(
        "/api/v1/users/",
        json={
            "username": "super1",
            "email": "super1@example.com",
            "password": "Kestrel-Meadow-4821",
            "role": UserRole.SUPERUSER.value,
        },
    )

    login = await client.post(
        "/api/v1/auth/token",
        data={"username": "super1", "password": "Kestrel-Meadow-4821"},
    )
    token = login.json()["access_token"]

    response = await client.post(
        "/api/v1/users/",
        json={
            "username": "super2",
            "email": "super2@example.com",
            "password": "Kestrel-Meadow-4821",
            "role": UserRole.SUPERUSER.value,
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 201
    assert response.json()["role"] == UserRole.SUPERUSER.value


@pytest.mark.asyncio
async def test_unauthenticated_superuser_creation_rejected(client):
    """After users exist, unauthenticated superuser creation is always rejected."""
    await client.post(
        "/api/v1/users/",
        json={
            "username": "existing",
            "email": "existing@example.com",
            "password": "Kestrel-Meadow-4821",
            "role": UserRole.SUPERUSER.value,
        },
    )

    response = await client.post(
        "/api/v1/users/",
        json={
            "username": "sneakysuper",
            "email": "sneaky@example.com",
            "password": "Kestrel-Meadow-4821",
            "role": UserRole.SUPERUSER.value,
        },
    )
    assert response.status_code == 403
    assert (
        response.json()["detail"]
        == "Not enough permissions to create admin or superuser accounts"
    )


@pytest.mark.asyncio
async def test_unauthenticated_user_creation_rejected_by_default(client):
    """After the first user exists, unauthenticated registration is rejected."""
    await client.post(
        "/api/v1/users/",
        json={
            "username": "firstuser",
            "email": "first@example.com",
            "password": "Kestrel-Meadow-4821",
            "role": UserRole.SUPERUSER.value,
        },
    )

    response = await client.post(
        "/api/v1/users/",
        json={
            "username": "seconduser",
            "email": "second@example.com",
            "password": "Kestrel-Meadow-4821",
        },
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Unauthenticated registration is disabled"


@pytest.mark.asyncio
async def test_unauthenticated_user_creation_allowed_when_enabled(client, monkeypatch):
    """When ALLOW_UNAUTHENTICATED_REGISTRATION is True, unauthenticated user creation works."""
    monkeypatch.setattr(settings, "ALLOW_UNAUTHENTICATED_REGISTRATION", True)

    await client.post(
        "/api/v1/users/",
        json={
            "username": "firstuser",
            "email": "first@example.com",
            "password": "Kestrel-Meadow-4821",
            "role": UserRole.SUPERUSER.value,
        },
    )

    response = await client.post(
        "/api/v1/users/",
        json={
            "username": "publicuser",
            "email": "public@example.com",
            "password": "Kestrel-Meadow-4821",
        },
    )
    assert response.status_code == 201
    assert response.json()["role"] == UserRole.USER.value


@pytest.mark.asyncio
async def test_unauthenticated_superuser_rejected_even_when_registration_enabled(
    client, monkeypatch
):
    """Even with open registration, unauthenticated superuser creation is rejected."""
    monkeypatch.setattr(settings, "ALLOW_UNAUTHENTICATED_REGISTRATION", True)

    await client.post(
        "/api/v1/users/",
        json={
            "username": "firstuser",
            "email": "first@example.com",
            "password": "Kestrel-Meadow-4821",
            "role": UserRole.SUPERUSER.value,
        },
    )

    response = await client.post(
        "/api/v1/users/",
        json={
            "username": "sneakysuper",
            "email": "sneaky@example.com",
            "password": "Kestrel-Meadow-4821",
            "role": UserRole.SUPERUSER.value,
        },
    )
    assert response.status_code == 403
    assert (
        response.json()["detail"]
        == "Not enough permissions to create admin or superuser accounts"
    )
