import logging

import jwt
import pytest

from app.core.config import logger, settings
from app.db.models import UserRole

# Set logging to debug for tests
logger.setLevel(logging.DEBUG)


@pytest.mark.asyncio
async def test_login_success(client, db):
    # Create a test user first (directly, not via API)
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "Kestrel-Meadow-4821",
    }

    await client.post("/api/v1/users/", json=user_data)

    # Try to login
    # Using form data directly in the request below

    # For OAuth2PasswordRequestForm, use proper formData format
    response = await client.post(
        "/api/v1/auth/token",
        data={"username": "testuser", "password": "Kestrel-Meadow-4821"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

    # Verify token contains expected data
    token = data["access_token"]
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])

    assert payload["sub"] == "testuser"
    assert "id" in payload
    assert payload["role"] == UserRole.USER.value


@pytest.mark.asyncio
async def test_login_invalid_credentials(client, db):
    # Create a test user first
    user_data = {
        "username": "invalidtest",
        "email": "invalid@example.com",
        "password": "Kestrel-Meadow-4821",
    }

    await client.post("/api/v1/users/", json=user_data)

    # Try to login with wrong password
    # Using form data directly in the request below

    response = await client.post(
        "/api/v1/auth/token",
        data={"username": "invalidtest", "password": "wrongpassword"},
    )

    assert response.status_code == 401
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "Incorrect username or password"


@pytest.mark.asyncio
async def test_login_nonexistent_user(client):
    # Try to login with non-existent user (with timestamp to ensure it doesn't exist)
    import time

    unique_id = int(time.time() * 1000)

    login_data = {"username": f"nonexistentuser_{unique_id}", "password": "anypassword"}

    response = await client.post(
        "/api/v1/auth/token",
        data={"username": login_data["username"], "password": "anypassword"},
    )

    assert response.status_code == 401
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "Incorrect username or password"


@pytest.mark.asyncio
async def test_access_protected_endpoint(client, db):
    # Create a test user
    user_data = {
        "username": "protecteduser",
        "email": "protected@example.com",
        "password": "Kestrel-Meadow-4821",
    }

    await client.post("/api/v1/users/", json=user_data)

    # Login to get a token
    # Using form data directly in the request below

    login_response = await client.post(
        "/api/v1/auth/token",
        data={"username": "protecteduser", "password": "Kestrel-Meadow-4821"},
    )
    token = login_response.json()["access_token"]

    # Access a protected endpoint with token
    headers = {"Authorization": f"Bearer {token}"}
    response = await client.get("/api/v1/users/me", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "protecteduser"
    assert data["email"] == "protected@example.com"


@pytest.mark.asyncio
async def test_access_protected_endpoint_without_token(client):
    # Try to access protected endpoint without a token
    response = await client.get("/api/v1/users/me")

    assert response.status_code == 401
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "Not authenticated"


@pytest.mark.asyncio
async def test_access_protected_endpoint_invalid_token(client):
    # Try to access protected endpoint with invalid token
    headers = {"Authorization": "Bearer invalidtoken"}
    response = await client.get("/api/v1/users/me", headers=headers)

    assert response.status_code == 401
    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_user_can_create_regular_user(client, db):
    # Test to verify a user can create another regular user
    # First create a regular user
    import time

    unique_id = int(time.time() * 1000)

    user_data = {
        "username": f"user_{unique_id}",
        "email": f"user_{unique_id}@example.com",
        "password": "Kestrel-Meadow-4821",
    }

    # Create user
    create_response = await client.post("/api/v1/users/", json=user_data)
    assert create_response.status_code == 201

    # Login as user
    login_response = await client.post(
        "/api/v1/auth/token",
        data={"username": user_data["username"], "password": "Kestrel-Meadow-4821"},
    )

    assert login_response.status_code == 200
    token = login_response.json()["access_token"]

    # Create another regular user
    unique_id_2 = int(time.time() * 1000) + 1
    new_user_data = {
        "username": f"new_user_{unique_id_2}",
        "email": f"new_{unique_id_2}@example.com",
        "password": "Saffron-Harbor-2846",
        "role": UserRole.USER.value,  # Regular user role
    }

    headers = {"Authorization": f"Bearer {token}"}
    response = await client.post("/api/v1/users/", json=new_user_data, headers=headers)

    assert response.status_code == 201
    data = response.json()
    assert data["username"] == new_user_data["username"]
    assert data["role"] == UserRole.USER.value


@pytest.mark.asyncio
async def test_regular_user_cannot_create_admin(client, db):
    # Create a unique regular user - use timestamp to ensure uniqueness
    import time

    unique_id = int(time.time() * 1000)

    user_data = {
        "username": f"regularuser_{unique_id}",
        "email": f"regular_{unique_id}@example.com",
        "password": "Saffron-Harbor-2846",
    }

    await client.post("/api/v1/users/", json=user_data)

    # Login as regular user
    # Using form data directly in the request below

    login_response = await client.post(
        "/api/v1/auth/token",
        data={"username": user_data["username"], "password": "Saffron-Harbor-2846"},
    )
    token = login_response.json()["access_token"]

    # Try to create an admin user
    new_user_data = {
        "username": "attempted_admin",
        "email": "attempted@example.com",
        "password": "Cobalt-Thistle-6194",
        "role": UserRole.ADMIN.value,
    }

    headers = {"Authorization": f"Bearer {token}"}
    response = await client.post("/api/v1/users/", json=new_user_data, headers=headers)

    # Should be forbidden
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_login_returns_refresh_token(client, db):
    user_data = {
        "username": "refreshloginuser",
        "email": "refreshlogin@example.com",
        "password": "Kestrel-Meadow-4821",
    }
    await client.post("/api/v1/users/", json=user_data)

    response = await client.post(
        "/api/v1/auth/token",
        data={"username": "refreshloginuser", "password": "Kestrel-Meadow-4821"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["refresh_token"] is not None
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_refresh_token_endpoint(client, db):
    user_data = {
        "username": "refreshuser",
        "email": "refresh@example.com",
        "password": "Kestrel-Meadow-4821",
    }
    await client.post("/api/v1/users/", json=user_data)

    login_response = await client.post(
        "/api/v1/auth/token",
        data={"username": "refreshuser", "password": "Kestrel-Meadow-4821"},
    )
    refresh_token = login_response.json()["refresh_token"]

    response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["refresh_token"] != refresh_token


@pytest.mark.asyncio
async def test_refresh_token_invalid(client, db):
    response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": "invalid-token"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_refresh_token_reuse_rejected(client, db):
    """Old refresh token should be rejected after rotation."""
    user_data = {
        "username": "reuseuser",
        "email": "reuse@example.com",
        "password": "Kestrel-Meadow-4821",
    }
    await client.post("/api/v1/users/", json=user_data)

    login_response = await client.post(
        "/api/v1/auth/token",
        data={"username": "reuseuser", "password": "Kestrel-Meadow-4821"},
    )
    old_refresh = login_response.json()["refresh_token"]

    await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": old_refresh},
    )

    response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": old_refresh},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_logout_endpoint(client, db):
    user_data = {
        "username": "logoutuser",
        "email": "logout@example.com",
        "password": "Kestrel-Meadow-4821",
    }
    await client.post("/api/v1/users/", json=user_data)

    login_response = await client.post(
        "/api/v1/auth/token",
        data={"username": "logoutuser", "password": "Kestrel-Meadow-4821"},
    )
    tokens = login_response.json()
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    response = await client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": tokens["refresh_token"]},
        headers=headers,
    )

    assert response.status_code == 204

    # Access token should now be rejected
    response = await client.get("/api/v1/users/me", headers=headers)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_invalidate_endpoint(client, db):
    import asyncio

    user_data = {
        "username": "invalidateuser",
        "email": "invalidate@example.com",
        "password": "Kestrel-Meadow-4821",
    }
    await client.post("/api/v1/users/", json=user_data)

    login1 = await client.post(
        "/api/v1/auth/token",
        data={"username": "invalidateuser", "password": "Kestrel-Meadow-4821"},
    )
    login2 = await client.post(
        "/api/v1/auth/token",
        data={"username": "invalidateuser", "password": "Kestrel-Meadow-4821"},
    )

    tokens1 = login1.json()
    tokens2 = login2.json()
    headers1 = {"Authorization": f"Bearer {tokens1['access_token']}"}
    headers2 = {"Authorization": f"Bearer {tokens2['access_token']}"}

    # Wait so invalidation timestamp is strictly after the tokens' iat (seconds precision)
    await asyncio.sleep(1.1)

    response = await client.post("/api/v1/auth/invalidate", headers=headers1)
    assert response.status_code == 204

    response = await client.get("/api/v1/users/me", headers=headers1)
    assert response.status_code == 401

    response = await client.get("/api/v1/users/me", headers=headers2)
    assert response.status_code == 401

    # Wait so new token's iat is after tokens_invalidated_at
    await asyncio.sleep(1.1)

    # But user can still log in again
    login3 = await client.post(
        "/api/v1/auth/token",
        data={"username": "invalidateuser", "password": "Kestrel-Meadow-4821"},
    )
    assert login3.status_code == 200
    headers3 = {"Authorization": f"Bearer {login3.json()['access_token']}"}
    response = await client.get("/api/v1/users/me", headers=headers3)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_refresh_token_with_deactivated_user(client, db):
    """Refresh should fail if the user was deactivated after getting a token."""
    user_data = {
        "username": "deactrefresh",
        "email": "deactrefresh@example.com",
        "password": "Kestrel-Meadow-4821",
    }
    create_resp = await client.post("/api/v1/users/", json=user_data)
    user_id = create_resp.json()["id"]

    await client.post(
        "/api/v1/auth/token",
        data={"username": "deactrefresh", "password": "Kestrel-Meadow-4821"},
    )
    # Deactivate user directly via DB
    from sqlmodel import select

    from app.db.models import User

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one()
    user.is_active = False
    db.add(user)
    await db.commit()

    # Create a new refresh token for this user (simulating one that wasn't revoked)
    from app.services.token import TokenService

    token_service = TokenService(db)
    refresh = await token_service.create_refresh_token(user_id=user_id)

    response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh},
    )
    assert response.status_code == 401
