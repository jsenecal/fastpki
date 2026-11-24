"""API-level tests for the password policy (issue #11)."""

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.db.models import UserRole
from app.db.session import get_session
from app.services.user import UserService
from tests.conftest import get_test_session

STRONG = "correct horse battery staple"


@pytest_asyncio.fixture
async def app_client(setup_db):
    """Client against the real app factory (includes the 422 redaction handler)."""
    from app.main import create_app

    app = create_app()
    app.dependency_overrides[get_session] = get_test_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _msgs(resp) -> str:
    return " ".join(e["msg"] for e in resp.json()["detail"])


async def _create(client, password, username="pwuser", email="pwuser@example.com"):
    return await client.post(
        "/api/v1/users/",
        json={"username": username, "email": email, "password": password},
    )


async def test_create_user_too_short(app_client):
    resp = await _create(app_client, "Xq7!vR2#mZ9")
    assert resp.status_code == 422
    assert "at least 12 characters" in _msgs(resp)


async def test_create_user_too_long(app_client):
    resp = await _create(app_client, "Q" * 73)
    assert resp.status_code == 422
    assert "at most 72 bytes" in _msgs(resp)


async def test_create_user_common_password(app_client):
    resp = await _create(app_client, "Qwerty123456")
    assert resp.status_code == 422
    assert "too common" in _msgs(resp)


async def test_create_user_contains_username(app_client):
    resp = await _create(app_client, "my pwuser password is long")
    assert resp.status_code == 422
    assert "username" in _msgs(resp)


async def test_create_user_valid_password(app_client):
    resp = await _create(app_client, STRONG)
    assert resp.status_code == 201


async def test_validation_error_does_not_echo_password(app_client):
    pw = "my pwuser password is long"
    resp = await _create(app_client, pw)
    assert resp.status_code == 422
    assert pw not in resp.text


async def test_missing_field_error_does_not_echo_password(app_client):
    # A body-level error (missing email) reports the whole body as `input`.
    pw = "a-very-private-password"
    resp = await app_client.post(
        "/api/v1/users/", json={"username": "nomail", "password": pw}
    )
    assert resp.status_code == 422
    assert pw not in resp.text
    assert "[redacted]" in resp.text


async def test_redaction_handles_nested_lists(app_client):
    # A list body yields an error whose input is a list of dicts.
    pw = "a-very-private-password"
    resp = await app_client.post("/api/v1/users/", json=[{"password": pw}])
    assert resp.status_code == 422
    assert pw not in resp.text


async def test_login_with_legacy_short_password_still_works(app_client, db):
    await UserService(db).create_user(
        username="legacy", email="legacy@example.com", password="short1"
    )
    resp = await app_client.post(
        "/api/v1/auth/token", data={"username": "legacy", "password": "short1"}
    )
    assert resp.status_code == 200
    assert "access_token" in resp.json()


async def test_login_with_overlong_password_is_401_not_500(app_client, db):
    await UserService(db).create_user(
        username="longlogin", email="longlogin@example.com", password=STRONG
    )
    resp = await app_client.post(
        "/api/v1/auth/token", data={"username": "longlogin", "password": "x" * 100}
    )
    assert resp.status_code == 401


async def _superuser_headers(db) -> dict[str, str]:
    svc = UserService(db)
    su = await svc.create_user(
        username="pwsuper",
        email="pwsuper@example.com",
        password=STRONG,
        role=UserRole.SUPERUSER,
    )
    token = svc.create_access_token(
        data={"sub": su.username, "id": su.id, "role": su.role}
    )
    return {"Authorization": f"Bearer {token}"}


async def _target(db):
    return await UserService(db).create_user(
        username="pwtarget", email="target.person@example.com", password="old-pass"
    )


async def test_update_password_too_short(app_client, db):
    headers = await _superuser_headers(db)
    target = await _target(db)
    resp = await app_client.patch(
        f"/api/v1/users/{target.id}", json={"password": "short"}, headers=headers
    )
    assert resp.status_code == 422
    assert "at least 12 characters" in _msgs(resp)


async def test_update_password_common(app_client, db):
    headers = await _superuser_headers(db)
    target = await _target(db)
    resp = await app_client.patch(
        f"/api/v1/users/{target.id}",
        json={"password": "1q2w3e4r5t6y"},
        headers=headers,
    )
    assert resp.status_code == 422
    assert "too common" in _msgs(resp)


async def test_update_password_contains_target_username(app_client, db):
    headers = await _superuser_headers(db)
    target = await _target(db)
    pw = "new-PWTARGET-password"
    resp = await app_client.patch(
        f"/api/v1/users/{target.id}", json={"password": pw}, headers=headers
    )
    assert resp.status_code == 422
    assert "username" in _msgs(resp)
    assert pw not in resp.text
    assert resp.json()["detail"][0]["loc"] == ["body", "password"]


async def test_update_password_contains_existing_email_local_part(app_client, db):
    headers = await _superuser_headers(db)
    target = await _target(db)
    resp = await app_client.patch(
        f"/api/v1/users/{target.id}",
        json={"password": "hello target.person 2026"},
        headers=headers,
    )
    assert resp.status_code == 422
    assert "email" in _msgs(resp)


async def test_update_password_valid(app_client, db):
    headers = await _superuser_headers(db)
    target = await _target(db)
    resp = await app_client.patch(
        f"/api/v1/users/{target.id}", json={"password": STRONG}, headers=headers
    )
    assert resp.status_code == 200
    login = await app_client.post(
        "/api/v1/auth/token", data={"username": "pwtarget", "password": STRONG}
    )
    assert login.status_code == 200


async def test_update_without_password_skips_policy(app_client, db):
    headers = await _superuser_headers(db)
    target = await _target(db)
    resp = await app_client.patch(
        f"/api/v1/users/{target.id}",
        json={"can_create_cert": True},
        headers=headers,
    )
    assert resp.status_code == 200
