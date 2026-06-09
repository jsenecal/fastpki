"""Edge-case coverage for provisioning API routes (#28/#29/#30)."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.deps import (
    get_current_active_admin_user,
    get_current_active_superuser,
    get_current_active_user,
    get_current_principal,
    get_current_user,
)
from app.db.models import User
from app.db.session import get_session
from app.services.user import UserService
from tests.conftest import (
    TestAuth,
    TestPrincipalAuth,
    create_test_app,
    get_test_session,
)

SA = "/api/v1/service-accounts"
CERTS = "/api/v1/certificates"


def _user_app(user: User) -> FastAPI:
    app = create_test_app()
    app.dependency_overrides[get_session] = get_test_session
    app.dependency_overrides[get_current_user] = TestAuth(user)
    app.dependency_overrides[get_current_active_user] = TestAuth(user)
    app.dependency_overrides[get_current_active_superuser] = TestAuth(user)
    app.dependency_overrides[get_current_active_admin_user] = TestAuth(user)
    app.dependency_overrides[get_current_principal] = TestPrincipalAuth(user)
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# --- service account router ---


@pytest.mark.asyncio
async def test_superuser_creates_sa_with_explicit_org(setup_db, superuser, test_org):
    async with _client(_user_app(superuser)) as c:
        resp = await c.post(
            f"{SA}/", json={"name": "su-sa", "organization_id": test_org.id}
        )
    assert resp.status_code == 201, resp.text
    assert resp.json()["organization_id"] == test_org.id


@pytest.mark.asyncio
async def test_superuser_create_sa_without_org_is_400(setup_db, superuser):
    async with _client(_user_app(superuser)) as c:
        resp = await c.post(f"{SA}/", json={"name": "no-org-sa"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_create_duplicate_sa_name_is_400(setup_db, admin_user):
    async with _client(_user_app(admin_user)) as c:
        assert (await c.post(f"{SA}/", json={"name": "dup"})).status_code == 201
        resp = await c.post(f"{SA}/", json={"name": "dup"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_update_to_duplicate_sa_name_is_400(setup_db, admin_user):
    async with _client(_user_app(admin_user)) as c:
        await c.post(f"{SA}/", json={"name": "first"})
        second = (await c.post(f"{SA}/", json={"name": "second"})).json()
        resp = await c.patch(f"{SA}/{second['id']}", json={"name": "first"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_list_sa_empty_for_user_without_org(setup_db, db):
    user = await UserService(db).create_user(
        username="noorg",
        email="noorg@example.com",
        password="password123",
    )
    async with _client(_user_app(user)) as c:
        resp = await c.get(f"{SA}/")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_revoke_unknown_token_is_404(setup_db, admin_user):
    async with _client(_user_app(admin_user)) as c:
        sa = (await c.post(f"{SA}/", json={"name": "tok-sa"})).json()
        resp = await c.delete(f"{SA}/{sa['id']}/tokens/999999")
    assert resp.status_code == 404


# --- certificates router ---


@pytest.mark.asyncio
async def test_sign_csr_unknown_ca_name_is_404(setup_db, superuser):
    async with _client(_user_app(superuser)) as c:
        resp = await c.post(
            f"{CERTS}/sign-csr",
            json={
                "csr": "-----BEGIN CERTIFICATE REQUEST-----\nbogus\n-----END CERTIFICATE REQUEST-----",
                "ca_name": "does-not-exist",
                "certificate_type": "server",
            },
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_read_unknown_certificate_is_404(setup_db, superuser):
    async with _client(_user_app(superuser)) as c:
        resp = await c.get(f"{CERTS}/999999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_revoke_unknown_certificate_is_404(setup_db, superuser):
    async with _client(_user_app(superuser)) as c:
        resp = await c.post(f"{CERTS}/999999/revoke", json={})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_renew_unknown_certificate_is_404(setup_db, superuser):
    async with _client(_user_app(superuser)) as c:
        resp = await c.post(f"{CERTS}/999999/renew", json={})
    assert resp.status_code == 404
