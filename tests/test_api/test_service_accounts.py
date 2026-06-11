import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.deps import (
    get_current_active_admin_user,
    get_current_active_superuser,
    get_current_active_user,
    get_current_user,
)
from app.db.models import User
from app.db.session import get_session
from app.services.organization import OrganizationService
from app.services.service_account import ServiceAccountService
from tests.conftest import TestAuth, create_test_app, get_test_session

PREFIX = "/api/v1/service-accounts"


def _app_for(user: User) -> FastAPI:
    app = create_test_app()
    app.dependency_overrides[get_session] = get_test_session
    app.dependency_overrides[get_current_user] = TestAuth(user)
    app.dependency_overrides[get_current_active_user] = TestAuth(user)
    app.dependency_overrides[get_current_active_superuser] = TestAuth(user)
    app.dependency_overrides[get_current_active_admin_user] = TestAuth(user)
    return app


@pytest_asyncio.fixture
async def admin_sa_client(setup_db, admin_user):
    app = _app_for(admin_user)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_create_service_account(admin_sa_client):
    resp = await admin_sa_client.post(
        f"{PREFIX}/",
        json={"name": "ci", "description": "CI", "can_create_cert": True},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "ci"
    assert body["can_create_cert"] is True
    assert body["can_create_ca"] is False
    assert body["disabled_at"] is None
    # Never leak secrets on the SA resource.
    assert "token" not in body
    assert "digest" not in body


@pytest.mark.asyncio
async def test_list_service_accounts_scoped(admin_sa_client):
    await admin_sa_client.post(f"{PREFIX}/", json={"name": "one"})
    await admin_sa_client.post(f"{PREFIX}/", json={"name": "two"})
    resp = await admin_sa_client.get(f"{PREFIX}/")
    assert resp.status_code == 200
    names = {sa["name"] for sa in resp.json()}
    assert names == {"one", "two"}


@pytest.mark.asyncio
async def test_superuser_lists_all_service_accounts(superuser_client, db, test_org):
    """An org-less superuser must see every service account, not [] (issue #51)."""
    other_org = await OrganizationService(db).create_organization(name="SAListOrg")
    sa_service = ServiceAccountService(db)
    await sa_service.create_service_account(name="alpha", organization_id=test_org.id)
    await sa_service.create_service_account(name="beta", organization_id=other_org.id)

    resp = await superuser_client.get(f"{PREFIX}/")
    assert resp.status_code == 200
    names = {sa["name"] for sa in resp.json()}
    assert names == {"alpha", "beta"}


@pytest.mark.asyncio
async def test_read_cross_org_service_account_is_404(admin_sa_client, db):
    other_org = await OrganizationService(db).create_organization(name="OtherSAOrg")
    other_sa = await ServiceAccountService(db).create_service_account(
        name="foreign", organization_id=other_org.id
    )
    resp = await admin_sa_client.get(f"{PREFIX}/{other_sa.id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_service_account_disable_and_rename(admin_sa_client):
    created = (await admin_sa_client.post(f"{PREFIX}/", json={"name": "edit"})).json()
    resp = await admin_sa_client.patch(
        f"{PREFIX}/{created['id']}",
        json={"name": "renamed", "disabled": True},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "renamed"
    assert body["disabled_at"] is not None


@pytest.mark.asyncio
async def test_delete_service_account(admin_sa_client):
    created = (await admin_sa_client.post(f"{PREFIX}/", json={"name": "gone"})).json()
    resp = await admin_sa_client.delete(f"{PREFIX}/{created['id']}")
    assert resp.status_code == 204
    assert (await admin_sa_client.get(f"{PREFIX}/{created['id']}")).status_code == 404


@pytest.mark.asyncio
async def test_mint_list_revoke_token(admin_sa_client):
    created = (await admin_sa_client.post(f"{PREFIX}/", json={"name": "tok"})).json()
    sa_id = created["id"]

    mint = await admin_sa_client.post(f"{PREFIX}/{sa_id}/tokens", json={"name": "k"})
    assert mint.status_code == 201, mint.text
    minted = mint.json()
    assert minted["token"].startswith("fpki_sa_")
    assert minted["public_id"] in minted["token"]

    listed = await admin_sa_client.get(f"{PREFIX}/{sa_id}/tokens")
    assert listed.status_code == 200
    tokens = listed.json()
    assert len(tokens) == 1
    # Listing must never expose the plaintext or the digest.
    assert "token" not in tokens[0]
    assert "digest" not in tokens[0]

    revoke = await admin_sa_client.delete(f"{PREFIX}/{sa_id}/tokens/{minted['id']}")
    assert revoke.status_code == 204
    assert (await admin_sa_client.get(f"{PREFIX}/{sa_id}/tokens")).json()[0][
        "revoked"
    ] is True


@pytest.mark.asyncio
async def test_policy_set_show_clear(admin_sa_client):
    created = (
        await admin_sa_client.post(
            f"{PREFIX}/", json={"name": "pol", "can_create_cert": True}
        )
    ).json()
    sid = created["id"]

    put = await admin_sa_client.put(
        f"{PREFIX}/{sid}/policy",
        json={
            "cn_patterns": ["*.example.com"],
            "allowed_ca_ids": [1],
            "allowed_certificate_types": ["server"],
            "max_validity_days": 90,
        },
    )
    assert put.status_code == 200, put.text
    assert put.json()["cn_patterns"] == ["*.example.com"]
    assert put.json()["allowed_certificate_types"] == ["server"]

    got = await admin_sa_client.get(f"{PREFIX}/{sid}/policy")
    assert got.status_code == 200
    assert got.json()["max_validity_days"] == 90

    clr = await admin_sa_client.delete(f"{PREFIX}/{sid}/policy")
    assert clr.status_code == 204
    assert (await admin_sa_client.get(f"{PREFIX}/{sid}/policy")).status_code == 404
