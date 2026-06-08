from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.services.service_account import TOKEN_PREFIX, ServiceAccountService

UTC = ZoneInfo("UTC")


@pytest.fixture
async def sa(db, test_org, admin_user):
    service = ServiceAccountService(db)
    return await service.create_service_account(
        name="tok-sa",
        organization_id=test_org.id,
        created_by_user_id=admin_user.id,
        can_create_cert=True,
    )


@pytest.mark.asyncio
async def test_mint_token_returns_plaintext_once_and_stores_digest(db, sa):
    service = ServiceAccountService(db)

    token, plaintext = await service.mint_token(sa.id, name="ci")

    assert plaintext.startswith(TOKEN_PREFIX)
    assert "." in plaintext
    # The stored record must not contain the plaintext secret anywhere.
    assert token.digest not in plaintext
    assert plaintext.split(".", 1)[1] not in token.digest
    assert token.public_id in plaintext
    assert token.name == "ci"
    assert token.revoked is False


@pytest.mark.asyncio
async def test_resolve_token_returns_service_account(db, sa):
    service = ServiceAccountService(db)
    _token, plaintext = await service.mint_token(sa.id)

    resolved = await service.resolve_token(plaintext)

    assert resolved is not None
    assert resolved.id == sa.id


@pytest.mark.asyncio
async def test_resolve_token_updates_last_used_at(db, sa):
    service = ServiceAccountService(db)
    token, plaintext = await service.mint_token(sa.id)
    assert token.last_used_at is None

    await service.resolve_token(plaintext)

    refreshed = await service.get_token_by_id(token.id)
    assert refreshed is not None
    assert refreshed.last_used_at is not None


@pytest.mark.asyncio
async def test_resolve_token_wrong_secret_rejected(db, sa):
    service = ServiceAccountService(db)
    token, _plaintext = await service.mint_token(sa.id)

    forged = f"{TOKEN_PREFIX}{token.public_id}.not-the-real-secret"
    assert await service.resolve_token(forged) is None


@pytest.mark.asyncio
async def test_resolve_token_malformed_rejected(db, sa):
    service = ServiceAccountService(db)
    assert await service.resolve_token("not-a-fastpki-token") is None
    assert await service.resolve_token(f"{TOKEN_PREFIX}nodot") is None


@pytest.mark.asyncio
async def test_resolve_token_revoked_rejected(db, sa):
    service = ServiceAccountService(db)
    token, plaintext = await service.mint_token(sa.id)

    await service.revoke_token(token.id)

    assert await service.resolve_token(plaintext) is None


@pytest.mark.asyncio
async def test_resolve_token_expired_rejected(db, sa):
    service = ServiceAccountService(db)
    past = datetime.now(UTC) - timedelta(minutes=1)
    _token, plaintext = await service.mint_token(sa.id, expires_at=past)

    assert await service.resolve_token(plaintext) is None


@pytest.mark.asyncio
async def test_resolve_token_disabled_service_account_rejected(db, sa):
    service = ServiceAccountService(db)
    _token, plaintext = await service.mint_token(sa.id)

    await service.set_disabled(sa.id, disabled=True)

    assert await service.resolve_token(plaintext) is None


@pytest.mark.asyncio
async def test_list_and_revoke_tokens(db, sa):
    service = ServiceAccountService(db)
    t1, _ = await service.mint_token(sa.id, name="one")
    t2, _ = await service.mint_token(sa.id, name="two")

    tokens = await service.list_tokens(sa.id)
    assert {t.id for t in tokens} == {t1.id, t2.id}

    await service.revoke_token(t1.id)
    refreshed = await service.get_token_by_id(t1.id)
    assert refreshed is not None
    assert refreshed.revoked is True
