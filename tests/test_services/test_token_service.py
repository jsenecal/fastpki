import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import BlocklistedToken, RefreshToken, UserRole
from app.services.token import TokenService
from app.services.user import UserService


@pytest_asyncio.fixture
async def test_user(db: AsyncSession):
    user_service = UserService(db)
    return await user_service.create_user(
        username="tokenuser",
        email="tokenuser@example.com",
        password="password123",
        role=UserRole.USER,
    )


@pytest_asyncio.fixture
async def test_user2(db: AsyncSession):
    user_service = UserService(db)
    return await user_service.create_user(
        username="tokenuser2",
        email="tokenuser2@example.com",
        password="password123",
        role=UserRole.USER,
    )


async def test_create_blocklisted_token(db: AsyncSession):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    token = BlocklistedToken(
        jti="test-jti-123",
        exp=datetime.now(ZoneInfo("UTC")),
    )
    db.add(token)
    await db.commit()
    await db.refresh(token)

    assert token.id is not None
    assert token.jti == "test-jti-123"


async def test_create_refresh_token(db: AsyncSession, test_user):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    token = RefreshToken(
        token_hash="abc123hash",
        user_id=test_user.id,
        expires_at=datetime.now(ZoneInfo("UTC")),
    )
    db.add(token)
    await db.commit()
    await db.refresh(token)

    assert token.id is not None
    assert token.token_hash == "abc123hash"
    assert token.revoked is False
    assert token.created_at is not None


async def test_user_tokens_invalidated_at_default_none(db: AsyncSession):
    from app.db.models import UserRole
    from app.services.user import UserService

    user_service = UserService(db)
    user = await user_service.create_user(
        username="tokentest",
        email="tokentest@example.com",
        password="password123",
        role=UserRole.USER,
    )

    assert user.tokens_invalidated_at is None


async def test_blocklist_token(db: AsyncSession):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    token_service = TokenService(db)
    await token_service.blocklist_token(
        jti="blocklist-jti-1", exp=datetime.now(ZoneInfo("UTC"))
    )
    assert await token_service.is_token_blocklisted("blocklist-jti-1") is True
    assert await token_service.is_token_blocklisted("nonexistent-jti") is False


async def test_blocklist_token_is_idempotent(db: AsyncSession):
    """A duplicate JTI is ignored without poisoning the caller's transaction."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from sqlalchemy import func
    from sqlmodel import select

    token_service = TokenService(db)
    exp = datetime.now(ZoneInfo("UTC"))
    await token_service.blocklist_token(jti="dup-jti", exp=exp)
    await token_service.blocklist_token(jti="dup-jti", exp=exp)

    count = await db.execute(
        select(func.count())
        .select_from(BlocklistedToken)
        .where(BlocklistedToken.jti == "dup-jti")
    )
    assert count.scalar_one() == 1
    # The session is still usable after the conflict.
    await db.commit()
    assert await token_service.is_token_blocklisted("dup-jti") is True


async def test_create_and_validate_refresh_token(db: AsyncSession, test_user):
    token_service = TokenService(db)
    refresh_token = await token_service.create_refresh_token(user_id=test_user.id)
    assert isinstance(refresh_token, str)
    assert len(refresh_token) > 0
    valid_token = await token_service.validate_refresh_token(refresh_token)
    assert valid_token is not None
    assert valid_token.user_id == test_user.id
    assert valid_token.revoked is False


async def test_revoke_refresh_token(db: AsyncSession, test_user):
    token_service = TokenService(db)
    refresh_token = await token_service.create_refresh_token(user_id=test_user.id)
    await token_service.revoke_refresh_token(refresh_token)
    result = await token_service.validate_refresh_token(refresh_token)
    assert result is None


async def test_revoke_all_user_refresh_tokens(db: AsyncSession, test_user, test_user2):
    token_service = TokenService(db)
    token1 = await token_service.create_refresh_token(user_id=test_user.id)
    token2 = await token_service.create_refresh_token(user_id=test_user.id)
    token3 = await token_service.create_refresh_token(user_id=test_user2.id)
    await token_service.revoke_all_user_refresh_tokens(user_id=test_user.id)
    assert await token_service.validate_refresh_token(token1) is None
    assert await token_service.validate_refresh_token(token2) is None
    assert await token_service.validate_refresh_token(token3) is not None


async def test_cleanup_expired_tokens(db: AsyncSession, test_user):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from sqlmodel import select

    utc = ZoneInfo("UTC")
    token_service = TokenService(db)
    now = datetime.now(utc)

    expired_bl = BlocklistedToken(jti="expired-jti", exp=now - timedelta(hours=1))
    db.add(expired_bl)
    valid_bl = BlocklistedToken(jti="valid-jti", exp=now + timedelta(hours=1))
    db.add(valid_bl)
    expired_rt = RefreshToken(
        token_hash="expired-rt-hash",
        user_id=test_user.id,
        expires_at=now - timedelta(hours=1),
    )
    db.add(expired_rt)
    valid_rt = RefreshToken(
        token_hash="valid-rt-hash",
        user_id=test_user.id,
        expires_at=now + timedelta(hours=1),
    )
    db.add(valid_rt)
    await db.commit()

    deleted_count = await token_service.cleanup_expired_tokens()
    assert deleted_count == 2

    result = await db.execute(
        select(BlocklistedToken).where(BlocklistedToken.jti == "expired-jti")
    )
    assert result.scalar_one_or_none() is None
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == "expired-rt-hash")
    )
    assert result.scalar_one_or_none() is None
    result = await db.execute(
        select(BlocklistedToken).where(BlocklistedToken.jti == "valid-jti")
    )
    assert result.scalar_one_or_none() is not None
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == "valid-rt-hash")
    )
    assert result.scalar_one_or_none() is not None


async def test_password_change_invalidates_tokens(db: AsyncSession):
    user_service = UserService(db)
    user = await user_service.create_user(
        username="pwchangeuser",
        email="pwchange@example.com",
        password="password123",
        role=UserRole.USER,
    )

    token_service = TokenService(db)
    refresh = await token_service.create_refresh_token(user_id=user.id)

    assert user.tokens_invalidated_at is None

    updated_user = await user_service.update_user(user.id, password="newpassword456")

    assert updated_user is not None
    assert updated_user.tokens_invalidated_at is not None

    result = await token_service.validate_refresh_token(refresh)
    assert result is None


async def test_deactivation_invalidates_tokens(db: AsyncSession):
    user_service = UserService(db)
    user = await user_service.create_user(
        username="deactuser",
        email="deact@example.com",
        password="password123",
        role=UserRole.USER,
    )

    token_service = TokenService(db)
    refresh = await token_service.create_refresh_token(user_id=user.id)

    updated_user = await user_service.update_user(user.id, is_active=False)

    assert updated_user is not None
    assert updated_user.tokens_invalidated_at is not None

    result = await token_service.validate_refresh_token(refresh)
    assert result is None


async def test_token_gc_loop_runs_cleanup():
    """Test the GC loop calls cleanup and handles cancellation."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_token_service = AsyncMock()
    mock_token_service.cleanup_expired_tokens.return_value = 3

    mock_session = AsyncMock()
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = mock_session
    mock_session_cm.__aexit__.return_value = False
    mock_factory = MagicMock(return_value=mock_session_cm)

    with (
        patch("app.main.asyncio.sleep", side_effect=asyncio.CancelledError),
        patch("app.db.session.async_session_factory", mock_factory),
        patch("app.services.token.TokenService", return_value=mock_token_service),
    ):
        import contextlib

        from app.main import token_gc_loop

        with contextlib.suppress(asyncio.CancelledError):
            await token_gc_loop()

    mock_token_service.cleanup_expired_tokens.assert_called_once()
    # Services no longer commit; the GC loop's unit of work must (issue #55).
    mock_session.commit.assert_awaited_once()


async def test_access_token_contains_jti_and_iat(db: AsyncSession):
    import jwt

    from app.core.config import settings
    from app.services.user import UserService

    user_service = UserService(db)
    token = user_service.create_access_token(
        data={"sub": "testuser", "id": 1, "role": "user"}
    )
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])

    assert "jti" in payload
    assert isinstance(payload["jti"], str)
    assert len(payload["jti"]) > 0

    assert "iat" in payload
    assert isinstance(payload["iat"], int)
