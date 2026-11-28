from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.core.config import settings
from app.db.models import UserRole
from app.services.user import UserService, get_password_hash, verify_password


@pytest.mark.asyncio
async def test_create_user(db):
    # Test creating a new user
    user_service = UserService(db)
    new_user = await user_service.create_user(
        username="testuser",
        email="test@example.com",
        password="securepassword",
        role=UserRole.USER,
    )

    assert new_user.id is not None
    assert new_user.username == "testuser"
    assert new_user.email == "test@example.com"
    assert new_user.role == UserRole.USER
    assert new_user.is_active is True
    # Password should be hashed
    assert new_user.hashed_password != "securepassword"
    # Verify the password hash works
    assert verify_password("securepassword", new_user.hashed_password)


@pytest.mark.asyncio
async def test_get_user_by_id(db):
    # Create a user first
    user_service = UserService(db)
    created_user = await user_service.create_user(
        username="getbyid",
        email="getbyid@example.com",
        password="password123",
    )

    # Get the user by ID
    user = await user_service.get_user_by_id(created_user.id)

    assert user is not None
    assert user.id == created_user.id
    assert user.username == "getbyid"
    assert user.email == "getbyid@example.com"


@pytest.mark.asyncio
async def test_get_user_by_username(db):
    # Create a user first
    user_service = UserService(db)
    await user_service.create_user(
        username="getbyusername",
        email="getbyusername@example.com",
        password="password123",
    )

    # Get the user by username
    user = await user_service.get_user_by_username("getbyusername")

    assert user is not None
    assert user.username == "getbyusername"
    assert user.email == "getbyusername@example.com"


@pytest.mark.asyncio
async def test_get_user_by_email(db):
    # Create a user first
    user_service = UserService(db)
    await user_service.create_user(
        username="getbyemail",
        email="getbyemail@example.com",
        password="password123",
    )

    # Get the user by email
    user = await user_service.get_user_by_email("getbyemail@example.com")

    assert user is not None
    assert user.username == "getbyemail"
    assert user.email == "getbyemail@example.com"


@pytest.mark.asyncio
async def test_password_hashing():
    # Test password hashing functions
    password = "testpassword"
    hashed = get_password_hash(password)

    # Hashed password should be different from original
    assert hashed != password

    # Verify should return True for correct password
    assert verify_password(password, hashed) is True

    # Verify should return False for incorrect password
    assert verify_password("wrongpassword", hashed) is False


@pytest.mark.asyncio
async def test_create_access_token():
    # Test JWT token generation
    user_service = UserService(None)  # No DB needed for token generation

    username = "tokenuser"
    user_id = 123
    role = UserRole.ADMIN

    # Create token with 30-minute expiry
    token = user_service.create_access_token(
        data={"sub": username, "id": user_id, "role": role},
        expires_delta=timedelta(minutes=30),
    )

    # Verify the token contains the right data
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])

    assert payload["sub"] == username
    assert payload["id"] == user_id
    assert payload["role"] == role

    # Verify expiration time is set correctly (approximately)
    expiry_time = datetime.fromtimestamp(payload["exp"], UTC)
    time_diff = expiry_time - datetime.now(UTC)

    # Should be close to 30 minutes (allow 5 seconds tolerance)
    assert (
        timedelta(minutes=29, seconds=55) < time_diff < timedelta(minutes=30, seconds=5)
    )


@pytest.mark.asyncio
async def test_authenticate_user(db):
    # Create a test user
    user_service = UserService(db)
    await user_service.create_user(
        username="authuser",
        email="authuser@example.com",
        password="correctpassword",
    )

    # Test successful authentication
    user = await user_service.authenticate_user("authuser", "correctpassword")
    assert user is not None
    assert user.username == "authuser"

    # Test failed authentication with wrong password
    user = await user_service.authenticate_user("authuser", "wrongpassword")
    assert user is None

    # Test failed authentication with non-existent username
    user = await user_service.authenticate_user("nonexistentuser", "anypassword")
    assert user is None


@pytest.mark.asyncio
async def test_inactive_user_authentication(db):
    # Create a test user that is inactive
    user_service = UserService(db)
    created_user = await user_service.create_user(
        username="inactiveuser",
        email="inactive@example.com",
        password="password123",
    )

    # Set user to inactive
    created_user.is_active = False
    db.add(created_user)
    await db.commit()
    await db.refresh(created_user)

    # Authentication should fail for inactive users
    user = await user_service.authenticate_user("inactiveuser", "password123")
    assert user is None


@pytest.mark.asyncio
async def test_update_user(db):
    # Create a test user
    user_service = UserService(db)
    created_user = await user_service.create_user(
        username="updateuser",
        email="update@example.com",
        password="password123",
    )

    # Update the user
    updated_user = await user_service.update_user(
        created_user.id,
        email="updated@example.com",
        role=UserRole.ADMIN,
        is_active=False,
    )

    assert updated_user.id == created_user.id
    assert updated_user.username == "updateuser"  # Username remains unchanged
    assert updated_user.email == "updated@example.com"
    assert updated_user.role == UserRole.ADMIN
    assert updated_user.is_active is False


@pytest.mark.asyncio
async def test_delete_user(db):
    # Create a test user
    user_service = UserService(db)
    created_user = await user_service.create_user(
        username="deleteuser",
        email="delete@example.com",
        password="password123",
    )

    # Delete the user
    result = await user_service.delete_user(created_user.id)
    assert result is True

    # Verify user is deleted
    user = await user_service.get_user_by_id(created_user.id)
    assert user is None


class TestAuthenticateTimingOracle:
    """authenticate_user must run exactly one password verification on every
    path so response time does not reveal whether a username exists (#70)."""

    @pytest.fixture
    def verify_spy(self, monkeypatch):
        import app.services.user as user_module

        calls: list[tuple[str, str]] = []
        real_verify = user_module.verify_password

        def spy(plain_password: str, hashed_password: str) -> bool:
            calls.append((plain_password, hashed_password))
            return real_verify(plain_password, hashed_password)

        monkeypatch.setattr(user_module, "verify_password", spy)
        return calls

    async def _make_user(self, db, *, active: bool = True):
        user_service = UserService(db)
        user = await user_service.create_user(
            username="timinguser",
            email="timing@example.com",
            password="correctpassword",
        )
        if not active:
            user.is_active = False
            db.add(user)
            await db.commit()
            await db.refresh(user)
        return user_service, user

    async def test_unknown_user_runs_verify_once(self, db, verify_spy):
        user_service = UserService(db)
        result = await user_service.authenticate_user("nosuchuser", "anypassword")
        assert result is None
        assert len(verify_spy) == 1
        assert verify_spy[0][0] == "anypassword"

    async def test_inactive_user_runs_verify_once(self, db, verify_spy):
        user_service, user = await self._make_user(db, active=False)
        verify_spy.clear()
        result = await user_service.authenticate_user("timinguser", "correctpassword")
        assert result is None
        assert len(verify_spy) == 1
        # Must not reveal the real hash outcome for inactive accounts.
        assert verify_spy[0][1] != user.hashed_password

    async def test_wrong_password_runs_verify_once(self, db, verify_spy):
        user_service, user = await self._make_user(db)
        verify_spy.clear()
        result = await user_service.authenticate_user("timinguser", "wrongpassword")
        assert result is None
        assert verify_spy == [("wrongpassword", user.hashed_password)]

    async def test_correct_password_runs_verify_once(self, db, verify_spy):
        user_service, user = await self._make_user(db)
        verify_spy.clear()
        result = await user_service.authenticate_user("timinguser", "correctpassword")
        assert result is not None
        assert result.id == user.id
        assert verify_spy == [("correctpassword", user.hashed_password)]

    def test_dummy_hash_is_cached_and_uses_real_hasher(self, monkeypatch):
        import app.services.user as user_module

        monkeypatch.setattr(user_module, "_DUMMY_PASSWORD_HASH", None)
        hash_calls: list[str] = []
        real_hash = user_module.get_password_hash

        def spy_hash(password: str) -> str:
            hash_calls.append(password)
            return real_hash(password)

        monkeypatch.setattr(user_module, "get_password_hash", spy_hash)
        first = user_module._get_dummy_password_hash()
        second = user_module._get_dummy_password_hash()
        assert first == second
        assert len(hash_calls) == 1
        assert first.startswith("$2")
