from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import bcrypt
import jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.core.config import logger, settings
from app.db.models import User, UserRole

UTC = ZoneInfo("UTC")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    # bcrypt only uses the first 72 bytes and bcrypt>=5 raises ValueError on
    # longer input. Truncate (as bcrypt<5 did implicitly) so an over-long login
    # attempt is a normal mismatch rather than a 500, and hashes created by
    # older bcrypt from >72-byte passwords still verify.
    return bcrypt.checkpw(
        plain_password.encode("utf-8")[:72], hashed_password.encode("utf-8")
    )


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


# Hash verified when the user is unknown or inactive, so every login attempt
# costs one bcrypt check and response time does not reveal which usernames
# exist. Built lazily via get_password_hash so its cost factor tracks the
# real hashing path.
_DUMMY_PASSWORD_HASH: str | None = None


def _get_dummy_password_hash() -> str:
    global _DUMMY_PASSWORD_HASH
    if _DUMMY_PASSWORD_HASH is None:
        _DUMMY_PASSWORD_HASH = get_password_hash("fastpki-dummy-password")
    return _DUMMY_PASSWORD_HASH


class UserService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_by_id(self, user_id: int) -> User | None:
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_user_by_username(self, username: str) -> User | None:
        result = await self.db.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    async def get_user_by_email(self, email: str) -> User | None:
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER,
        organization_id: int | None = None,
        can_create_ca: bool = False,
        can_create_cert: bool = False,
        can_revoke_cert: bool = False,
        can_export_private_key: bool = False,
        can_delete_ca: bool = False,
    ) -> User:
        logger.debug("Creating user: %s, role: %s", username, role)
        hashed_password = get_password_hash(password)

        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            role=role,
            organization_id=organization_id,
            can_create_ca=can_create_ca,
            can_create_cert=can_create_cert,
            can_revoke_cert=can_revoke_cert,
            can_export_private_key=can_export_private_key,
            can_delete_ca=can_delete_ca,
        )

        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)

        logger.debug("User created successfully: %s with ID %s", username, user.id)
        return user

    async def update_user(
        self,
        user_id: int,
        email: str | None = None,
        password: str | None = None,
        role: UserRole | None = None,
        is_active: bool | None = None,
        organization_id: int | None = None,
        can_create_ca: bool | None = None,
        can_create_cert: bool | None = None,
        can_revoke_cert: bool | None = None,
        can_export_private_key: bool | None = None,
        can_delete_ca: bool | None = None,
    ) -> User | None:
        user = await self.get_user_by_id(user_id)

        if not user:
            return None

        if email is not None:
            user.email = email

        if password is not None:
            user.hashed_password = get_password_hash(password)

        if role is not None:
            user.role = role

        if is_active is not None:
            user.is_active = is_active

        if organization_id is not None:
            user.organization_id = organization_id

        if can_create_ca is not None:
            user.can_create_ca = can_create_ca

        if can_create_cert is not None:
            user.can_create_cert = can_create_cert

        if can_revoke_cert is not None:
            user.can_revoke_cert = can_revoke_cert

        if can_export_private_key is not None:
            user.can_export_private_key = can_export_private_key

        if can_delete_ca is not None:
            user.can_delete_ca = can_delete_ca

        should_invalidate_tokens = password is not None or (
            is_active is not None and not is_active
        )

        if should_invalidate_tokens:
            user.tokens_invalidated_at = datetime.now(UTC)
            from app.services.token import TokenService

            token_service = TokenService(self.db)
            await token_service.revoke_all_user_refresh_tokens(
                user_id=user_id, commit=False
            )

        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def delete_user(self, user_id: int) -> bool:
        user = await self.get_user_by_id(user_id)

        if not user:
            return False

        await self.db.delete(user)
        await self.db.commit()

        return True

    async def authenticate_user(self, username: str, password: str) -> User | None:
        logger.debug("Authenticating user: %s", username)
        user = await self.get_user_by_username(username)

        if not user or not user.is_active:
            # Burn the same bcrypt cost as a real check to avoid a timing
            # oracle on username existence / account state.
            verify_password(password, _get_dummy_password_hash())
            logger.debug("User not found or inactive: %s", username)
            return None

        if not verify_password(password, user.hashed_password):
            logger.debug("Invalid password for user: %s", username)
            return None

        logger.debug("Authentication successful: %s", username)
        return user

    def create_access_token(
        self, data: dict[str, Any], expires_delta: timedelta | None = None
    ) -> str:
        import uuid

        to_encode = data.copy()

        now = datetime.now(UTC)
        if expires_delta:
            expire = now + expires_delta
        else:
            expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update(
            {
                "exp": int(expire.timestamp()),
                "iat": int(now.timestamp()),
                "jti": str(uuid.uuid4()),
            }
        )

        encoded_jwt: str = jwt.encode(
            to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
        )

        return encoded_jwt
