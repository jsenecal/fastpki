from zoneinfo import ZoneInfo

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import User, UserRole
from app.db.session import get_session
from app.schemas.user import TokenPayload
from app.services.principal import Principal

UTC = ZoneInfo("UTC")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/token")


async def _validate_token(db: AsyncSession, token: str) -> tuple[User, TokenPayload]:
    """Decode and validate an access token, returning user and payload."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
    except (InvalidTokenError, ValidationError) as err:
        raise credentials_exception from err

    if token_data.jti:
        from app.services.token import TokenService

        token_service = TokenService(db)
        if await token_service.is_token_blocklisted(token_data.jti):
            raise credentials_exception

    from app.services.user import UserService

    user_service = UserService(db)
    user = await user_service.get_user_by_id(token_data.id)

    if user is None:
        raise credentials_exception

    if user.tokens_invalidated_at:
        if token_data.iat is None:
            raise credentials_exception
        invalidated_at = user.tokens_invalidated_at
        if invalidated_at.tzinfo is None:
            invalidated_at = invalidated_at.replace(tzinfo=UTC)
        if token_data.iat < invalidated_at.timestamp():
            raise credentials_exception

    return user, token_data


async def get_current_user(
    db: AsyncSession = Depends(get_session),  # noqa: B008
    token: str = Depends(oauth2_scheme),
) -> User:
    """Validate access token and return current user."""
    user, _payload = await _validate_token(db, token)
    return user


async def get_current_user_with_payload(
    db: AsyncSession = Depends(get_session),  # noqa: B008
    token: str = Depends(oauth2_scheme),
) -> tuple[User, TokenPayload]:
    """Validate access token and return current user with parsed payload."""
    return await _validate_token(db, token)


async def get_current_active_user(
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_current_principal(
    db: AsyncSession = Depends(get_session),  # noqa: B008
    token: str = Depends(oauth2_scheme),
) -> Principal:
    """Resolve a bearer token to a Principal — a user or a service account.

    Service-account tokens (prefixed `fpki_sa_`) resolve to a service principal;
    anything else falls through to the existing user-JWT path unchanged.
    """
    from app.services.service_account import TOKEN_PREFIX, ServiceAccountService

    if token.startswith(TOKEN_PREFIX):
        sa = await ServiceAccountService(db).resolve_token(token)
        if sa is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return Principal.from_service_account(sa)

    user, _payload = await _validate_token(db, token)
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return Principal.from_user(user)


async def get_current_active_superuser(
    current_user: User = Depends(get_current_active_user),  # noqa: B008
) -> User:
    """Get current active superuser."""
    if current_user.role != UserRole.SUPERUSER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have sufficient privileges",
        )
    return current_user


async def get_current_active_admin_user(
    current_user: User = Depends(get_current_active_user),  # noqa: B008
) -> User:
    """Get current active user with admin privileges (ADMIN or SUPERUSER)."""
    if current_user.role not in [UserRole.ADMIN, UserRole.SUPERUSER]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have sufficient privileges",
        )
    return current_user
