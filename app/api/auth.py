from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_active_user, get_current_user_with_payload
from app.core.config import logger, settings
from app.db.models import AuditAction, User
from app.db.session import get_session
from app.schemas.user import RefreshTokenRequest, Token, TokenPayload
from app.services.audit import AuditService
from app.services.token import TokenService
from app.services.user import UserService

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/token", response_model=Token)
@limiter.limit(settings.AUTH_RATE_LIMIT)
async def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),  # noqa: B008
    db: AsyncSession = Depends(get_session, scope="function"),  # noqa: B008
) -> Any:
    logger.debug("Login attempt for username: %s", form_data.username)
    user_service = UserService(db)
    user = await user_service.authenticate_user(form_data.username, form_data.password)

    audit_service = AuditService(db)

    if not user:
        logger.info("Failed login attempt for username: %s", form_data.username)
        await audit_service.log_action(
            action=AuditAction.LOGIN_FAILURE,
            username=form_data.username,
        )
        # Failure audit: commit before raising, otherwise the request unit of
        # work rolls the entry back along with the failed request.
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = user_service.create_access_token(
        data={"sub": user.username, "id": user.id, "role": user.role},
        expires_delta=access_token_expires,
    )

    token_service = TokenService(db)
    refresh_token = await token_service.create_refresh_token(user_id=user.id)

    await audit_service.log_action(
        action=AuditAction.LOGIN_SUCCESS,
        user_id=user.id,
        username=user.username,
        organization_id=user.organization_id,
    )

    logger.info("Successful login for user: %s", user.username)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


@router.post("/refresh", response_model=Token)
async def refresh_access_token(
    body: RefreshTokenRequest,
    db: AsyncSession = Depends(get_session, scope="function"),  # noqa: B008
) -> Any:
    token_service = TokenService(db)
    refresh_token = await token_service.validate_and_revoke_refresh_token(
        body.refresh_token
    )

    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    user_service = UserService(db)
    user = await user_service.get_user_by_id(refresh_token.user_id)
    if not user or not user.is_active:
        # Keep the presented token revoked even though the request fails.
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    new_access_token = user_service.create_access_token(
        data={"sub": user.username, "id": user.id, "role": user.role},
        expires_delta=access_token_expires,
    )
    new_refresh_token = await token_service.create_refresh_token(user_id=user.id)

    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
    }


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    body: RefreshTokenRequest,
    user_and_payload: tuple[User, TokenPayload] = Depends(
        get_current_user_with_payload
    ),
    db: AsyncSession = Depends(get_session, scope="function"),  # noqa: B008
) -> None:
    current_user, token_data = user_and_payload

    token_service = TokenService(db)

    refresh_token = await token_service.validate_refresh_token(body.refresh_token)
    if refresh_token and refresh_token.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Refresh token does not belong to current user",
        )

    if token_data.jti and token_data.exp:
        exp = datetime.fromtimestamp(token_data.exp, tz=ZoneInfo("UTC"))
        await token_service.blocklist_token(jti=token_data.jti, exp=exp)

    await token_service.revoke_refresh_token(body.refresh_token)


@router.post("/invalidate", status_code=status.HTTP_204_NO_CONTENT)
async def invalidate_all_tokens(
    current_user: User = Depends(get_current_active_user),  # noqa: B008
    db: AsyncSession = Depends(get_session, scope="function"),  # noqa: B008
) -> None:
    current_user.tokens_invalidated_at = datetime.now(ZoneInfo("UTC"))
    db.add(current_user)

    token_service = TokenService(db)
    await token_service.revoke_all_user_refresh_tokens(user_id=current_user.id)
