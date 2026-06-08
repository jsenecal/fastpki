from datetime import timedelta

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_principal
from app.db.models import UserRole
from app.services.organization import OrganizationService
from app.services.service_account import ServiceAccountService
from app.services.user import UserService


async def _user_jwt(db: AsyncSession, **kwargs) -> tuple:
    user_service = UserService(db)
    user = await user_service.create_user(
        username=kwargs.pop("username", "princuser"),
        email=kwargs.pop("email", "princuser@example.com"),
        password="password123",
        role=kwargs.pop("role", UserRole.USER),
        **kwargs,
    )
    token = user_service.create_access_token(
        data={"sub": user.username, "id": user.id, "role": user.role},
        expires_delta=timedelta(minutes=30),
    )
    return user, token


@pytest.mark.asyncio
async def test_user_jwt_resolves_to_user_principal(db: AsyncSession):
    user, token = await _user_jwt(db)
    principal = await get_current_principal(db=db, token=token)
    assert principal.kind == "user"
    assert principal.id == user.id


@pytest.mark.asyncio
async def test_service_account_token_resolves_to_service_principal(db: AsyncSession):
    org = await OrganizationService(db).create_organization(name="PrincResOrg")
    sa_service = ServiceAccountService(db)
    sa = await sa_service.create_service_account(
        name="resolver-sa", organization_id=org.id, can_create_cert=True
    )
    _token, plaintext = await sa_service.mint_token(sa.id)

    principal = await get_current_principal(db=db, token=plaintext)
    assert principal.kind == "service_account"
    assert principal.id == sa.id
    assert principal.organization_id == org.id
    assert principal.can_create_cert is True


@pytest.mark.asyncio
async def test_invalid_service_account_token_rejected(db: AsyncSession):
    with pytest.raises(HTTPException) as exc:
        await get_current_principal(db=db, token="fpki_sa_bogus.secret")
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_inactive_user_rejected(db: AsyncSession):
    user, token = await _user_jwt(
        db, username="inactiveprinc", email="inactiveprinc@example.com"
    )
    user.is_active = False
    db.add(user)
    await db.commit()

    with pytest.raises(HTTPException) as exc:
        await get_current_principal(db=db, token=token)
    assert exc.value.status_code == 400
