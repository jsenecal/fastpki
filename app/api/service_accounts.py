from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_active_admin_user, get_current_active_user
from app.db.models import (
    AuditAction,
    ServiceAccount,
    ServiceAccountToken,
    User,
    UserRole,
)
from app.db.session import get_session
from app.schemas.service_account import (
    IssuancePolicyResponse,
    IssuancePolicyUpsert,
    ServiceAccountCreate,
    ServiceAccountResponse,
    ServiceAccountTokenCreate,
    ServiceAccountTokenCreateResponse,
    ServiceAccountTokenResponse,
    ServiceAccountUpdate,
)
from app.services.audit import AuditService
from app.services.exceptions import AlreadyExistsError, NotFoundError
from app.services.issuance_policy import IssuancePolicyService
from app.services.service_account import ServiceAccountService

router = APIRouter()

_NOT_FOUND = "Service account not found"


async def _get_in_scope(
    service: ServiceAccountService, sa_id: int, current_user: User
) -> ServiceAccount:
    """Load a service account the caller is allowed to see, else 404.

    Org isolation is enforced by treating out-of-org accounts as not-found so
    existence isn't leaked across organizations.
    """
    sa = await service.get_service_account_by_id(sa_id)
    if sa is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)
    if (
        current_user.role != UserRole.SUPERUSER
        and sa.organization_id != current_user.organization_id
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)
    return sa


@router.post(
    "/", response_model=ServiceAccountResponse, status_code=status.HTTP_201_CREATED
)
async def create_service_account(
    sa_in: ServiceAccountCreate,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    current_user: User = Depends(get_current_active_admin_user),  # noqa: B008
) -> ServiceAccount:
    """Create a service account in the caller's organization (admin only)."""
    if current_user.role == UserRole.SUPERUSER:
        organization_id = sa_in.organization_id or current_user.organization_id
    else:
        organization_id = current_user.organization_id
    if organization_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="organization_id is required",
        )

    service = ServiceAccountService(db)
    try:
        sa = await service.create_service_account(
            name=sa_in.name,
            organization_id=organization_id,
            created_by_user_id=current_user.id,
            description=sa_in.description,
            can_create_ca=sa_in.can_create_ca,
            can_create_cert=sa_in.can_create_cert,
            can_revoke_cert=sa_in.can_revoke_cert,
            can_export_private_key=sa_in.can_export_private_key,
            can_delete_ca=sa_in.can_delete_ca,
        )
    except AlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e

    await AuditService(db).log_action(
        action=AuditAction.SERVICE_ACCOUNT_CREATE,
        user_id=current_user.id,
        username=current_user.username,
        organization_id=organization_id,
        resource_type="service_account",
        resource_id=sa.id,
        detail=f"Created service account '{sa.name}'",
    )
    return sa


@router.get("/", response_model=list[ServiceAccountResponse])
async def list_service_accounts(
    db: AsyncSession = Depends(get_session),  # noqa: B008
    current_user: User = Depends(get_current_active_user),  # noqa: B008
) -> list[ServiceAccount]:
    """List service accounts in the caller's organization."""
    if current_user.organization_id is None and current_user.role != UserRole.SUPERUSER:
        return []
    service = ServiceAccountService(db)
    # A superuser without an org sees nothing here; org-scoped listing only.
    org_id = current_user.organization_id
    if org_id is None:
        return []
    return await service.list_service_accounts(organization_id=org_id)


@router.get("/{sa_id}", response_model=ServiceAccountResponse)
async def read_service_account(
    sa_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    current_user: User = Depends(get_current_active_user),  # noqa: B008
) -> ServiceAccount:
    """Get a service account by ID (org-scoped)."""
    service = ServiceAccountService(db)
    return await _get_in_scope(service, sa_id, current_user)


@router.patch("/{sa_id}", response_model=ServiceAccountResponse)
async def update_service_account(
    sa_id: int,
    sa_in: ServiceAccountUpdate,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    current_user: User = Depends(get_current_active_admin_user),  # noqa: B008
) -> ServiceAccount:
    """Update a service account (admin only)."""
    service = ServiceAccountService(db)
    await _get_in_scope(service, sa_id, current_user)
    try:
        sa = await service.update_service_account(
            sa_id,
            name=sa_in.name,
            description=sa_in.description,
            disabled=sa_in.disabled,
            can_create_ca=sa_in.can_create_ca,
            can_create_cert=sa_in.can_create_cert,
            can_revoke_cert=sa_in.can_revoke_cert,
            can_export_private_key=sa_in.can_export_private_key,
            can_delete_ca=sa_in.can_delete_ca,
        )
    except AlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e

    await AuditService(db).log_action(
        action=AuditAction.SERVICE_ACCOUNT_UPDATE,
        user_id=current_user.id,
        username=current_user.username,
        organization_id=sa.organization_id,
        resource_type="service_account",
        resource_id=sa.id,
        detail=f"Updated service account '{sa.name}'",
    )
    return sa


@router.delete("/{sa_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_service_account(
    sa_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    current_user: User = Depends(get_current_active_admin_user),  # noqa: B008
) -> None:
    """Delete a service account and its tokens (admin only)."""
    service = ServiceAccountService(db)
    sa = await _get_in_scope(service, sa_id, current_user)
    await service.delete_service_account(sa_id)
    await AuditService(db).log_action(
        action=AuditAction.SERVICE_ACCOUNT_DELETE,
        user_id=current_user.id,
        username=current_user.username,
        organization_id=sa.organization_id,
        resource_type="service_account",
        resource_id=sa_id,
        detail=f"Deleted service account '{sa.name}'",
    )


@router.post(
    "/{sa_id}/tokens",
    response_model=ServiceAccountTokenCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_service_account_token(
    sa_id: int,
    token_in: ServiceAccountTokenCreate,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    current_user: User = Depends(get_current_active_admin_user),  # noqa: B008
) -> ServiceAccountTokenCreateResponse:
    """Mint a token (admin only). The plaintext is returned only here."""
    service = ServiceAccountService(db)
    sa = await _get_in_scope(service, sa_id, current_user)
    token, plaintext = await service.mint_token(
        sa_id, name=token_in.name, expires_at=token_in.expires_at
    )
    await AuditService(db).log_action(
        action=AuditAction.SERVICE_ACCOUNT_TOKEN_CREATE,
        user_id=current_user.id,
        username=current_user.username,
        organization_id=sa.organization_id,
        resource_type="service_account",
        resource_id=sa_id,
        detail=f"Minted token for service account '{sa.name}'",
    )
    return ServiceAccountTokenCreateResponse(
        **ServiceAccountTokenResponse.model_validate(token).model_dump(),
        token=plaintext,
    )


@router.get("/{sa_id}/tokens", response_model=list[ServiceAccountTokenResponse])
async def list_service_account_tokens(
    sa_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    current_user: User = Depends(get_current_active_user),  # noqa: B008
) -> list[ServiceAccountToken]:
    """List token metadata for a service account (org-scoped)."""
    service = ServiceAccountService(db)
    await _get_in_scope(service, sa_id, current_user)
    return await service.list_tokens(sa_id)


@router.delete("/{sa_id}/tokens/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_service_account_token(
    sa_id: int,
    token_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    current_user: User = Depends(get_current_active_admin_user),  # noqa: B008
) -> None:
    """Revoke a service account token (admin only)."""
    service = ServiceAccountService(db)
    sa = await _get_in_scope(service, sa_id, current_user)
    token = await service.get_token_by_id(token_id)
    if token is None or token.service_account_id != sa_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Token not found"
        )
    try:
        await service.revoke_token(token_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    await AuditService(db).log_action(
        action=AuditAction.SERVICE_ACCOUNT_TOKEN_REVOKE,
        user_id=current_user.id,
        username=current_user.username,
        organization_id=sa.organization_id,
        resource_type="service_account",
        resource_id=sa_id,
        detail=f"Revoked token {token_id} for service account '{sa.name}'",
    )


@router.put("/{sa_id}/policy", response_model=IssuancePolicyResponse)
async def set_service_account_policy(
    sa_id: int,
    policy_in: IssuancePolicyUpsert,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    current_user: User = Depends(get_current_active_admin_user),  # noqa: B008
) -> object:
    """Create or replace the issuance policy for a service account (admin only)."""
    service = ServiceAccountService(db)
    sa = await _get_in_scope(service, sa_id, current_user)
    policy = await IssuancePolicyService(db).set_policy(
        sa_id,
        cn_patterns=policy_in.cn_patterns,
        san_dns_patterns=policy_in.san_dns_patterns,
        san_ip_cidrs=policy_in.san_ip_cidrs,
        san_email_domains=policy_in.san_email_domains,
        allowed_ca_ids=policy_in.allowed_ca_ids,
        allowed_certificate_types=policy_in.allowed_certificate_types,
        max_validity_days=policy_in.max_validity_days,
    )
    await AuditService(db).log_action(
        action=AuditAction.SERVICE_ACCOUNT_UPDATE,
        user_id=current_user.id,
        username=current_user.username,
        organization_id=sa.organization_id,
        resource_type="service_account",
        resource_id=sa_id,
        detail=f"Set issuance policy for service account '{sa.name}'",
    )
    return policy


@router.get("/{sa_id}/policy", response_model=IssuancePolicyResponse)
async def read_service_account_policy(
    sa_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    current_user: User = Depends(get_current_active_user),  # noqa: B008
) -> object:
    """Read the issuance policy for a service account (org-scoped)."""
    service = ServiceAccountService(db)
    await _get_in_scope(service, sa_id, current_user)
    policy = await IssuancePolicyService(db).get_policy(sa_id)
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service account has no issuance policy",
        )
    return policy


@router.delete("/{sa_id}/policy", status_code=status.HTTP_204_NO_CONTENT)
async def delete_service_account_policy(
    sa_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    current_user: User = Depends(get_current_active_admin_user),  # noqa: B008
) -> None:
    """Delete the issuance policy, reverting the account to deny-all (admin only)."""
    service = ServiceAccountService(db)
    sa = await _get_in_scope(service, sa_id, current_user)
    await IssuancePolicyService(db).delete_policy(sa_id)
    await AuditService(db).log_action(
        action=AuditAction.SERVICE_ACCOUNT_UPDATE,
        user_id=current_user.id,
        username=current_user.username,
        organization_id=sa.organization_id,
        resource_type="service_account",
        resource_id=sa_id,
        detail=f"Cleared issuance policy for service account '{sa.name}'",
    )
