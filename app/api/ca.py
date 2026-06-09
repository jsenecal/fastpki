from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_principal
from app.db.models import AuditAction, PermissionAction, UserRole
from app.db.session import get_session
from app.schemas.ca import CACreate, CADetailResponse, CAResponse
from app.services.audit import AuditService
from app.services.ca import CAService
from app.services.encryption import EncryptionService
from app.services.exceptions import (
    HasDependentsError,
    NotFoundError,
    PermissionDeniedError,
)
from app.services.permission import PermissionService
from app.services.principal import Principal

router = APIRouter()


@router.post("/", response_model=CADetailResponse, status_code=status.HTTP_201_CREATED)
async def create_ca(
    ca_in: CACreate,
    request: Request,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> CADetailResponse:
    """Create a new Certificate Authority."""
    perm = PermissionService(db)
    if not perm.can_create_in_org(
        principal,
        principal.organization_id,
        PermissionAction.CREATE_CA,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )
    ca_service = CAService(db)
    try:
        base_url = str(request.base_url).rstrip("/")
        ca = await ca_service.create_ca(
            name=ca_in.name,
            subject_dn=ca_in.subject_dn,
            description=ca_in.description,
            key_size=ca_in.key_size,
            valid_days=ca_in.valid_days,
            organization_id=principal.organization_id,
            parent_ca_id=ca_in.parent_ca_id,
            path_length=ca_in.path_length,
            allow_leaf_certs=ca_in.allow_leaf_certs,
            crl_base_url=ca_in.crl_base_url,
            base_url=base_url,
            **principal.creator_fields(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create CA: {e!s}",
        ) from e
    else:
        audit_service = AuditService(db)
        await audit_service.log_action(
            action=AuditAction.CA_CREATE,
            organization_id=principal.organization_id,
            resource_type="ca",
            resource_id=ca.id,
            detail=f"Created CA '{ca.name}'",
            **AuditService.actor_fields(principal),
        )
        ca.private_key = EncryptionService.decrypt_private_key(ca.private_key)
        return ca


@router.get("/", response_model=list[CAResponse])
async def read_cas(
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> list[CAResponse]:
    """Get all Certificate Authorities."""
    ca_service = CAService(db)
    if principal.role == UserRole.SUPERUSER:
        cas = await ca_service.list_cas()
    else:
        cas = await ca_service.list_cas(organization_id=principal.organization_id)
    return cas


@router.get("/{ca_id}", response_model=CAResponse)
async def read_ca(
    ca_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> CAResponse:
    """Get a specific Certificate Authority by ID."""
    perm = PermissionService(db)
    try:
        ca = await perm.check_ca_access(principal, ca_id, PermissionAction.READ)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e
    return ca


@router.get("/{ca_id}/private-key", response_model=CADetailResponse)
async def read_ca_with_private_key(
    ca_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> CADetailResponse:
    """Get a specific Certificate Authority by ID, including private key."""
    perm = PermissionService(db)
    try:
        ca = await perm.check_ca_access(
            principal, ca_id, PermissionAction.EXPORT_PRIVATE_KEY
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e
    audit_service = AuditService(db)
    await audit_service.log_action(
        action=AuditAction.CA_EXPORT_PRIVATE_KEY,
        organization_id=principal.organization_id,
        resource_type="ca",
        resource_id=ca_id,
        **AuditService.actor_fields(principal),
    )
    ca.private_key = EncryptionService.decrypt_private_key(ca.private_key)
    return ca


@router.get("/{ca_id}/chain", response_model=list[CAResponse])
async def read_ca_chain(
    ca_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> list[CAResponse]:
    """Get the certificate chain for a CA, from the CA up to the root."""
    perm = PermissionService(db)
    try:
        await perm.check_ca_access(principal, ca_id, PermissionAction.READ)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e
    ca_service = CAService(db)
    chain = await ca_service.get_ca_chain(ca_id)
    return chain


@router.get("/{ca_id}/children", response_model=list[CAResponse])
async def read_ca_children(
    ca_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> list[CAResponse]:
    """Get direct child CAs of the specified CA."""
    perm = PermissionService(db)
    try:
        await perm.check_ca_access(principal, ca_id, PermissionAction.READ)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e
    ca_service = CAService(db)
    children = await ca_service.get_child_cas(ca_id)
    return children


@router.delete("/{ca_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_ca(
    ca_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> None:
    """Delete a Certificate Authority by ID."""
    perm = PermissionService(db)
    try:
        await perm.check_ca_access(principal, ca_id, PermissionAction.DELETE_CA)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e
    ca_service = CAService(db)
    try:
        await ca_service.delete_ca(ca_id)
    except HasDependentsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    audit_service = AuditService(db)
    await audit_service.log_action(
        action=AuditAction.CA_DELETE,
        organization_id=principal.organization_id,
        resource_type="ca",
        resource_id=ca_id,
        **AuditService.actor_fields(principal),
    )
