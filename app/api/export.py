from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_principal
from app.db.models import AuditAction, PermissionAction
from app.db.session import get_session
from app.services.audit import AuditService
from app.services.ca import CAService
from app.services.cert import CertificateService
from app.services.encryption import EncryptionService
from app.services.exceptions import NotFoundError, PermissionDeniedError
from app.services.permission import PermissionService
from app.services.principal import Principal

router = APIRouter()


@router.get("/ca/{ca_id}/certificate")
async def export_ca_certificate(
    ca_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> Response:
    """Export a CA certificate in PEM format."""
    perm = PermissionService(db)
    try:
        ca = await perm.check_ca_access(principal, ca_id, PermissionAction.READ)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e

    return Response(
        content=ca.certificate,
        media_type="application/x-pem-file",
        headers={
            "Content-Disposition": (f"attachment; filename=ca_{ca_id}_certificate.pem")
        },
    )


@router.get("/ca/{ca_id}/private-key")
async def export_ca_private_key(
    ca_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> Response:
    """Export a CA private key in PEM format."""
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

    return Response(
        content=EncryptionService.decrypt_private_key(ca.private_key),
        media_type="application/x-pem-file",
        headers={
            "Content-Disposition": (f"attachment; filename=ca_{ca_id}_private_key.pem")
        },
    )


@router.get("/certificate/{cert_id}")
async def export_certificate(
    cert_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> Response:
    """Export a certificate in PEM format."""
    perm = PermissionService(db)
    try:
        cert = await perm.check_cert_access(principal, cert_id, PermissionAction.READ)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e

    return Response(
        content=cert.certificate,
        media_type="application/x-pem-file",
        headers={
            "Content-Disposition": (f"attachment; filename=certificate_{cert_id}.pem")
        },
    )


@router.get("/certificate/{cert_id}/private-key")
async def export_certificate_private_key(
    cert_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> Response:
    """Export a certificate's private key in PEM format."""
    perm = PermissionService(db)
    try:
        cert = await perm.check_cert_access(
            principal, cert_id, PermissionAction.EXPORT_PRIVATE_KEY
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e

    if not cert.private_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Certificate with ID {cert_id} does not have a private key",
        )

    audit_service = AuditService(db)
    await audit_service.log_action(
        action=AuditAction.CERT_EXPORT_PRIVATE_KEY,
        organization_id=principal.organization_id,
        resource_type="certificate",
        resource_id=cert_id,
        **AuditService.actor_fields(principal),
    )

    return Response(
        content=EncryptionService.decrypt_private_key(cert.private_key),
        media_type="application/x-pem-file",
        headers={
            "Content-Disposition": (
                f"attachment; filename=certificate_{cert_id}_private_key.pem"
            )
        },
    )


@router.get("/certificate/{cert_id}/chain")
async def export_certificate_chain(
    cert_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> Response:
    """Export a certificate with its complete certificate chain in PEM format."""
    perm = PermissionService(db)
    try:
        cert = await perm.check_cert_access(principal, cert_id, PermissionAction.READ)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e

    cert_service = CertificateService(db)
    ca_service = CAService(db)

    # Get the certificate chain
    chain = []
    chain.append(cert.certificate)

    # Add issuer certificates, walking up the CA hierarchy
    current_issuer_id = cert.issuer_id
    while current_issuer_id is not None:
        issuer = await ca_service.get_ca(current_issuer_id)
        if not issuer:
            issuer_cert = await cert_service.get_certificate(current_issuer_id)
            if issuer_cert:
                chain.append(issuer_cert.certificate)
                current_issuer_id = issuer_cert.issuer_id
            else:
                break
        else:
            chain.append(issuer.certificate)
            current_issuer_id = issuer.parent_ca_id

    chain_pem = "\n".join(chain)

    return Response(
        content=chain_pem,
        media_type="application/x-pem-file",
        headers={
            "Content-Disposition": (
                f"attachment; filename=certificate_{cert_id}_chain.pem"
            )
        },
    )
