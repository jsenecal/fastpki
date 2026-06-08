from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_principal
from app.db.models import AuditAction, CertificateType, PermissionAction, UserRole
from app.db.session import get_session
from app.schemas.cert import (
    CertificateCreate,
    CertificateDetailResponse,
    CertificateRenewRequest,
    CertificateResponse,
    CertificateRevoke,
    CSRSignRequest,
)
from app.services.audit import AuditService
from app.services.ca import CAService
from app.services.cert import CertificateService
from app.services.encryption import EncryptionService
from app.services.exceptions import (
    CsrNotAllowedError,
    CsrRequiredError,
    IssuancePolicyMissingError,
    LeafCertNotAllowedError,
    NotFoundError,
    PermissionDeniedError,
    PolicyViolationError,
)
from app.services.issuance_policy import IssuancePolicyService
from app.services.permission import PermissionService
from app.services.principal import Principal

router = APIRouter()


async def _enforce_issuance_policy(
    db: AsyncSession,
    principal: Principal,
    *,
    common_name: str,
    san_dns_names: list[str] | None,
    san_ip_addresses: list[str] | None,
    san_email_addresses: list[str] | None,
    ca_id: int,
    certificate_type: CertificateType,
    valid_days: int | None,
) -> None:
    """Enforce the issuance policy for service-account principals.

    User-bound principals bypass policy entirely (current behavior preserved).
    """
    if principal.kind != "service_account":
        return
    try:
        await IssuancePolicyService(db).enforce(
            principal.id,
            common_name=common_name,
            san_dns_names=san_dns_names,
            san_ip_addresses=san_ip_addresses,
            san_email_addresses=san_email_addresses,
            ca_id=ca_id,
            certificate_type=certificate_type,
            valid_days=valid_days,
        )
    except IssuancePolicyMissingError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "service_account_has_no_policy"},
        ) from e
    except PolicyViolationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "policy_violation",
                "field": e.field,
                "value": e.value,
            },
        ) from e


@router.post(
    "/", response_model=CertificateDetailResponse, status_code=status.HTTP_201_CREATED
)
async def create_certificate(
    cert_in: CertificateCreate,
    ca_id: int,
    request: Request,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> CertificateDetailResponse:
    """Create a new certificate signed by the specified CA."""
    perm = PermissionService(db)
    try:
        await perm.check_ca_access(principal, ca_id, PermissionAction.CREATE_CERT)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e

    await _enforce_issuance_policy(
        db,
        principal,
        common_name=cert_in.common_name,
        san_dns_names=cert_in.san_dns_names,
        san_ip_addresses=cert_in.san_ip_addresses,
        san_email_addresses=cert_in.san_email_addresses,
        ca_id=ca_id,
        certificate_type=cert_in.certificate_type,
        valid_days=cert_in.valid_days,
    )

    cert_service = CertificateService(db)
    try:
        base_url = str(request.base_url).rstrip("/")
        cert = await cert_service.create_certificate(
            ca_id=ca_id,
            common_name=cert_in.common_name,
            subject_dn=cert_in.subject_dn,
            certificate_type=cert_in.certificate_type,
            key_size=cert_in.key_size,
            valid_days=cert_in.valid_days,
            include_private_key=cert_in.include_private_key,
            organization_id=principal.organization_id,
            base_url=base_url,
            san_dns_names=cert_in.san_dns_names,
            san_ip_addresses=cert_in.san_ip_addresses,
            san_email_addresses=cert_in.san_email_addresses,
            **principal.creator_fields(),
        )
    except LeafCertNotAllowedError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create certificate: {e!s}",
        ) from e
    else:
        audit_service = AuditService(db)
        await audit_service.log_action(
            action=AuditAction.CERT_CREATE,
            organization_id=principal.organization_id,
            resource_type="certificate",
            resource_id=cert.id,
            detail=f"Created certificate '{cert.common_name}'",
            **AuditService.actor_fields(principal),
        )
        cert.private_key = EncryptionService.decrypt_optional_private_key(
            cert.private_key
        )
        return cert


@router.post(
    "/sign-csr", response_model=CertificateResponse, status_code=status.HTTP_201_CREATED
)
async def sign_csr(
    csr_in: CSRSignRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> CertificateResponse:
    """Sign a CSR. Extracts defaults from the CSR; explicit fields override."""
    # Resolve CA by id or name
    if csr_in.ca_id is None and csr_in.ca_name is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either ca_id or ca_name must be provided",
        )
    ca_service = CAService(db)
    if csr_in.ca_id is not None:
        ca_id = csr_in.ca_id
    else:
        org_id = (
            principal.organization_id if principal.role != UserRole.SUPERUSER else None
        )
        ca = await ca_service.get_ca_by_name(csr_in.ca_name, organization_id=org_id)
        if ca is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"CA '{csr_in.ca_name}' not found",
            )
        ca_id = ca.id  # type: ignore[assignment]

    perm = PermissionService(db)
    try:
        await perm.check_ca_access(principal, ca_id, PermissionAction.CREATE_CERT)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e

    cert_service = CertificateService(db)

    if principal.kind == "service_account":
        try:
            effective = cert_service.resolve_csr_fields(
                csr_in.csr,
                common_name=csr_in.common_name,
                subject_dn=csr_in.subject_dn,
                san_dns_names=csr_in.san_dns_names,
                san_ip_addresses=csr_in.san_ip_addresses,
                san_email_addresses=csr_in.san_email_addresses,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
            ) from e
        await _enforce_issuance_policy(
            db,
            principal,
            common_name=effective.common_name,
            san_dns_names=effective.san_dns_names,
            san_ip_addresses=effective.san_ip_addresses,
            san_email_addresses=effective.san_email_addresses,
            ca_id=ca_id,
            certificate_type=csr_in.certificate_type,
            valid_days=csr_in.valid_days,
        )

    try:
        base_url = str(request.base_url).rstrip("/")
        cert = await cert_service.sign_csr(
            csr_pem=csr_in.csr,
            ca_id=ca_id,
            certificate_type=csr_in.certificate_type,
            valid_days=csr_in.valid_days,
            common_name=csr_in.common_name,
            subject_dn=csr_in.subject_dn,
            san_dns_names=csr_in.san_dns_names,
            san_ip_addresses=csr_in.san_ip_addresses,
            san_email_addresses=csr_in.san_email_addresses,
            organization_id=principal.organization_id,
            base_url=base_url,
            **principal.creator_fields(),
        )
    except LeafCertNotAllowedError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e

    audit_service = AuditService(db)
    await audit_service.log_action(
        action=AuditAction.CERT_CREATE,
        organization_id=principal.organization_id,
        resource_type="certificate",
        resource_id=cert.id,
        detail=f"Signed CSR for '{cert.common_name}'",
        **AuditService.actor_fields(principal),
    )
    return cert


@router.get("/", response_model=list[CertificateResponse])
async def read_certificates(
    ca_id: int | None = Query(None, description="Filter by CA ID"),
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> list[CertificateResponse]:
    """Get all certificates, optionally filtered by CA ID."""
    cert_service = CertificateService(db)
    if principal.role == UserRole.SUPERUSER:
        certs = await cert_service.list_certificates(ca_id=ca_id)
    else:
        certs = await cert_service.list_certificates(
            ca_id=ca_id,
            organization_id=principal.organization_id,
        )
    return certs


@router.get("/{cert_id}", response_model=CertificateResponse)
async def read_certificate(
    cert_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> CertificateResponse:
    """Get a specific certificate by ID."""
    perm = PermissionService(db)
    try:
        cert = await perm.check_cert_access(principal, cert_id, PermissionAction.READ)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e
    response = CertificateResponse.model_validate(cert)
    response.renewed_to_ids = await CertificateService(db).get_renewed_to_ids(cert_id)
    return response


@router.get("/{cert_id}/private-key", response_model=CertificateDetailResponse)
async def read_certificate_with_private_key(
    cert_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> CertificateDetailResponse:
    """Get a specific certificate by ID, including private key if available."""
    perm = PermissionService(db)
    try:
        cert = await perm.check_cert_access(
            principal, cert_id, PermissionAction.EXPORT_PRIVATE_KEY
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e
    audit_service = AuditService(db)
    await audit_service.log_action(
        action=AuditAction.CERT_EXPORT_PRIVATE_KEY,
        organization_id=principal.organization_id,
        resource_type="certificate",
        resource_id=cert_id,
        **AuditService.actor_fields(principal),
    )
    cert.private_key = EncryptionService.decrypt_optional_private_key(cert.private_key)
    return cert


@router.post("/{cert_id}/revoke", response_model=CertificateResponse)
async def revoke_certificate(
    cert_id: int,
    revoke_data: CertificateRevoke,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> CertificateResponse:
    """Revoke a certificate by ID."""
    perm = PermissionService(db)
    try:
        await perm.check_cert_access(principal, cert_id, PermissionAction.REVOKE_CERT)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e
    cert_service = CertificateService(db)
    try:
        cert = await cert_service.revoke_certificate(cert_id, reason=revoke_data.reason)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    if not cert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Certificate with ID {cert_id} not found",
        )
    audit_service = AuditService(db)
    await audit_service.log_action(
        action=AuditAction.CERT_REVOKE,
        organization_id=principal.organization_id,
        resource_type="certificate",
        resource_id=cert_id,
        detail=f"Revoked certificate '{cert.common_name}'",
        **AuditService.actor_fields(principal),
    )
    return cert


@router.post(
    "/{cert_id}/renew",
    response_model=CertificateDetailResponse,
    status_code=status.HTTP_201_CREATED,
)
async def renew_certificate_endpoint(
    cert_id: int,
    renew_in: CertificateRenewRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(get_current_principal),  # noqa: B008
) -> CertificateDetailResponse:
    """Renew a certificate, inheriting subject/SANs/CA/type and recording lineage.

    Empty body renews a server-key certificate; a ``csr`` body renews a
    CSR-origin certificate (its subject is ignored).
    """
    perm = PermissionService(db)
    try:
        predecessor = await perm.check_cert_access(
            principal, cert_id, PermissionAction.CREATE_CERT
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e

    cert_service = CertificateService(db)
    try:
        params = cert_service.inherited_renewal_params(predecessor)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e

    # Re-evaluate the policy against the inherited parameters (service accounts).
    await _enforce_issuance_policy(
        db,
        principal,
        common_name=params.common_name,
        san_dns_names=params.san_dns_names,
        san_ip_addresses=params.san_ip_addresses,
        san_email_addresses=params.san_email_addresses,
        ca_id=params.ca_id,
        certificate_type=params.certificate_type,
        valid_days=params.valid_days,
    )

    try:
        base_url = str(request.base_url).rstrip("/")
        cert = await cert_service.renew_certificate(
            cert_id,
            csr_pem=renew_in.csr,
            organization_id=principal.organization_id,
            base_url=base_url,
            **principal.creator_fields(),
        )
    except CsrRequiredError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "csr_required_for_csr_origin_cert", "message": str(e)},
        ) from e
    except CsrNotAllowedError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "csr_not_allowed_for_server_key_cert", "message": str(e)},
        ) from e
    except LeafCertNotAllowedError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e

    audit_service = AuditService(db)
    await audit_service.log_action(
        action=AuditAction.CERT_CREATE,
        organization_id=principal.organization_id,
        resource_type="certificate",
        resource_id=cert.id,
        detail=f"Renewed certificate '{cert.common_name}' from #{cert_id}",
        **AuditService.actor_fields(principal),
    )
    cert.private_key = EncryptionService.decrypt_optional_private_key(cert.private_key)
    return cert
