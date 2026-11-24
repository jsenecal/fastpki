from cryptography import x509
from cryptography.hazmat.primitives import serialization
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CertificateAuthority
from app.db.session import get_session
from app.services.ca import CAService

crl_router = APIRouter()
ca_router = APIRouter()


async def _resolve_ca(ca_slug: str, db: AsyncSession) -> CertificateAuthority:
    ca_service = CAService(db)
    ca = await ca_service.get_ca_by_slug(ca_slug)
    if not ca:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="CA not found"
        )
    return ca


# PEM routes MUST be registered before the catch-all {ca_slug} routes,
# otherwise FastAPI treats "slug.pem" as a single path parameter value.


@crl_router.get("/{ca_slug}.pem")
async def get_crl_pem(
    ca_slug: str,
    db: AsyncSession = Depends(get_session, scope="function"),  # noqa: B008
) -> Response:
    """Download the CRL for a CA in PEM format."""
    ca = await _resolve_ca(ca_slug, db)
    ca_service = CAService(db)
    crl_der = await ca_service.generate_crl(ca.id)
    crl = x509.load_der_x509_crl(crl_der)
    crl_pem = crl.public_bytes(serialization.Encoding.PEM)
    return Response(
        content=crl_pem,
        media_type="application/x-pem-file",
        headers={
            "Content-Disposition": f"attachment; filename={ca_slug}.crl.pem",
        },
    )


@crl_router.get("/{ca_slug}")
async def get_crl_der(
    ca_slug: str,
    db: AsyncSession = Depends(get_session, scope="function"),  # noqa: B008
) -> Response:
    """Download the CRL for a CA in DER format."""
    ca = await _resolve_ca(ca_slug, db)
    ca_service = CAService(db)
    crl_der = await ca_service.generate_crl(ca.id)
    return Response(
        content=crl_der,
        media_type="application/pkix-crl",
        headers={
            "Content-Disposition": f"attachment; filename={ca_slug}.crl",
        },
    )


@ca_router.get("/{ca_slug}.pem")
async def get_ca_cert_pem(
    ca_slug: str,
    db: AsyncSession = Depends(get_session, scope="function"),  # noqa: B008
) -> Response:
    """Download a CA certificate in PEM format."""
    ca = await _resolve_ca(ca_slug, db)
    return Response(
        content=ca.certificate,
        media_type="application/x-pem-file",
        headers={
            "Content-Disposition": f"attachment; filename={ca_slug}.pem",
        },
    )


@ca_router.get("/{ca_slug}.crt")
async def get_ca_cert_der(
    ca_slug: str,
    db: AsyncSession = Depends(get_session, scope="function"),  # noqa: B008
) -> Response:
    """Download a CA certificate in DER format."""
    ca = await _resolve_ca(ca_slug, db)
    cert = x509.load_pem_x509_certificate(ca.certificate.encode("utf-8"))
    cert_der = cert.public_bytes(serialization.Encoding.DER)
    return Response(
        content=cert_der,
        media_type="application/pkix-cert",
        headers={
            "Content-Disposition": f"attachment; filename={ca_slug}.crt",
        },
    )
