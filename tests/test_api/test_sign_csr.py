"""API-level tests for POST /certificates/sign-csr.

Service-layer behavior is covered by tests/test_services/test_certificate_service.py;
these exercise the FastAPI route (CA resolution, audit logging, response shape).
"""

import ipaddress

import pytest
import pytest_asyncio
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi import status
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import AuditLog, User


def _make_csr(
    common_name: str = "csr.example.com",
    dns_names: list[str] | None = None,
) -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "CSR Org"),
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        ]
    )
    builder = x509.CertificateSigningRequestBuilder().subject_name(subject)
    if dns_names:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName(n) for n in dns_names]),
            critical=False,
        )
    csr = builder.sign(key, hashes.SHA256())
    return csr.public_bytes(serialization.Encoding.PEM).decode("utf-8")


@pytest_asyncio.fixture
async def test_ca_id(superuser_client: AsyncClient) -> int:
    ca_data = {
        "name": "Sign-CSR Test CA",
        "subject_dn": "CN=Sign-CSR Test CA,O=Test,C=US",
    }
    response = await superuser_client.post(
        f"{settings.API_V1_STR}/cas/",
        json=ca_data,
    )
    assert response.status_code == status.HTTP_201_CREATED
    return response.json()["id"]


@pytest.mark.asyncio
async def test_sign_csr_by_ca_id_extracts_csr_values(
    superuser_client: AsyncClient, test_ca_id: int
):
    csr_pem = _make_csr(
        common_name="api-csr.example.com",
        dns_names=["api-csr.example.com", "alt.example.com"],
    )
    payload = {
        "csr": csr_pem,
        "ca_id": test_ca_id,
        "certificate_type": "server",
    }

    response = await superuser_client.post(
        f"{settings.API_V1_STR}/certificates/sign-csr",
        json=payload,
    )

    assert response.status_code == status.HTTP_201_CREATED, response.text
    body = response.json()
    assert body["common_name"] == "api-csr.example.com"
    assert body["certificate_type"] == "server"
    assert body["issuer_id"] == test_ca_id
    assert body["status"] == "valid"
    assert "certificate" in body

    cert_pem = body["certificate"]
    parsed = x509.load_pem_x509_certificate(cert_pem.encode("utf-8"))
    san = parsed.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    dns_names = san.value.get_values_for_type(x509.DNSName)
    assert "api-csr.example.com" in dns_names
    assert "alt.example.com" in dns_names


@pytest.mark.asyncio
async def test_sign_csr_by_ca_name(superuser_client: AsyncClient, test_ca_id: int):
    csr_pem = _make_csr(common_name="byname.example.com")
    payload = {
        "csr": csr_pem,
        "ca_name": "Sign-CSR Test CA",
        "certificate_type": "server",
    }

    response = await superuser_client.post(
        f"{settings.API_V1_STR}/certificates/sign-csr",
        json=payload,
    )

    assert response.status_code == status.HTTP_201_CREATED, response.text
    body = response.json()
    assert body["issuer_id"] == test_ca_id
    assert body["common_name"] == "byname.example.com"


@pytest.mark.asyncio
async def test_sign_csr_overrides_cn_and_sans(
    superuser_client: AsyncClient, test_ca_id: int
):
    csr_pem = _make_csr(
        common_name="original.example.com",
        dns_names=["original.example.com"],
    )
    payload = {
        "csr": csr_pem,
        "ca_id": test_ca_id,
        "certificate_type": "server",
        "common_name": "override.example.com",
        "san_dns_names": ["override.example.com", "extra.example.com"],
        "san_ip_addresses": ["10.0.0.5"],
    }

    response = await superuser_client.post(
        f"{settings.API_V1_STR}/certificates/sign-csr",
        json=payload,
    )

    assert response.status_code == status.HTTP_201_CREATED, response.text
    body = response.json()
    assert body["common_name"] == "override.example.com"

    parsed = x509.load_pem_x509_certificate(body["certificate"].encode("utf-8"))
    san = parsed.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    dns_names = san.value.get_values_for_type(x509.DNSName)
    ips = san.value.get_values_for_type(x509.IPAddress)
    assert "override.example.com" in dns_names
    assert "extra.example.com" in dns_names
    assert "original.example.com" not in dns_names
    assert ipaddress.IPv4Address("10.0.0.5") in ips


@pytest.mark.asyncio
async def test_sign_csr_requires_ca_identifier(
    superuser_client: AsyncClient, test_ca_id: int
):
    csr_pem = _make_csr()
    payload = {
        "csr": csr_pem,
        "certificate_type": "server",
    }

    response = await superuser_client.post(
        f"{settings.API_V1_STR}/certificates/sign-csr",
        json=payload,
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "ca_id" in response.json()["detail"]


@pytest.mark.asyncio
async def test_sign_csr_unknown_ca_name_returns_404(
    superuser_client: AsyncClient, test_ca_id: int
):
    csr_pem = _make_csr()
    payload = {
        "csr": csr_pem,
        "ca_name": "Nonexistent CA",
        "certificate_type": "server",
    }

    response = await superuser_client.post(
        f"{settings.API_V1_STR}/certificates/sign-csr",
        json=payload,
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_sign_csr_invalid_csr_returns_400(
    superuser_client: AsyncClient, test_ca_id: int
):
    payload = {
        "csr": "-----BEGIN CERTIFICATE REQUEST-----\nnot-valid\n-----END CERTIFICATE REQUEST-----",
        "ca_id": test_ca_id,
        "certificate_type": "server",
    }

    response = await superuser_client.post(
        f"{settings.API_V1_STR}/certificates/sign-csr",
        json=payload,
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.asyncio
async def test_sign_csr_writes_audit_log(
    superuser_client: AsyncClient,
    test_ca_id: int,
    db: AsyncSession,
    superuser: User,
):
    csr_pem = _make_csr(common_name="audited.example.com")
    payload = {
        "csr": csr_pem,
        "ca_id": test_ca_id,
        "certificate_type": "server",
    }

    response = await superuser_client.post(
        f"{settings.API_V1_STR}/certificates/sign-csr",
        json=payload,
    )
    assert response.status_code == status.HTTP_201_CREATED

    result = await db.execute(
        select(AuditLog)
        .where(AuditLog.user_id == superuser.id)
        .where(AuditLog.action == "cert_create")
    )
    logs = result.scalars().all()
    assert any(
        log.detail and "audited.example.com" in log.detail and "CSR" in log.detail
        for log in logs
    ), [log.detail for log in logs]
