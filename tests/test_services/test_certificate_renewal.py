import pytest
import pytest_asyncio
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from app.db.models import CertificateType
from app.services.ca import CAService
from app.services.cert import CertificateService
from app.services.exceptions import CsrNotAllowedError, CsrRequiredError
from app.services.organization import OrganizationService


def _make_csr(common_name: str) -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)]))
        .sign(key, hashes.SHA256())
    )
    return csr.public_bytes(serialization.Encoding.PEM).decode()


@pytest_asyncio.fixture
async def ca(db):
    org = await OrganizationService(db).create_organization(name="RenewOrg")
    return await CAService(db).create_ca(
        name="Renew CA", subject_dn="CN=Renew CA", organization_id=org.id
    )


@pytest.mark.asyncio
async def test_renew_server_key_cert_mints_fresh_key(db, ca):
    svc = CertificateService(db)
    original = await svc.create_certificate(
        ca_id=ca.id,
        common_name="svc.example.com",
        subject_dn="CN=svc.example.com",
        certificate_type=CertificateType.SERVER,
        valid_days=120,
        include_private_key=True,
        san_dns_names=["svc.example.com", "alt.example.com"],
    )

    renewed = await svc.renew_certificate(original.id)

    assert renewed.id != original.id
    assert renewed.renewed_from_id == original.id
    assert renewed.private_key is not None  # fresh server-generated key
    assert renewed.common_name == "svc.example.com"
    assert renewed.certificate_type == CertificateType.SERVER
    assert renewed.issuer_id == ca.id
    assert renewed.valid_days == 120
    # Inherited SANs from the predecessor certificate.
    dns, _ips, _emails = svc.extract_cert_san_names(renewed.certificate)
    assert set(dns) == {"svc.example.com", "alt.example.com"}


@pytest.mark.asyncio
async def test_renew_server_key_cert_rejects_csr(db, ca):
    svc = CertificateService(db)
    original = await svc.create_certificate(
        ca_id=ca.id,
        common_name="svc.example.com",
        subject_dn="CN=svc.example.com",
        certificate_type=CertificateType.SERVER,
        include_private_key=True,
    )
    with pytest.raises(CsrNotAllowedError):
        await svc.renew_certificate(original.id, csr_pem=_make_csr("svc.example.com"))


@pytest.mark.asyncio
async def test_renew_csr_origin_requires_csr(db, ca):
    svc = CertificateService(db)
    original = await svc.sign_csr(
        csr_pem=_make_csr("csr.example.com"),
        ca_id=ca.id,
        certificate_type=CertificateType.SERVER,
    )
    with pytest.raises(CsrRequiredError):
        await svc.renew_certificate(original.id)


@pytest.mark.asyncio
async def test_renew_csr_origin_with_new_csr_ignores_csr_subject(db, ca):
    svc = CertificateService(db)
    original = await svc.sign_csr(
        csr_pem=_make_csr("csr.example.com"),
        ca_id=ca.id,
        certificate_type=CertificateType.SERVER,
    )

    # New CSR carries a different subject, which must be ignored.
    renewed = await svc.renew_certificate(
        original.id, csr_pem=_make_csr("attacker.evil.com")
    )

    assert renewed.renewed_from_id == original.id
    assert renewed.common_name == "csr.example.com"  # inherited, not from CSR
    assert renewed.private_key is None  # CSR-origin: server never holds the key


@pytest.mark.asyncio
async def test_renewed_to_ids(db, ca):
    svc = CertificateService(db)
    original = await svc.create_certificate(
        ca_id=ca.id,
        common_name="svc.example.com",
        subject_dn="CN=svc.example.com",
        certificate_type=CertificateType.SERVER,
        include_private_key=True,
    )
    r1 = await svc.renew_certificate(original.id)
    assert await svc.get_renewed_to_ids(original.id) == [r1.id]
