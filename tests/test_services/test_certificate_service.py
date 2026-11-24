import ipaddress

import pytest
import pytest_asyncio
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    CertificateAuthority,
    CertificateStatus,
    CertificateType,
)
from app.services.ca import CAService
from app.services.cert import CertificateService


@pytest_asyncio.fixture
async def test_ca(db: AsyncSession) -> CertificateAuthority:
    """Create a test CA for certificate tests."""
    ca_service = CAService(db)
    ca = await ca_service.create_ca(
        name="Test CA for Certs",
        subject_dn="CN=Test CA for Certs,O=Test Organization,C=US",
        key_size=2048,
        valid_days=3650,
    )
    return ca


@pytest.mark.asyncio
async def test_create_certificate(db: AsyncSession, test_ca: CertificateAuthority):
    """Test creating a new certificate."""
    cert_service = CertificateService(db)
    cert = await cert_service.create_certificate(
        ca_id=test_ca.id,
        common_name="test.example.com",
        subject_dn="CN=test.example.com,O=Test Organization,C=US",
        certificate_type=CertificateType.SERVER,
        key_size=2048,
        valid_days=365,
    )

    assert cert.id is not None
    assert cert.common_name == "test.example.com"
    assert cert.subject_dn == "CN=test.example.com,O=Test Organization,C=US"
    assert cert.certificate_type == CertificateType.SERVER
    assert cert.key_size == 2048
    assert cert.valid_days == 365
    assert cert.status == CertificateStatus.VALID
    assert cert.private_key is not None
    assert cert.certificate is not None
    assert cert.serial_number is not None
    assert cert.not_before is not None
    assert cert.not_after is not None
    assert cert.issuer_id == test_ca.id


@pytest.mark.asyncio
async def test_create_certificate_without_private_key(
    db: AsyncSession, test_ca: CertificateAuthority
):
    """Test creating a new certificate without including the private key."""
    cert_service = CertificateService(db)
    cert = await cert_service.create_certificate(
        ca_id=test_ca.id,
        common_name="no-key.example.com",
        subject_dn="CN=no-key.example.com,O=Test Organization,C=US",
        certificate_type=CertificateType.SERVER,
        include_private_key=False,
    )

    assert cert.id is not None
    assert cert.common_name == "no-key.example.com"
    assert cert.private_key is None
    assert cert.certificate is not None


@pytest.mark.asyncio
async def test_get_certificate(db: AsyncSession, test_ca: CertificateAuthority):
    """Test retrieving a certificate by ID."""
    cert_service = CertificateService(db)
    created_cert = await cert_service.create_certificate(
        ca_id=test_ca.id,
        common_name="get.example.com",
        subject_dn="CN=get.example.com,O=Test Organization,C=US",
        certificate_type=CertificateType.SERVER,
    )

    cert = await cert_service.get_certificate(created_cert.id)

    assert cert is not None
    assert cert.id == created_cert.id
    assert cert.common_name == "get.example.com"


@pytest.mark.asyncio
async def test_list_certificates(db: AsyncSession, test_ca: CertificateAuthority):
    """Test listing certificates."""
    cert_service = CertificateService(db)
    for i in range(3):
        await cert_service.create_certificate(
            ca_id=test_ca.id,
            common_name=f"list{i}.example.com",
            subject_dn=f"CN=list{i}.example.com,O=Test Organization,C=US",
            certificate_type=CertificateType.SERVER,
        )

    certs = await cert_service.list_certificates()

    assert len(certs) >= 3

    certs = await cert_service.list_certificates(ca_id=test_ca.id)

    assert all(cert.issuer_id == test_ca.id for cert in certs)


@pytest.mark.asyncio
async def test_revoke_certificate(db: AsyncSession, test_ca: CertificateAuthority):
    """Test revoking a certificate."""
    cert_service = CertificateService(db)
    cert = await cert_service.create_certificate(
        ca_id=test_ca.id,
        common_name="revoke.example.com",
        subject_dn="CN=revoke.example.com,O=Test Organization,C=US",
        certificate_type=CertificateType.SERVER,
    )

    revoked_cert = await cert_service.revoke_certificate(
        cert.id, reason="Key compromise"
    )

    assert revoked_cert is not None
    assert revoked_cert.id == cert.id
    assert revoked_cert.status == CertificateStatus.REVOKED
    assert revoked_cert.revoked_at is not None


@pytest.mark.asyncio
async def test_double_revocation(db: AsyncSession, test_ca: CertificateAuthority):
    """Test that revoking an already-revoked certificate raises ValueError."""
    cert_service = CertificateService(db)
    cert = await cert_service.create_certificate(
        ca_id=test_ca.id,
        common_name="double-revoke.example.com",
        subject_dn="CN=double-revoke.example.com,O=Test Organization,C=US",
        certificate_type=CertificateType.SERVER,
    )

    await cert_service.revoke_certificate(cert.id, reason="Key compromise")

    with pytest.raises(ValueError, match="already revoked"):
        await cert_service.revoke_certificate(cert.id, reason="Duplicate")


@pytest.mark.asyncio
async def test_different_cert_types(db: AsyncSession, test_ca: CertificateAuthority):
    """Test creating different types of certificates."""
    cert_service = CertificateService(db)

    client_cert = await cert_service.create_certificate(
        ca_id=test_ca.id,
        common_name="client.example.com",
        subject_dn="CN=client.example.com,O=Test Organization,C=US",
        certificate_type=CertificateType.CLIENT,
    )

    assert client_cert.certificate_type == CertificateType.CLIENT

    ca_cert = await cert_service.create_certificate(
        ca_id=test_ca.id,
        common_name="subca.example.com",
        subject_dn="CN=subca.example.com,O=Test Organization,C=US",
        certificate_type=CertificateType.CA,
    )

    assert ca_cert.certificate_type == CertificateType.CA


@pytest.mark.asyncio
async def test_dual_purpose_cert_has_both_ekus(
    db: AsyncSession, test_ca: CertificateAuthority
):
    """Test that dual_purpose certs have both SERVER_AUTH and CLIENT_AUTH EKUs."""
    cert_service = CertificateService(db)
    cert = await cert_service.create_certificate(
        ca_id=test_ca.id,
        common_name="peer.example.com",
        subject_dn="CN=peer.example.com,O=Test Organization,C=US",
        certificate_type=CertificateType.DUAL_PURPOSE,
    )

    assert cert.certificate_type == CertificateType.DUAL_PURPOSE

    # Parse the X.509 certificate and check EKUs
    x509_cert = x509.load_pem_x509_certificate(cert.certificate.encode("utf-8"))
    eku_ext = x509_cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage)
    ekus = list(eku_ext.value)

    assert ExtendedKeyUsageOID.SERVER_AUTH in ekus
    assert ExtendedKeyUsageOID.CLIENT_AUTH in ekus


@pytest.mark.asyncio
async def test_server_cert_has_san_from_common_name(
    db: AsyncSession, test_ca: CertificateAuthority
):
    """Server certs should auto-populate SAN with common_name as DNS entry."""
    cert_service = CertificateService(db)
    cert = await cert_service.create_certificate(
        ca_id=test_ca.id,
        common_name="auto-san.example.com",
        subject_dn="CN=auto-san.example.com,O=Test,C=US",
        certificate_type=CertificateType.SERVER,
    )

    x509_cert = x509.load_pem_x509_certificate(cert.certificate.encode("utf-8"))
    san_ext = x509_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    dns_names = san_ext.value.get_values_for_type(x509.DNSName)
    assert "auto-san.example.com" in dns_names


@pytest.mark.asyncio
async def test_server_cert_with_explicit_sans(
    db: AsyncSession, test_ca: CertificateAuthority
):
    """Server certs should support explicit DNS and IP SANs."""
    cert_service = CertificateService(db)
    cert = await cert_service.create_certificate(
        ca_id=test_ca.id,
        common_name="multi-san.example.com",
        subject_dn="CN=multi-san.example.com,O=Test,C=US",
        certificate_type=CertificateType.SERVER,
        san_dns_names=["multi-san.example.com", "alt.example.com"],
        san_ip_addresses=["10.0.0.1", "::1"],
    )

    x509_cert = x509.load_pem_x509_certificate(cert.certificate.encode("utf-8"))
    san_ext = x509_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    dns_names = san_ext.value.get_values_for_type(x509.DNSName)
    assert "multi-san.example.com" in dns_names
    assert "alt.example.com" in dns_names

    import ipaddress

    ips = san_ext.value.get_values_for_type(x509.IPAddress)
    assert ipaddress.IPv4Address("10.0.0.1") in ips
    assert ipaddress.IPv6Address("::1") in ips


@pytest.mark.asyncio
async def test_wildcard_server_cert_san(
    db: AsyncSession, test_ca: CertificateAuthority
):
    """Wildcard common_name should appear as DNS SAN."""
    cert_service = CertificateService(db)
    cert = await cert_service.create_certificate(
        ca_id=test_ca.id,
        common_name="*.example.com",
        subject_dn="CN=*.example.com,O=Test,C=US",
        certificate_type=CertificateType.SERVER,
    )

    x509_cert = x509.load_pem_x509_certificate(cert.certificate.encode("utf-8"))
    san_ext = x509_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    dns_names = san_ext.value.get_values_for_type(x509.DNSName)
    assert "*.example.com" in dns_names


@pytest.mark.asyncio
async def test_client_cert_with_email_san(
    db: AsyncSession, test_ca: CertificateAuthority
):
    """Client certs should support email SANs."""
    cert_service = CertificateService(db)
    cert = await cert_service.create_certificate(
        ca_id=test_ca.id,
        common_name="user@example.com",
        subject_dn="CN=user@example.com,O=Test,C=US",
        certificate_type=CertificateType.CLIENT,
        san_email_addresses=["user@example.com"],
    )

    x509_cert = x509.load_pem_x509_certificate(cert.certificate.encode("utf-8"))
    san_ext = x509_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    emails = san_ext.value.get_values_for_type(x509.RFC822Name)
    assert "user@example.com" in emails


@pytest.mark.asyncio
async def test_client_cert_auto_email_from_cn(
    db: AsyncSession, test_ca: CertificateAuthority
):
    """Client certs should auto-add CN as email SAN if it looks like an email."""
    cert_service = CertificateService(db)
    cert = await cert_service.create_certificate(
        ca_id=test_ca.id,
        common_name="auto@example.com",
        subject_dn="CN=auto@example.com,O=Test,C=US",
        certificate_type=CertificateType.CLIENT,
    )

    x509_cert = x509.load_pem_x509_certificate(cert.certificate.encode("utf-8"))
    san_ext = x509_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    emails = san_ext.value.get_values_for_type(x509.RFC822Name)
    assert "auto@example.com" in emails


@pytest.mark.asyncio
async def test_dual_purpose_cert_supports_all_san_types(
    db: AsyncSession, test_ca: CertificateAuthority
):
    """Dual-purpose certs should support DNS, IP, and email SANs."""
    cert_service = CertificateService(db)
    cert = await cert_service.create_certificate(
        ca_id=test_ca.id,
        common_name="dual.example.com",
        subject_dn="CN=dual.example.com,O=Test,C=US",
        certificate_type=CertificateType.DUAL_PURPOSE,
        san_dns_names=["dual.example.com"],
        san_ip_addresses=["192.168.1.1"],
        san_email_addresses=["admin@example.com"],
    )

    x509_cert = x509.load_pem_x509_certificate(cert.certificate.encode("utf-8"))
    san_ext = x509_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    assert "dual.example.com" in san_ext.value.get_values_for_type(x509.DNSName)
    assert "admin@example.com" in san_ext.value.get_values_for_type(x509.RFC822Name)

    import ipaddress

    ips = san_ext.value.get_values_for_type(x509.IPAddress)
    assert ipaddress.IPv4Address("192.168.1.1") in ips


@pytest.mark.asyncio
async def test_server_cert_rejects_email_san(
    db: AsyncSession, test_ca: CertificateAuthority
):
    """Server certs should reject email SANs."""
    cert_service = CertificateService(db)
    with pytest.raises(ValueError, match=r"[Ee]mail"):
        await cert_service.create_certificate(
            ca_id=test_ca.id,
            common_name="server.example.com",
            subject_dn="CN=server.example.com,O=Test,C=US",
            certificate_type=CertificateType.SERVER,
            san_email_addresses=["bad@example.com"],
        )


@pytest.mark.asyncio
async def test_client_cert_rejects_dns_san(
    db: AsyncSession, test_ca: CertificateAuthority
):
    """Client certs should reject DNS SANs."""
    cert_service = CertificateService(db)
    with pytest.raises(ValueError, match="DNS"):
        await cert_service.create_certificate(
            ca_id=test_ca.id,
            common_name="client@example.com",
            subject_dn="CN=client@example.com,O=Test,C=US",
            certificate_type=CertificateType.CLIENT,
            san_dns_names=["bad.example.com"],
        )


@pytest.mark.asyncio
async def test_client_cert_rejects_ip_san(
    db: AsyncSession, test_ca: CertificateAuthority
):
    """Client certs should reject IP SANs."""
    cert_service = CertificateService(db)
    with pytest.raises(ValueError, match="IP"):
        await cert_service.create_certificate(
            ca_id=test_ca.id,
            common_name="client@example.com",
            subject_dn="CN=client@example.com,O=Test,C=US",
            certificate_type=CertificateType.CLIENT,
            san_ip_addresses=["10.0.0.1"],
        )


@pytest.mark.asyncio
async def test_ca_cert_no_auto_san(db: AsyncSession, test_ca: CertificateAuthority):
    """CA type certs should not get auto-populated SANs."""
    cert_service = CertificateService(db)
    cert = await cert_service.create_certificate(
        ca_id=test_ca.id,
        common_name="subca.example.com",
        subject_dn="CN=subca.example.com,O=Test,C=US",
        certificate_type=CertificateType.CA,
    )

    x509_cert = x509.load_pem_x509_certificate(cert.certificate.encode("utf-8"))
    with pytest.raises(x509.ExtensionNotFound):
        x509_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)


@pytest_asyncio.fixture
async def sample_csr() -> str:
    """Generate a sample CSR for testing."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(
            x509.Name(
                [
                    x509.NameAttribute(NameOID.COMMON_NAME, "csr.example.com"),
                    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "CSR Org"),
                    x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                ]
            )
        )
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("csr.example.com"),
                    x509.DNSName("alt.example.com"),
                    x509.IPAddress(ipaddress.IPv4Address("10.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    return csr.public_bytes(serialization.Encoding.PEM).decode("utf-8")


@pytest_asyncio.fixture
async def simple_csr() -> str:
    """Generate a CSR with no SANs."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(
            x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "simple.example.com")])
        )
        .sign(key, hashes.SHA256())
    )
    return csr.public_bytes(serialization.Encoding.PEM).decode("utf-8")


@pytest.mark.asyncio
async def test_sign_csr_uses_csr_values(
    db: AsyncSession, test_ca: CertificateAuthority, sample_csr: str
):
    """sign_csr should extract CN, subject, and SANs from the CSR."""
    cert_service = CertificateService(db)
    cert = await cert_service.sign_csr(
        csr_pem=sample_csr,
        ca_id=test_ca.id,
        certificate_type=CertificateType.SERVER,
    )

    assert cert.common_name == "csr.example.com"
    assert cert.private_key is None

    x509_cert = x509.load_pem_x509_certificate(cert.certificate.encode("utf-8"))
    san_ext = x509_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    dns_names = san_ext.value.get_values_for_type(x509.DNSName)
    assert "csr.example.com" in dns_names
    assert "alt.example.com" in dns_names

    ips = san_ext.value.get_values_for_type(x509.IPAddress)
    assert ipaddress.IPv4Address("10.0.0.1") in ips


@pytest.mark.asyncio
async def test_sign_csr_overrides_cn(
    db: AsyncSession, test_ca: CertificateAuthority, sample_csr: str
):
    """Explicit common_name should override the CSR's CN."""
    cert_service = CertificateService(db)
    cert = await cert_service.sign_csr(
        csr_pem=sample_csr,
        ca_id=test_ca.id,
        certificate_type=CertificateType.SERVER,
        common_name="override.example.com",
        san_dns_names=["override.example.com"],
    )

    assert cert.common_name == "override.example.com"


@pytest.mark.asyncio
async def test_sign_csr_overrides_sans(
    db: AsyncSession, test_ca: CertificateAuthority, sample_csr: str
):
    """Explicit SANs should override the CSR's SANs."""
    cert_service = CertificateService(db)
    cert = await cert_service.sign_csr(
        csr_pem=sample_csr,
        ca_id=test_ca.id,
        certificate_type=CertificateType.SERVER,
        san_dns_names=["only-this.example.com"],
    )

    x509_cert = x509.load_pem_x509_certificate(cert.certificate.encode("utf-8"))
    san_ext = x509_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    dns_names = san_ext.value.get_values_for_type(x509.DNSName)
    assert dns_names == ["only-this.example.com"]


@pytest.mark.asyncio
async def test_sign_csr_auto_populates_san_when_csr_has_none(
    db: AsyncSession, test_ca: CertificateAuthority, simple_csr: str
):
    """CSR with no SANs should auto-populate from CN for server certs."""
    cert_service = CertificateService(db)
    cert = await cert_service.sign_csr(
        csr_pem=simple_csr,
        ca_id=test_ca.id,
        certificate_type=CertificateType.SERVER,
    )

    x509_cert = x509.load_pem_x509_certificate(cert.certificate.encode("utf-8"))
    san_ext = x509_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    dns_names = san_ext.value.get_values_for_type(x509.DNSName)
    assert "simple.example.com" in dns_names


@pytest.mark.asyncio
async def test_sign_csr_uses_csr_public_key(
    db: AsyncSession, test_ca: CertificateAuthority, sample_csr: str
):
    """The signed certificate should use the public key from the CSR."""
    csr = x509.load_pem_x509_csr(sample_csr.encode("utf-8"))
    csr_pub_bytes = csr.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )

    cert_service = CertificateService(db)
    cert = await cert_service.sign_csr(
        csr_pem=sample_csr,
        ca_id=test_ca.id,
        certificate_type=CertificateType.SERVER,
    )

    x509_cert = x509.load_pem_x509_certificate(cert.certificate.encode("utf-8"))
    cert_pub_bytes = x509_cert.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    assert csr_pub_bytes == cert_pub_bytes


@pytest.mark.asyncio
async def test_sign_csr_rejects_invalid_csr(
    db: AsyncSession, test_ca: CertificateAuthority
):
    """Invalid CSR PEM should raise ValueError."""
    cert_service = CertificateService(db)
    with pytest.raises(ValueError, match="Invalid CSR"):
        await cert_service.sign_csr(
            csr_pem="not a valid CSR",
            ca_id=test_ca.id,
            certificate_type=CertificateType.SERVER,
        )


def test_parse_subject_dn_escaped_comma():
    """Test that escaped commas in DN values are handled correctly."""
    name = CAService.parse_subject_dn("CN=Doe\\, John,O=Test Org,C=US")
    cn = name.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
    assert cn == "Doe, John"
    org = name.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)[0].value
    assert org == "Test Org"


# --- Effective SANs (issue #53) -------------------------------------------


def _x509_sans(cert_pem: str) -> tuple[list[str], list[str], list[str]]:
    return CertificateService.extract_cert_san_names(cert_pem)


@pytest.mark.asyncio
async def test_server_cert_ip_only_sans_do_not_auto_add_cn(
    db: AsyncSession, test_ca: CertificateAuthority
):
    """An explicit IP-only SAN list is intentional: no CN DNS SAN is added."""
    cert = await CertificateService(db).create_certificate(
        ca_id=test_ca.id,
        common_name="ip-only.example.com",
        subject_dn="CN=ip-only.example.com",
        certificate_type=CertificateType.SERVER,
        san_ip_addresses=["10.24.0.5"],
    )
    dns, ips, _ = _x509_sans(cert.certificate)
    assert dns == []
    assert ips == ["10.24.0.5"]


@pytest.mark.asyncio
async def test_server_cert_invalid_dns_cn_not_auto_added(
    db: AsyncSession, test_ca: CertificateAuthority
):
    """A CN that is not a valid DNS name must not be written into a DNS SAN."""
    cert = await CertificateService(db).create_certificate(
        ca_id=test_ca.id,
        common_name="My Web Server",
        subject_dn="CN=My Web Server",
        certificate_type=CertificateType.SERVER,
    )
    x509_cert = x509.load_pem_x509_certificate(cert.certificate.encode("utf-8"))
    with pytest.raises(x509.ExtensionNotFound):
        x509_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)


@pytest.mark.parametrize(
    ("cert_type", "cn", "dns", "ips", "emails", "expected"),
    [
        # No SANs requested: CN becomes the DNS SAN for server-ish certs.
        (
            CertificateType.SERVER,
            "a.example.com",
            None,
            None,
            None,
            (["a.example.com"], None, None),
        ),
        (
            CertificateType.DUAL_PURPOSE,
            "*.example.com",
            [],
            [],
            [],
            (["*.example.com"], None, None),
        ),
        (
            CertificateType.SERVER,
            "localhost",
            None,
            None,
            None,
            (["localhost"], None, None),
        ),
        (
            CertificateType.SERVER,
            "under_score.example.com",
            None,
            None,
            None,
            (["under_score.example.com"], None, None),
        ),
        # Any explicit SAN list suppresses auto-population.
        (
            CertificateType.SERVER,
            "a.example.com",
            None,
            ["10.0.0.1"],
            None,
            (None, ["10.0.0.1"], None),
        ),
        (
            CertificateType.DUAL_PURPOSE,
            "a.example.com",
            None,
            None,
            ["x@ex.com"],
            (None, None, ["x@ex.com"]),
        ),
        (
            CertificateType.SERVER,
            "a.example.com",
            ["b.example.com"],
            None,
            None,
            (["b.example.com"], None, None),
        ),
        # CNs that are not valid DNS names are never auto-added.
        (CertificateType.SERVER, "My Web Server", None, None, None, (None, None, None)),
        (CertificateType.SERVER, "10.0.0.1", None, None, None, (None, None, None)),
        (
            CertificateType.SERVER,
            "héllo.example.com",
            None,
            None,
            None,
            (None, None, None),
        ),
        (
            CertificateType.SERVER,
            "-bad.example.com",
            None,
            None,
            None,
            (None, None, None),
        ),
        (
            CertificateType.SERVER,
            "a..example.com",
            None,
            None,
            None,
            (None, None, None),
        ),
        (
            CertificateType.SERVER,
            "a.*.example.com",
            None,
            None,
            None,
            (None, None, None),
        ),
        (
            CertificateType.SERVER,
            "a" * 64 + ".example.com",
            None,
            None,
            None,
            (None, None, None),
        ),
        (
            CertificateType.SERVER,
            ("a" * 63 + ".") * 4 + "com",
            None,
            None,
            None,
            (None, None, None),
        ),
        (CertificateType.SERVER, "", None, None, None, (None, None, None)),
        # Client: CN auto-added as email only when it looks like one.
        (
            CertificateType.CLIENT,
            "u@example.com",
            None,
            None,
            None,
            (None, None, ["u@example.com"]),
        ),
        (CertificateType.CLIENT, "not-an-email", None, None, None, (None, None, None)),
        (
            CertificateType.CLIENT,
            "u@example.com",
            None,
            None,
            ["o@example.com"],
            (None, None, ["o@example.com"]),
        ),
        # CA certificates never carry SANs.
        (
            CertificateType.CA,
            "ca.example.com",
            ["x.example.com"],
            None,
            None,
            (None, None, None),
        ),
    ],
)
def test_effective_san_names(cert_type, cn, dns, ips, emails, expected):
    eff = CertificateService.effective_san_names(
        cert_type,
        cn,
        san_dns_names=dns,
        san_ip_addresses=ips,
        san_email_addresses=emails,
    )
    assert (eff.san_dns_names, eff.san_ip_addresses, eff.san_email_addresses) == (
        expected
    )


def test_effective_san_names_is_idempotent():
    first = CertificateService.effective_san_names(
        CertificateType.SERVER, "a.example.com", None, None, None
    )
    second = CertificateService.effective_san_names(
        CertificateType.SERVER,
        "a.example.com",
        first.san_dns_names,
        first.san_ip_addresses,
        first.san_email_addresses,
    )
    assert first == second


@pytest.mark.asyncio
async def test_resolve_csr_fields_includes_auto_populated_cn(
    db: AsyncSession, simple_csr: str
):
    """Resolved CSR fields reflect the effective SANs, incl. auto-population."""
    fields = CertificateService(db).resolve_csr_fields(
        simple_csr, certificate_type=CertificateType.SERVER
    )
    assert fields.san_dns_names == ["simple.example.com"]


@pytest.mark.asyncio
async def test_resolve_csr_fields_ip_override_suppresses_auto_population(
    db: AsyncSession, simple_csr: str
):
    fields = CertificateService(db).resolve_csr_fields(
        simple_csr,
        certificate_type=CertificateType.SERVER,
        san_ip_addresses=["10.24.0.9"],
    )
    assert fields.san_dns_names is None
    assert fields.san_ip_addresses == ["10.24.0.9"]
