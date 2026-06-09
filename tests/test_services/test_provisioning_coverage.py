"""Edge-case coverage for the provisioning features (#28/#29/#30)."""

import pytest
import pytest_asyncio
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from app.db.models import CertificateStatus, CertificateType
from app.services.ca import CAService
from app.services.cert import CertificateService
from app.services.exceptions import (
    AlreadyExistsError,
    NotFoundError,
    PolicyViolationError,
)
from app.services.issuance_policy import IssuancePolicyService
from app.services.organization import OrganizationService
from app.services.service_account import ServiceAccountService


@pytest_asyncio.fixture
async def org(db):
    return await OrganizationService(db).create_organization(name="CovOrg")


@pytest_asyncio.fixture
async def ca(db, org):
    return await CAService(db).create_ca(
        name="Cov CA", subject_dn="CN=Cov CA", organization_id=org.id
    )


@pytest_asyncio.fixture
async def sa(db, org):
    return await ServiceAccountService(db).create_service_account(
        name="cov-sa", organization_id=org.id, can_create_cert=True
    )


# --- issuance policy IP/CIDR error paths ---


def _policy_with_cidrs(cidrs):
    from app.db.models import IssuancePolicy

    return IssuancePolicy(
        service_account_id=1,
        cn_patterns=["*"],
        san_dns_patterns=[],
        san_ip_cidrs=cidrs,
        san_email_domains=[],
        allowed_ca_ids=[1],
        allowed_certificate_types=[CertificateType.SERVER.value],
        max_validity_days=365,
    )


def test_evaluate_malformed_ip_san_rejected():
    policy = _policy_with_cidrs(["10.0.0.0/8"])
    with pytest.raises(PolicyViolationError) as exc:
        IssuancePolicyService.evaluate(
            policy,
            common_name="x",
            san_dns_names=None,
            san_ip_addresses=["not-an-ip"],
            san_email_addresses=None,
            ca_id=1,
            certificate_type=CertificateType.SERVER,
            valid_days=1,
        )
    assert exc.value.field == "san_ip_cidrs"


def test_evaluate_malformed_cidr_in_policy_is_skipped():
    # An unparseable CIDR is skipped; with no valid CIDR the IP is rejected.
    policy = _policy_with_cidrs(["not-a-cidr"])
    with pytest.raises(PolicyViolationError) as exc:
        IssuancePolicyService.evaluate(
            policy,
            common_name="x",
            san_dns_names=None,
            san_ip_addresses=["10.0.0.1"],
            san_email_addresses=None,
            ca_id=1,
            certificate_type=CertificateType.SERVER,
            valid_days=1,
        )
    assert exc.value.field == "san_ip_cidrs"


# --- service account update + token edge paths ---


@pytest.mark.asyncio
async def test_update_service_account_all_fields(db, sa):
    service = ServiceAccountService(db)
    updated = await service.update_service_account(
        sa.id,
        description="changed",
        can_create_ca=True,
        can_create_cert=False,
        can_revoke_cert=True,
        can_export_private_key=True,
        can_delete_ca=True,
    )
    assert updated.description == "changed"
    assert updated.can_create_ca is True
    assert updated.can_create_cert is False
    assert updated.can_revoke_cert is True
    assert updated.can_export_private_key is True
    assert updated.can_delete_ca is True


@pytest.mark.asyncio
async def test_update_service_account_duplicate_name_rejected(db, org, sa):
    service = ServiceAccountService(db)
    other = await service.create_service_account(name="taken", organization_id=org.id)
    with pytest.raises(AlreadyExistsError):
        await service.update_service_account(sa.id, name=other.name)


@pytest.mark.asyncio
async def test_revoke_unknown_token_raises(db):
    with pytest.raises(NotFoundError):
        await ServiceAccountService(db).revoke_token(999999)


# --- certificate service error paths ---


@pytest.mark.asyncio
async def test_create_certificate_unknown_ca_raises(db):
    with pytest.raises(ValueError, match="No CA"):
        await CertificateService(db).create_certificate(
            ca_id=999999,
            common_name="x.example.com",
            subject_dn="CN=x.example.com",
            certificate_type=CertificateType.SERVER,
        )


@pytest.mark.asyncio
async def test_sign_csr_without_common_name_raises(db, ca):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(x509.Name([]))  # no CN
        .sign(key, hashes.SHA256())
    )
    csr_pem = csr.public_bytes(serialization.Encoding.PEM).decode()
    with pytest.raises(ValueError, match="No common name"):
        await CertificateService(db).sign_csr(
            csr_pem=csr_pem, ca_id=ca.id, certificate_type=CertificateType.SERVER
        )


@pytest.mark.asyncio
async def test_renew_unknown_certificate_raises(db):
    with pytest.raises(ValueError, match="not found"):
        await CertificateService(db).renew_certificate(999999)


@pytest.mark.asyncio
async def test_renew_certificate_without_issuer_raises(db, ca):
    svc = CertificateService(db)
    cert = await svc.create_certificate(
        ca_id=ca.id,
        common_name="noissuer.example.com",
        subject_dn="CN=noissuer.example.com",
        certificate_type=CertificateType.SERVER,
        include_private_key=True,
    )
    cert.issuer_id = None
    db.add(cert)
    await db.commit()
    with pytest.raises(ValueError, match="no issuer"):
        await svc.renew_certificate(cert.id)


@pytest.mark.asyncio
async def test_renew_certificate_without_sans(db, ca):
    # A predecessor with no SAN extension exercises the ExtensionNotFound path.
    svc = CertificateService(db)
    cert = await svc.create_certificate(
        ca_id=ca.id,
        common_name="plain.example.com",
        subject_dn="CN=plain.example.com",
        certificate_type=CertificateType.SERVER,
        include_private_key=True,
    )
    renewed = await svc.renew_certificate(cert.id)
    assert renewed.renewed_from_id == cert.id


@pytest.mark.asyncio
async def test_revoke_already_revoked_raises(db, ca):
    svc = CertificateService(db)
    cert = await svc.create_certificate(
        ca_id=ca.id,
        common_name="rev.example.com",
        subject_dn="CN=rev.example.com",
        certificate_type=CertificateType.SERVER,
        include_private_key=True,
    )
    await svc.revoke_certificate(cert.id)
    with pytest.raises(ValueError, match="already revoked"):
        await svc.revoke_certificate(cert.id)


@pytest.mark.asyncio
async def test_revoke_certificate_without_issuer_raises(db, ca):
    svc = CertificateService(db)
    cert = await svc.create_certificate(
        ca_id=ca.id,
        common_name="noiss-rev.example.com",
        subject_dn="CN=noiss-rev.example.com",
        certificate_type=CertificateType.SERVER,
        include_private_key=True,
    )
    cert.issuer_id = None
    cert.status = CertificateStatus.VALID
    db.add(cert)
    await db.commit()
    with pytest.raises(ValueError, match="no issuer"):
        await svc.revoke_certificate(cert.id)
