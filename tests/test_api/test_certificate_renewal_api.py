import pytest
import pytest_asyncio
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_principal
from app.db.models import CertificateType
from app.db.session import get_session
from app.services.ca import CAService
from app.services.cert import CertificateService
from app.services.issuance_policy import IssuancePolicyService
from app.services.organization import OrganizationService
from app.services.principal import Principal
from app.services.service_account import ServiceAccountService
from app.services.user import UserService
from tests.conftest import create_test_app, get_test_session


def _client(principal: Principal) -> AsyncClient:
    app: FastAPI = create_test_app()
    app.dependency_overrides[get_session] = get_test_session
    app.dependency_overrides[get_current_principal] = lambda: principal
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _make_csr(common_name: str) -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)]))
        .sign(key, hashes.SHA256())
    )
    return csr.public_bytes(serialization.Encoding.PEM).decode()


@pytest_asyncio.fixture
async def org(db):
    return await OrganizationService(db).create_organization(name="RenewApiOrg")


@pytest_asyncio.fixture
async def ca(db, org):
    return await CAService(db).create_ca(
        name="Renew API CA", subject_dn="CN=Renew API CA", organization_id=org.id
    )


@pytest_asyncio.fixture
async def user(db, org):
    return await UserService(db).create_user(
        username="renewer",
        email="renewer@example.com",
        password="password123",
        organization_id=org.id,
        can_create_cert=True,
    )


async def _server_cert(db, ca, org):
    return await CertificateService(db).create_certificate(
        ca_id=ca.id,
        common_name="svc.example.com",
        subject_dn="CN=svc.example.com",
        certificate_type=CertificateType.SERVER,
        valid_days=90,
        include_private_key=True,
        organization_id=org.id,
        san_dns_names=["svc.example.com"],
    )


@pytest.mark.asyncio
async def test_renew_server_cert_and_lineage(db, ca, org, user):
    original = await _server_cert(db, ca, org)
    async with _client(Principal.from_user(user)) as c:
        resp = await c.post(f"/api/v1/certificates/{original.id}/renew", json={})
        assert resp.status_code == 201, resp.text
        new = resp.json()
        assert new["renewed_from_id"] == original.id
        assert new["private_key"]  # server-key renewal returns a fresh key

        # Lineage is exposed on the predecessor's read response.
        read = await c.get(f"/api/v1/certificates/{original.id}")
        assert read.status_code == 200
        assert read.json()["renewed_to_ids"] == [new["id"]]


@pytest.mark.asyncio
async def test_renew_server_cert_rejects_csr(db, ca, org, user):
    original = await _server_cert(db, ca, org)
    async with _client(Principal.from_user(user)) as c:
        resp = await c.post(
            f"/api/v1/certificates/{original.id}/renew",
            json={"csr": _make_csr("svc.example.com")},
        )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "csr_not_allowed_for_server_key_cert"


@pytest.mark.asyncio
async def test_renew_csr_cert_requires_csr(db, ca, org, user):
    original = await CertificateService(db).sign_csr(
        csr_pem=_make_csr("csr.example.com"),
        ca_id=ca.id,
        certificate_type=CertificateType.SERVER,
        organization_id=org.id,
    )
    async with _client(Principal.from_user(user)) as c:
        empty = await c.post(f"/api/v1/certificates/{original.id}/renew", json={})
        ok = await c.post(
            f"/api/v1/certificates/{original.id}/renew",
            json={"csr": _make_csr("whatever.example.com")},
        )
    assert empty.status_code == 400
    assert empty.json()["detail"]["code"] == "csr_required_for_csr_origin_cert"
    assert ok.status_code == 201, ok.text


@pytest.mark.asyncio
async def test_tightened_policy_blocks_renewal(db, ca, org):
    sa = await ServiceAccountService(db).create_service_account(
        name="renew-sa", organization_id=org.id, can_create_cert=True
    )
    original = await CertificateService(db).create_certificate(
        ca_id=ca.id,
        common_name="svc.example.com",
        subject_dn="CN=svc.example.com",
        certificate_type=CertificateType.SERVER,
        valid_days=90,
        include_private_key=True,
        organization_id=org.id,
        created_by_service_account_id=sa.id,
    )
    # Policy that no longer permits the inherited CN.
    await IssuancePolicyService(db).set_policy(
        sa.id,
        cn_patterns=["*.allowed.com"],
        san_dns_patterns=[],
        san_ip_cidrs=[],
        san_email_domains=[],
        allowed_ca_ids=[ca.id],
        allowed_certificate_types=[CertificateType.SERVER],
        max_validity_days=365,
    )
    async with _client(Principal.from_service_account(sa)) as c:
        resp = await c.post(f"/api/v1/certificates/{original.id}/renew", json={})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "policy_violation"
    assert resp.json()["detail"]["field"] == "cn_patterns"
