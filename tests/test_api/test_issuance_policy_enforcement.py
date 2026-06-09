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
from app.services.issuance_policy import IssuancePolicyService
from app.services.organization import OrganizationService
from app.services.principal import Principal
from app.services.service_account import ServiceAccountService
from app.services.user import UserService
from tests.conftest import create_test_app, get_test_session


def _client_for(principal: Principal) -> AsyncClient:
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
    return await OrganizationService(db).create_organization(name="EnforceOrg")


@pytest_asyncio.fixture
async def ca(db, org):
    return await CAService(db).create_ca(
        name="Enforce CA", subject_dn="CN=Enforce CA", organization_id=org.id
    )


@pytest_asyncio.fixture
async def sa(db, org):
    return await ServiceAccountService(db).create_service_account(
        name="enf-sa", organization_id=org.id, can_create_cert=True
    )


async def _allow_policy(db, sa_id, ca_id, **overrides):
    kwargs = {
        "cn_patterns": ["*.example.com"],
        "san_dns_patterns": ["*.example.com"],
        "san_ip_cidrs": [],
        "san_email_domains": [],
        "allowed_ca_ids": [ca_id],
        "allowed_certificate_types": [CertificateType.SERVER],
        "max_validity_days": 90,
    }
    kwargs.update(overrides)
    return await IssuancePolicyService(db).set_policy(sa_id, **kwargs)


def _body(cn="svc.example.com"):
    return {
        "common_name": cn,
        "subject_dn": f"CN={cn}",
        "certificate_type": CertificateType.SERVER.value,
        "valid_days": 30,
    }


@pytest.mark.asyncio
async def test_service_account_without_policy_denied(sa, ca):
    async with _client_for(Principal.from_service_account(sa)) as c:
        resp = await c.post(f"/api/v1/certificates/?ca_id={ca.id}", json=_body())
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "service_account_has_no_policy"


@pytest.mark.asyncio
async def test_compliant_request_allowed(db, sa, ca):
    await _allow_policy(db, sa.id, ca.id)
    async with _client_for(Principal.from_service_account(sa)) as c:
        resp = await c.post(f"/api/v1/certificates/?ca_id={ca.id}", json=_body())
    assert resp.status_code == 201, resp.text


@pytest.mark.asyncio
async def test_noncompliant_cn_rejected(db, sa, ca):
    await _allow_policy(db, sa.id, ca.id)
    async with _client_for(Principal.from_service_account(sa)) as c:
        resp = await c.post(
            f"/api/v1/certificates/?ca_id={ca.id}", json=_body(cn="svc.evil.com")
        )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["code"] == "policy_violation"
    assert detail["field"] == "cn_patterns"


@pytest.mark.asyncio
async def test_sign_csr_enforced(db, sa, ca):
    await _allow_policy(db, sa.id, ca.id)
    async with _client_for(Principal.from_service_account(sa)) as c:
        ok = await c.post(
            "/api/v1/certificates/sign-csr",
            json={
                "csr": _make_csr("good.example.com"),
                "ca_id": ca.id,
                "certificate_type": CertificateType.SERVER.value,
                "valid_days": 30,
            },
        )
        bad = await c.post(
            "/api/v1/certificates/sign-csr",
            json={
                "csr": _make_csr("bad.evil.com"),
                "ca_id": ca.id,
                "certificate_type": CertificateType.SERVER.value,
                "valid_days": 30,
            },
        )
    assert ok.status_code == 201, ok.text
    assert bad.status_code == 400
    assert bad.json()["detail"]["field"] == "cn_patterns"


@pytest.mark.asyncio
async def test_user_principal_bypasses_policy(db, org, ca):
    user = await UserService(db).create_user(
        username="bypass",
        email="bypass@example.com",
        password="password123",
        organization_id=org.id,
        can_create_cert=True,
    )
    # No policy exists anywhere; a user-bound principal must still issue.
    async with _client_for(Principal.from_user(user)) as c:
        resp = await c.post(
            f"/api/v1/certificates/?ca_id={ca.id}", json=_body(cn="anything.test")
        )
    assert resp.status_code == 201, resp.text
