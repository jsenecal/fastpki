import pytest
import pytest_asyncio

from app.db.models import CertificateType
from app.services.exceptions import (
    IssuancePolicyMissingError,
    PolicyViolationError,
)
from app.services.issuance_policy import IssuancePolicyService
from app.services.organization import OrganizationService
from app.services.service_account import ServiceAccountService


@pytest_asyncio.fixture
async def sa(db):
    org = await OrganizationService(db).create_organization(name="PolOrg")
    return await ServiceAccountService(db).create_service_account(
        name="pol-sa", organization_id=org.id, can_create_cert=True
    )


def _compliant_kwargs(ca_id):
    return {
        "cn_patterns": ["*.example.com"],
        "san_dns_patterns": ["*.example.com"],
        "san_ip_cidrs": ["10.0.0.0/8"],
        "san_email_domains": ["example.com"],
        "allowed_ca_ids": [ca_id],
        "allowed_certificate_types": [CertificateType.SERVER],
        "max_validity_days": 90,
    }


async def _set(db, sa_id, ca_id=1, **overrides):
    kwargs = _compliant_kwargs(ca_id)
    kwargs.update(overrides)
    return await IssuancePolicyService(db).set_policy(sa_id, **kwargs)


def _issuance(ca_id=1, **overrides):
    params = {
        "common_name": "svc.example.com",
        "san_dns_names": ["svc.example.com"],
        "san_ip_addresses": ["10.1.2.3"],
        "san_email_addresses": ["a@example.com"],
        "ca_id": ca_id,
        "certificate_type": CertificateType.SERVER,
        "valid_days": 30,
    }
    params.update(overrides)
    return params


@pytest.mark.asyncio
async def test_set_get_delete_policy(db, sa):
    service = IssuancePolicyService(db)
    await _set(db, sa.id, ca_id=7)
    fetched = await service.get_policy(sa.id)
    assert fetched is not None
    assert fetched.allowed_ca_ids == [7]

    # set_policy is create-or-replace.
    await _set(db, sa.id, ca_id=7, max_validity_days=10)
    again = await service.get_policy(sa.id)
    assert again.max_validity_days == 10

    await service.delete_policy(sa.id)
    assert await service.get_policy(sa.id) is None


@pytest.mark.asyncio
async def test_evaluate_passes_when_compliant(db, sa):
    policy = await _set(db, sa.id)
    # Should not raise.
    IssuancePolicyService.evaluate(policy, **_issuance())


@pytest.mark.asyncio
async def test_enforce_without_policy_raises(db, sa):
    with pytest.raises(IssuancePolicyMissingError):
        await IssuancePolicyService(db).enforce(sa.id, **_issuance())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override,field",
    [
        ({"certificate_type": CertificateType.CLIENT}, "allowed_certificate_types"),
        ({"ca_id": 999}, "allowed_ca_ids"),
        ({"common_name": "svc.evil.com"}, "cn_patterns"),
        ({"san_dns_names": ["svc.evil.com"]}, "san_dns_patterns"),
        ({"san_ip_addresses": ["192.168.1.1"]}, "san_ip_cidrs"),
        ({"san_email_addresses": ["a@evil.com"]}, "san_email_domains"),
        ({"valid_days": 365}, "max_validity_days"),
    ],
)
async def test_evaluate_rejects_each_field(db, sa, override, field):
    policy = await _set(db, sa.id)
    with pytest.raises(PolicyViolationError) as exc:
        IssuancePolicyService.evaluate(policy, **_issuance(**override))
    assert exc.value.field == field


@pytest.mark.asyncio
async def test_empty_allowlist_denies(db, sa):
    # Deny-by-default: an empty CN allowlist rejects every CN.
    policy = await _set(db, sa.id, cn_patterns=[])
    with pytest.raises(PolicyViolationError) as exc:
        IssuancePolicyService.evaluate(policy, **_issuance())
    assert exc.value.field == "cn_patterns"


@pytest.mark.asyncio
async def test_cidr_boundary(db, sa):
    policy = await _set(db, sa.id, san_ip_cidrs=["10.0.0.0/30"])  # .0-.3
    IssuancePolicyService.evaluate(policy, **_issuance(san_ip_addresses=["10.0.0.3"]))
    with pytest.raises(PolicyViolationError):
        IssuancePolicyService.evaluate(
            policy, **_issuance(san_ip_addresses=["10.0.0.4"])
        )
