# mypy: disable-error-code="arg-type"
import fnmatch
import ipaddress
from zoneinfo import ZoneInfo

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import CertificateType, IssuancePolicy
from app.services.exceptions import (
    IssuancePolicyMissingError,
    PolicyViolationError,
)

UTC = ZoneInfo("UTC")


class IssuancePolicyService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_policy(self, service_account_id: int) -> IssuancePolicy | None:
        result = await self.db.execute(
            select(IssuancePolicy).where(
                IssuancePolicy.service_account_id == service_account_id
            )
        )
        return result.scalar_one_or_none()

    async def set_policy(
        self,
        service_account_id: int,
        *,
        cn_patterns: list[str],
        san_dns_patterns: list[str],
        san_ip_cidrs: list[str],
        san_email_domains: list[str],
        allowed_ca_ids: list[int],
        allowed_certificate_types: list[CertificateType],
        max_validity_days: int,
    ) -> IssuancePolicy:
        """Create or replace the policy attached to a service account."""
        policy = await self.get_policy(service_account_id)
        if policy is None:
            policy = IssuancePolicy(
                service_account_id=service_account_id,
                max_validity_days=max_validity_days,
            )
        policy.cn_patterns = cn_patterns
        policy.san_dns_patterns = san_dns_patterns
        policy.san_ip_cidrs = san_ip_cidrs
        policy.san_email_domains = san_email_domains
        policy.allowed_ca_ids = allowed_ca_ids
        policy.allowed_certificate_types = [t.value for t in allowed_certificate_types]
        policy.max_validity_days = max_validity_days
        self.db.add(policy)
        await self.db.commit()
        await self.db.refresh(policy)
        return policy

    async def delete_policy(self, service_account_id: int) -> None:
        await self.db.execute(
            delete(IssuancePolicy).where(
                IssuancePolicy.service_account_id == service_account_id
            )
        )
        await self.db.commit()

    async def enforce(
        self,
        service_account_id: int,
        *,
        common_name: str,
        san_dns_names: list[str] | None,
        san_ip_addresses: list[str] | None,
        san_email_addresses: list[str] | None,
        ca_id: int,
        certificate_type: CertificateType,
        valid_days: int | None,
    ) -> None:
        """Load the policy and evaluate the request, raising on any violation."""
        policy = await self.get_policy(service_account_id)
        if policy is None:
            raise IssuancePolicyMissingError(  # noqa: TRY003
                "Service account has no issuance policy"
            )
        self.evaluate(
            policy,
            common_name=common_name,
            san_dns_names=san_dns_names,
            san_ip_addresses=san_ip_addresses,
            san_email_addresses=san_email_addresses,
            ca_id=ca_id,
            certificate_type=certificate_type,
            valid_days=valid_days,
        )

    @staticmethod
    def evaluate(
        policy: IssuancePolicy,
        *,
        common_name: str,
        san_dns_names: list[str] | None,
        san_ip_addresses: list[str] | None,
        san_email_addresses: list[str] | None,
        ca_id: int,
        certificate_type: CertificateType,
        valid_days: int | None,
    ) -> None:
        """Evaluate an issuance request against a policy (deny-by-default).

        Each constraint is checked independently; the first failure raises a
        ``PolicyViolationError`` naming the field and the offending value.
        """
        if certificate_type.value not in policy.allowed_certificate_types:
            raise PolicyViolationError(
                "allowed_certificate_types", certificate_type.value
            )

        if ca_id not in policy.allowed_ca_ids:
            raise PolicyViolationError("allowed_ca_ids", ca_id)

        if not _matches_glob(common_name, policy.cn_patterns):
            raise PolicyViolationError("cn_patterns", common_name)

        for dns in san_dns_names or []:
            if not _matches_glob(dns, policy.san_dns_patterns):
                raise PolicyViolationError("san_dns_patterns", dns)

        for ip in san_ip_addresses or []:
            if not _ip_in_cidrs(ip, policy.san_ip_cidrs):
                raise PolicyViolationError("san_ip_cidrs", ip)

        for email in san_email_addresses or []:
            domain = email.rsplit("@", 1)[-1].lower()
            allowed = {d.lower() for d in policy.san_email_domains}
            if domain not in allowed:
                raise PolicyViolationError("san_email_domains", email)

        effective_validity = (
            valid_days if valid_days is not None else settings.CERT_DAYS
        )
        if effective_validity > policy.max_validity_days:
            raise PolicyViolationError("max_validity_days", effective_validity)


def _matches_glob(value: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(value, pattern) for pattern in patterns)


def _ip_in_cidrs(ip: str, cidrs: list[str]) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for cidr in cidrs:
        try:
            if addr in ipaddress.ip_network(cidr, strict=False):
                return True
        except ValueError:
            continue
    return False
