from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Certificate,
    CertificateAuthority,
    PermissionAction,
    User,
    UserRole,
)
from app.services.exceptions import NotFoundError, PermissionDeniedError
from app.services.principal import Principal

CAPABILITY_MAP: dict[PermissionAction, str] = {
    PermissionAction.CREATE_CA: "can_create_ca",
    PermissionAction.CREATE_CERT: "can_create_cert",
    PermissionAction.REVOKE_CERT: "can_revoke_cert",
    PermissionAction.EXPORT_PRIVATE_KEY: "can_export_private_key",
    PermissionAction.DELETE_CA: "can_delete_ca",
}


def _as_principal(actor: Principal | User) -> Principal:
    """Coerce a User into a Principal; pass a Principal through unchanged."""
    if isinstance(actor, Principal):
        return actor
    return Principal.from_user(actor)


class PermissionService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def check_ca_access(
        self, actor: Principal | User, ca_id: int, action: PermissionAction
    ) -> CertificateAuthority:
        ca = await self.db.get(CertificateAuthority, ca_id)
        if not ca:
            raise NotFoundError(f"Certificate Authority with ID {ca_id} not found")  # noqa: TRY003
        if not self._can_perform(_as_principal(actor), ca.organization_id, action):
            raise PermissionDeniedError("Insufficient permissions")  # noqa: TRY003
        return ca

    async def check_cert_access(
        self, actor: Principal | User, cert_id: int, action: PermissionAction
    ) -> Certificate:
        cert = await self.db.get(Certificate, cert_id)
        if not cert:
            raise NotFoundError(f"Certificate with ID {cert_id} not found")  # noqa: TRY003
        if not self._can_perform(_as_principal(actor), cert.organization_id, action):
            raise PermissionDeniedError("Insufficient permissions")  # noqa: TRY003
        return cert

    def can_create_in_org(
        self,
        actor: Principal | User,
        organization_id: int | None,
        action: PermissionAction,
    ) -> bool:
        """Whether the actor may create a new resource in the given org.

        Used on creation paths where no resource exists yet.
        """
        return self._can_perform(_as_principal(actor), organization_id, action)

    @staticmethod
    def _can_perform(
        principal: Principal,
        resource_org_id: int | None,
        action: PermissionAction,
    ) -> bool:
        # Authorization derives strictly from the principal's *current* role,
        # organization, and capability flags. Creator attribution
        # (`created_by_user_id` / `created_by_service_account_id`) is kept for
        # audit only and grants nothing, so demotions, capability revocations,
        # and org moves take effect on previously created resources (#7).

        # 1. SUPERUSER -> always allowed
        if principal.role == UserRole.SUPERUSER:
            return True

        # 2. Unowned resource (org_id=None) -> superuser-only
        if resource_org_id is None:
            return False

        # 3. Wrong org -> denied
        if principal.organization_id != resource_org_id:
            return False

        # 4. ADMIN in same org -> full access
        if principal.role == UserRole.ADMIN:
            return True

        # 5. USER/service account with READ -> allowed in same org
        if action == PermissionAction.READ:
            return True

        # 6. Check capability flag for write actions
        cap_attr = CAPABILITY_MAP.get(action)
        return bool(cap_attr and getattr(principal, cap_attr, False))
