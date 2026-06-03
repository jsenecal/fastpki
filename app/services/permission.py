from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Certificate,
    CertificateAuthority,
    PermissionAction,
    User,
    UserRole,
)
from app.services.exceptions import NotFoundError, PermissionDeniedError

CAPABILITY_MAP: dict[PermissionAction, str] = {
    PermissionAction.CREATE_CA: "can_create_ca",
    PermissionAction.CREATE_CERT: "can_create_cert",
    PermissionAction.REVOKE_CERT: "can_revoke_cert",
    PermissionAction.EXPORT_PRIVATE_KEY: "can_export_private_key",
    PermissionAction.DELETE_CA: "can_delete_ca",
}


class PermissionService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def check_ca_access(
        self, user: User, ca_id: int, action: PermissionAction
    ) -> CertificateAuthority:
        ca = await self.db.get(CertificateAuthority, ca_id)
        if not ca:
            raise NotFoundError(f"Certificate Authority with ID {ca_id} not found")  # noqa: TRY003
        if not self._user_can_perform(user, ca.organization_id, action):
            raise PermissionDeniedError("Insufficient permissions")  # noqa: TRY003
        return ca

    async def check_cert_access(
        self, user: User, cert_id: int, action: PermissionAction
    ) -> Certificate:
        cert = await self.db.get(Certificate, cert_id)
        if not cert:
            raise NotFoundError(f"Certificate with ID {cert_id} not found")  # noqa: TRY003
        if not self._user_can_perform(user, cert.organization_id, action):
            raise PermissionDeniedError("Insufficient permissions")  # noqa: TRY003
        return cert

    @staticmethod
    def _user_can_perform(
        user: User,
        resource_org_id: int | None,
        action: PermissionAction,
    ) -> bool:
        # Creator status (created_by_user_id) is retained for audit only;
        # authorization derives strictly from role, org, and capability flags
        # so that demotions, capability revocations, and org moves take effect
        # for previously created resources (issue #7).

        if user.role == UserRole.SUPERUSER:
            return True

        if resource_org_id is None:
            return False

        if user.organization_id != resource_org_id:
            return False

        if user.role == UserRole.ADMIN:
            return True

        if action == PermissionAction.READ:
            return True

        cap_attr = CAPABILITY_MAP.get(action)
        return bool(cap_attr and getattr(user, cap_attr, False))
