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
        if not self._can_perform(
            _as_principal(actor),
            ca.organization_id,
            ca.created_by_user_id,
            ca.created_by_service_account_id,
            action,
        ):
            raise PermissionDeniedError("Insufficient permissions")  # noqa: TRY003
        return ca

    async def check_cert_access(
        self, actor: Principal | User, cert_id: int, action: PermissionAction
    ) -> Certificate:
        cert = await self.db.get(Certificate, cert_id)
        if not cert:
            raise NotFoundError(f"Certificate with ID {cert_id} not found")  # noqa: TRY003
        if not self._can_perform(
            _as_principal(actor),
            cert.organization_id,
            cert.created_by_user_id,
            cert.created_by_service_account_id,
            action,
        ):
            raise PermissionDeniedError("Insufficient permissions")  # noqa: TRY003
        return cert

    def can_create_in_org(
        self,
        actor: Principal | User,
        organization_id: int | None,
        action: PermissionAction,
    ) -> bool:
        """Whether the actor may create a new resource in the given org.

        Used on creation paths where no resource exists yet (so there is no
        creator to match against).
        """
        return self._can_perform(
            _as_principal(actor), organization_id, None, None, action
        )

    @staticmethod
    def _can_perform(
        principal: Principal,
        resource_org_id: int | None,
        creator_user_id: int | None,
        creator_service_account_id: int | None,
        action: PermissionAction,
    ) -> bool:
        # 1. SUPERUSER -> always allowed
        if principal.role == UserRole.SUPERUSER:
            return True

        # 2. Resource creator -> always allowed (matched per principal kind)
        if principal.kind == "user":
            if creator_user_id is not None and creator_user_id == principal.id:
                return True
        elif (
            creator_service_account_id is not None
            and creator_service_account_id == principal.id
        ):
            return True

        # 3. Unowned resource (org_id=None) -> superuser-only
        if resource_org_id is None:
            return False

        # 4. Wrong org -> denied
        if principal.organization_id != resource_org_id:
            return False

        # 5. ADMIN in same org -> full access
        if principal.role == UserRole.ADMIN:
            return True

        # 6. USER/service account with READ -> allowed in same org
        if action == PermissionAction.READ:
            return True

        # 7. Check capability flag for write actions
        cap_attr = CAPABILITY_MAP.get(action)
        return bool(cap_attr and getattr(principal, cap_attr, False))
