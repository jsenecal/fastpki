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

ELEVATED_ROLE_DENIED = "Not enough permissions to create admin or superuser accounts"
CAPABILITIES_DENIED = "Not enough permissions to set capabilities"


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
    def authorize_user_creation(
        actor: User | None,
        role: UserRole,
        organization_id: int | None,
        *,
        is_first_user: bool,
        allow_unauthenticated_registration: bool,
        grants_capabilities: bool,
    ) -> int | None:
        """Decide whether ``actor`` may create a user and return its org.

        Returns the organization the new user must be placed in, or raises
        ``PermissionDeniedError``. This is a pure decision with no DB access
        so it can run before any uniqueness lookup (issue #8 ordering).

        ``grants_capabilities`` is true when the request sets any capability
        flag. Callers allowed to create the account (superusers, and admins
        in their own organization) may set flags; anonymous callers may not.

        - First user in the system: bootstrap, any role, org as requested.
        - Anonymous: USER role only, only when open registration is enabled,
          never with capability flags, and never into an organization (an org
          admin must add them).
        - SUPERUSER: any role, any organization.
        - ADMIN: USER role only, only in their own organization (defaulting
          to it when none is requested).
        - USER: may not create users.
        """
        if is_first_user:
            return organization_id

        elevated = role in (UserRole.ADMIN, UserRole.SUPERUSER)

        if actor is None:
            if elevated:
                raise PermissionDeniedError(ELEVATED_ROLE_DENIED)
            if not allow_unauthenticated_registration:
                raise PermissionDeniedError("Unauthenticated registration is disabled")  # noqa: TRY003
            if grants_capabilities:
                raise PermissionDeniedError(CAPABILITIES_DENIED)
            return None

        if actor.role == UserRole.SUPERUSER:
            return organization_id

        if actor.role != UserRole.ADMIN:
            raise PermissionDeniedError("Not enough permissions to create users")  # noqa: TRY003
        if elevated:
            raise PermissionDeniedError(ELEVATED_ROLE_DENIED)
        if actor.organization_id is None:
            raise PermissionDeniedError(  # noqa: TRY003
                "Admins without an organization cannot create users"
            )
        if organization_id is None:
            return actor.organization_id
        if organization_id != actor.organization_id:
            raise PermissionDeniedError(  # noqa: TRY003
                "Not enough permissions to create users in this organization"
            )
        return organization_id

    @staticmethod
    def can_view_user(
        actor: User, target: User, *, include_unassigned: bool = False
    ) -> bool:
        """Whether ``actor`` may know that ``target`` exists.

        Superusers see everyone and every user sees themselves. An ADMIN sees
        the accounts in their own organization. With ``include_unassigned``
        (adding members to an organization), an ADMIN with an organization
        also sees non-superuser accounts that belong to no organization.

        Endpoints answer ``404`` for an account the caller cannot view, the
        same as for a missing one, so user IDs cannot be enumerated across
        organizations.
        """
        if actor.role == UserRole.SUPERUSER or actor.id == target.id:
            return True
        if actor.role != UserRole.ADMIN or actor.organization_id is None:
            return False
        if target.organization_id == actor.organization_id:
            return True
        return (
            include_unassigned
            and target.organization_id is None
            and target.role != UserRole.SUPERUSER
        )

    @staticmethod
    def can_manage_user(actor: User, target: User) -> bool:
        """Whether ``actor`` may modify ``target``'s account.

        Superusers manage everyone and every user manages themselves. An
        ADMIN manages USER-role accounts in their own organization only;
        peer ADMIN accounts require a superuser, so one org admin cannot
        take over another's account.
        """
        if actor.role == UserRole.SUPERUSER or actor.id == target.id:
            return True
        return (
            actor.role == UserRole.ADMIN
            and actor.organization_id is not None
            and target.organization_id == actor.organization_id
            and target.role == UserRole.USER
        )

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
