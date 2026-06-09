from dataclasses import dataclass
from typing import Literal

from app.db.models import ServiceAccount, User, UserRole


@dataclass(frozen=True)
class Principal:
    """A unified authenticated actor — a human user or a service account.

    Authorization (`app/services/permission.py`) operates on this value rather
    than on `User` directly, so user-bound and service-account-bound tokens
    flow through the same checks.
    """

    kind: Literal["user", "service_account"]
    id: int
    organization_id: int | None
    role: UserRole
    is_active: bool
    display_name: str
    can_create_ca: bool
    can_create_cert: bool
    can_revoke_cert: bool
    can_export_private_key: bool
    can_delete_ca: bool

    def creator_fields(self) -> dict[str, int]:
        """`created_by_*` kwargs for resource creation, keyed by principal kind."""
        if self.kind == "service_account":
            return {"created_by_service_account_id": self.id}
        return {"created_by_user_id": self.id}

    @classmethod
    def from_user(cls, user: User) -> "Principal":
        assert user.id is not None
        return cls(
            kind="user",
            id=user.id,
            organization_id=user.organization_id,
            role=user.role,
            is_active=user.is_active,
            display_name=user.username,
            can_create_ca=user.can_create_ca,
            can_create_cert=user.can_create_cert,
            can_revoke_cert=user.can_revoke_cert,
            can_export_private_key=user.can_export_private_key,
            can_delete_ca=user.can_delete_ca,
        )

    @classmethod
    def from_service_account(cls, sa: ServiceAccount) -> "Principal":
        assert sa.id is not None
        return cls(
            kind="service_account",
            id=sa.id,
            organization_id=sa.organization_id,
            # Service accounts are never SUPERUSER/ADMIN; they authorize purely
            # via org scope, resource ownership, and capability flags.
            role=UserRole.USER,
            is_active=sa.disabled_at is None,
            display_name=sa.name,
            can_create_ca=sa.can_create_ca,
            can_create_cert=sa.can_create_cert,
            can_revoke_cert=sa.can_revoke_cert,
            can_export_private_key=sa.can_export_private_key,
            can_delete_ca=sa.can_delete_ca,
        )
