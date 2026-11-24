# mypy: disable-error-code="arg-type"
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from app.db.models import Organization, User, UserRole
from app.services.exceptions import (
    AlreadyExistsError,
    HasDependentsError,
    NotFoundError,
    PermissionDeniedError,
)

_ORG_NOT_FOUND = "Organization not found"
_ORG_EXISTS = "Organization with this name already exists"
_USER_NOT_FOUND = "User not found"
_NO_ADD_PERM = "You don't have permission to add users to this org"
_NO_REMOVE_PERM = "You don't have permission to remove this user"
_HAS_USERS = "Cannot delete organization with users. Remove all users first."


class OrganizationService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_organization(
        self, name: str, description: str | None = None
    ) -> Organization:
        """Create a new organization."""
        exists = await self.get_organization_by_name(name)
        if exists:
            raise AlreadyExistsError(_ORG_EXISTS)

        org = Organization(
            name=name,
            description=description,
        )

        self.db.add(org)
        await self.db.commit()
        await self.db.refresh(org)
        return org

    async def get_organization_by_id(self, org_id: int) -> Organization | None:
        """Get an organization by ID."""
        query = select(Organization).where(Organization.id == org_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_organization_by_name(self, name: str) -> Organization | None:
        """Get an organization by name."""
        query = select(Organization).where(Organization.name == name)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_all_organizations(self) -> list[Organization]:
        """Get all organizations."""
        query = select(Organization).order_by(Organization.name)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def update_organization(
        self,
        org_id: int,
        name: str | None = None,
        description: str | None = None,
    ) -> Organization:
        """Update an organization."""
        org = await self.get_organization_by_id(org_id)
        if not org:
            raise NotFoundError(_ORG_NOT_FOUND)

        if name and name != org.name:
            exists = await self.get_organization_by_name(name)
            if exists:
                raise AlreadyExistsError(_ORG_EXISTS)
            org.name = name

        if description is not None:
            org.description = description

        self.db.add(org)
        await self.db.commit()
        await self.db.refresh(org)
        return org

    async def delete_organization(self, org_id: int) -> bool:
        """Delete an organization."""
        org = await self.get_organization_by_id(org_id)
        if not org:
            raise NotFoundError(_ORG_NOT_FOUND)

        users = await self.get_organization_users(org_id)
        if users:
            raise HasDependentsError(_HAS_USERS)

        await self.db.delete(org)
        await self.db.commit()
        return True

    async def add_user_to_organization(
        self, user_id: int, org_id: int, admin_user_id: int | None = None
    ) -> User:
        """Add a user to an organization."""
        org = await self.get_organization_by_id(org_id)
        if not org:
            raise NotFoundError(_ORG_NOT_FOUND)

        query = select(User).where(User.id == user_id)
        result = await self.db.execute(query)
        user = result.scalar_one_or_none()

        if not user:
            raise NotFoundError(_USER_NOT_FOUND)

        if admin_user_id:
            can_add = await self.user_can_add_user_to_organization(
                admin_user_id, org_id, user_id
            )
            if not can_add:
                raise PermissionDeniedError(_NO_ADD_PERM)

        user.organization_id = org_id

        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def remove_user_from_organization(
        self, user_id: int, admin_user_id: int | None = None
    ) -> User:
        """Remove a user from their organization."""
        query = select(User).where(User.id == user_id)
        result = await self.db.execute(query)
        user = result.scalar_one_or_none()

        if not user:
            raise NotFoundError(_USER_NOT_FOUND)

        if user.organization_id is None:
            return user

        if admin_user_id:
            can_remove = await self.user_can_remove_user_from_organization(
                admin_user_id, user_id
            )
            if not can_remove:
                raise PermissionDeniedError(_NO_REMOVE_PERM)

        user.organization_id = None

        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def get_organization_users(self, org_id: int) -> list[User]:
        """Get all users in an organization."""
        query = select(User).where(User.organization_id == org_id)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_organization_user_count(self, org_id: int) -> int:
        """Get the count of users in an organization."""
        query = select(func.count(User.id)).where(User.organization_id == org_id)
        result = await self.db.execute(query)
        return result.scalar_one()

    async def user_has_organization_access(self, user_id: int, org_id: int) -> bool:
        """Check if a user has access to an organization."""
        query = select(User).where(User.id == user_id)
        result = await self.db.execute(query)
        user = result.scalar_one_or_none()

        if not user:
            return False

        if user.role == UserRole.SUPERUSER:
            return True

        return user.organization_id == org_id

    async def user_has_organization_admin_access(
        self, user_id: int, org_id: int
    ) -> bool:
        """
        Check if a user has admin access to an organization.

        A user has admin access if:
        1. They are a superuser
        2. They are an admin and belong to the organization
        """
        query = select(User).where(User.id == user_id)
        result = await self.db.execute(query)
        user = result.scalar_one_or_none()

        if not user:
            return False

        if user.role == UserRole.SUPERUSER:
            return True

        return user.role == UserRole.ADMIN and user.organization_id == org_id

    async def user_can_add_user_to_organization(
        self, admin_user_id: int, org_id: int, user_id: int
    ) -> bool:
        """Check if a user can add another user to an organization."""
        query = select(User).where(User.id == admin_user_id)
        result = await self.db.execute(query)
        admin_user = result.scalar_one_or_none()

        if not admin_user:
            return False

        if admin_user.role == UserRole.SUPERUSER:
            return True

        if not (
            admin_user.role == UserRole.ADMIN and admin_user.organization_id == org_id
        ):
            return False

        # An org admin may only bring in unassigned users (or re-add their
        # own members); they must not pull another tenant's user, or a
        # superuser, into their organization (issue #69). Only USER-role
        # accounts qualify: granting admin rights over an org is a superuser
        # decision.
        target = await self.db.get(User, user_id)
        if target is None:
            return False
        if target.id == admin_user.id:
            return True
        return target.role == UserRole.USER and target.organization_id in (
            None,
            org_id,
        )

    async def user_can_remove_user_from_organization(
        self, admin_user_id: int, user_id: int
    ) -> bool:
        """Check if a user can remove another user from an organization."""
        admin_query = select(User).where(User.id == admin_user_id)
        user_query = select(User).where(User.id == user_id)

        admin_result = await self.db.execute(admin_query)
        user_result = await self.db.execute(user_query)

        admin_user = admin_result.scalar_one_or_none()
        user = user_result.scalar_one_or_none()

        if not admin_user or not user or user.organization_id is None:
            return False

        if admin_user.role == UserRole.SUPERUSER:
            return True

        # An org admin removes USER-role members and may leave themselves;
        # removing a peer admin requires a superuser.
        return (
            admin_user.role == UserRole.ADMIN
            and admin_user.organization_id == user.organization_id
            and (user.role == UserRole.USER or user.id == admin_user.id)
        )
