import pytest

from app.db.models import UserRole
from app.services.exceptions import HasDependentsError
from app.services.organization import OrganizationService
from app.services.user import UserService


@pytest.mark.asyncio
async def test_role_based_organization_access(db):
    # Create users with different roles
    user_service = UserService(db)
    org_service = OrganizationService(db)

    superuser = await user_service.create_user(
        username="org_superuser",
        email="org_superuser@example.com",
        password="password123",
        role=UserRole.SUPERUSER,
    )

    admin = await user_service.create_user(
        username="org_admin",
        email="org_admin@example.com",
        password="password123",
        role=UserRole.ADMIN,
    )

    regular_user = await user_service.create_user(
        username="org_user",
        email="org_user@example.com",
        password="password123",
        role=UserRole.USER,
    )

    # Create an organization
    org = await org_service.create_organization(
        name="Role Test Org", description="Organization for testing role-based access"
    )

    # Add the admin to the organization
    await org_service.add_user_to_organization(admin.id, org.id)

    # Add the regular user to the organization
    await org_service.add_user_to_organization(regular_user.id, org.id)

    # Test role-based permissions
    # Superuser should have access to all organizations
    has_access = await org_service.user_has_organization_access(superuser.id, org.id)
    assert has_access is True

    # Admin should have access to their organization
    has_access = await org_service.user_has_organization_access(admin.id, org.id)
    assert has_access is True

    # Regular user should have access to their organization but not admin capabilities
    has_access = await org_service.user_has_organization_access(regular_user.id, org.id)
    assert has_access is True

    # Regular user should not have admin capabilities
    has_admin_access = await org_service.user_has_organization_admin_access(
        regular_user.id, org.id
    )
    assert has_admin_access is False

    # Admin user should have admin capabilities
    has_admin_access = await org_service.user_has_organization_admin_access(
        admin.id, org.id
    )
    assert has_admin_access is True

    # Superuser should have admin capabilities for all organizations
    has_admin_access = await org_service.user_has_organization_admin_access(
        superuser.id, org.id
    )
    assert has_admin_access is True


@pytest.mark.asyncio
async def test_user_organization_relationship(db):
    # Test user-organization relationships
    user_service = UserService(db)
    org_service = OrganizationService(db)

    # Create two organizations
    org1 = await org_service.create_organization(
        name="Org One", description="First test organization"
    )

    org2 = await org_service.create_organization(
        name="Org Two", description="Second test organization"
    )

    # Create a user
    user = await user_service.create_user(
        username="multiorg_user", email="multiorg@example.com", password="password123"
    )

    # A user can only be in one organization at a time
    # Add user to first organization
    await org_service.add_user_to_organization(user.id, org1.id)

    # Verify user is in first organization
    updated_user = await user_service.get_user_by_id(user.id)
    assert updated_user.organization_id == org1.id

    # Move user to second organization
    await org_service.add_user_to_organization(user.id, org2.id)

    # Verify user is now in second organization, not the first
    updated_user = await user_service.get_user_by_id(user.id)
    assert updated_user.organization_id == org2.id

    # Verify organization users list is updated correctly
    org1_users = await org_service.get_organization_users(org1.id)
    org2_users = await org_service.get_organization_users(org2.id)

    assert user.id not in [u.id for u in org1_users]
    assert user.id in [u.id for u in org2_users]


@pytest.mark.asyncio
async def test_cannot_delete_organization_with_users(db):
    # Test that an organization with users cannot be deleted
    user_service = UserService(db)
    org_service = OrganizationService(db)

    # Create an organization
    org = await org_service.create_organization(
        name="Non-Empty Org",
        description="Organization with users that should not be deleted",
    )

    # Create a user and add to the organization
    user = await user_service.create_user(
        username="nodelete_user", email="nodelete@example.com", password="password123"
    )

    await org_service.add_user_to_organization(user.id, org.id)

    # Attempt to delete the organization should raise HasDependentsError
    with pytest.raises(
        HasDependentsError, match="Cannot delete organization with users"
    ):
        await org_service.delete_organization(org.id)

    # Verify the organization still exists
    org_check = await org_service.get_organization_by_id(org.id)
    assert org_check is not None


@pytest.mark.asyncio
async def test_organization_admin_permissions(db):
    # Test that organization admin can manage users in their organization
    user_service = UserService(db)
    org_service = OrganizationService(db)

    # Create an organization
    org = await org_service.create_organization(
        name="Admin Org", description="Organization for testing admin permissions"
    )

    # Create an admin user
    admin = await user_service.create_user(
        username="org_admin_test",
        email="org_admin_test@example.com",
        password="password123",
        role=UserRole.ADMIN,
    )

    # Add admin to the organization
    await org_service.add_user_to_organization(admin.id, org.id)

    # Create a regular user
    user = await user_service.create_user(
        username="org_regular_user",
        email="org_regular@example.com",
        password="password123",
        role=UserRole.USER,
    )

    # Admin should be able to add users to their organization
    can_add = await org_service.user_can_add_user_to_organization(
        admin.id, org.id, user.id
    )
    assert can_add is True

    # Actually add the user
    await org_service.add_user_to_organization(user.id, org.id, admin_user_id=admin.id)

    # Verify user was added
    updated_user = await user_service.get_user_by_id(user.id)
    assert updated_user.organization_id == org.id

    # Admin should be able to remove users from their organization
    can_remove = await org_service.user_can_remove_user_from_organization(
        admin.id, user.id
    )
    assert can_remove is True

    # Actually remove the user
    await org_service.remove_user_from_organization(user.id, admin_user_id=admin.id)

    # Verify user was removed
    updated_user = await user_service.get_user_by_id(user.id)
    assert updated_user.organization_id is None


@pytest.mark.asyncio
async def test_org_admin_membership_limited_to_user_role_and_self(db):
    """An org admin may add/remove USER-role accounts and themselves, but not
    peer admins: that takes a superuser."""
    user_service = UserService(db)
    org_service = OrganizationService(db)
    org = await org_service.create_organization(name="PeerAdminOrg")

    admin = await user_service.create_user(
        username="peer_rule_admin",
        email="peer_rule_admin@example.com",
        password="password123",
        role=UserRole.ADMIN,
        organization_id=org.id,
    )
    peer = await user_service.create_user(
        username="peer_rule_other",
        email="peer_rule_other@example.com",
        password="password123",
        role=UserRole.ADMIN,
        organization_id=org.id,
    )
    loose_admin = await user_service.create_user(
        username="peer_rule_loose",
        email="peer_rule_loose@example.com",
        password="password123",
        role=UserRole.ADMIN,
    )

    can_add = org_service.user_can_add_user_to_organization
    can_remove = org_service.user_can_remove_user_from_organization

    assert await can_add(admin.id, org.id, admin.id) is True
    assert await can_add(admin.id, org.id, peer.id) is False
    assert await can_add(admin.id, org.id, loose_admin.id) is False
    assert await can_add(admin.id, org.id, 99999) is False

    assert await can_remove(admin.id, peer.id) is False
    assert await can_remove(admin.id, admin.id) is True
