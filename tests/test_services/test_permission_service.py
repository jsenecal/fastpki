import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CertificateType, PermissionAction, UserRole
from app.services.ca import CAService
from app.services.cert import CertificateService
from app.services.exceptions import NotFoundError, PermissionDeniedError
from app.services.organization import OrganizationService
from app.services.permission import PermissionService
from app.services.user import UserService


@pytest_asyncio.fixture
async def org(db: AsyncSession):
    org_service = OrganizationService(db)
    return await org_service.create_organization(name="PermOrg", description="Test")


@pytest_asyncio.fixture
async def other_org(db: AsyncSession):
    org_service = OrganizationService(db)
    return await org_service.create_organization(name="OtherOrg", description="Other")


@pytest_asyncio.fixture
async def su(db: AsyncSession):
    user_service = UserService(db)
    return await user_service.create_user(
        username="permsu",
        email="permsu@example.com",
        password="password123",
        role=UserRole.SUPERUSER,
    )


@pytest_asyncio.fixture
async def admin_in_org(db: AsyncSession, org):
    user_service = UserService(db)
    return await user_service.create_user(
        username="permadmin",
        email="permadmin@example.com",
        password="password123",
        role=UserRole.ADMIN,
        organization_id=org.id,
    )


@pytest_asyncio.fixture
async def user_in_org(db: AsyncSession, org):
    user_service = UserService(db)
    return await user_service.create_user(
        username="permuser",
        email="permuser@example.com",
        password="password123",
        role=UserRole.USER,
        organization_id=org.id,
    )


@pytest_asyncio.fixture
async def user_in_other_org(db: AsyncSession, other_org):
    user_service = UserService(db)
    return await user_service.create_user(
        username="otheruser",
        email="otheruser@example.com",
        password="password123",
        role=UserRole.USER,
        organization_id=other_org.id,
    )


@pytest_asyncio.fixture
async def org_ca(db: AsyncSession, org, user_in_org):
    ca_service = CAService(db)
    return await ca_service.create_ca(
        name="Perm CA",
        subject_dn="CN=Perm CA",
        organization_id=org.id,
        created_by_user_id=user_in_org.id,
    )


@pytest_asyncio.fixture
async def unowned_ca(db: AsyncSession):
    ca_service = CAService(db)
    return await ca_service.create_ca(
        name="Unowned CA",
        subject_dn="CN=Unowned CA",
    )


@pytest_asyncio.fixture
async def org_cert(db: AsyncSession, org_ca, org, user_in_org):
    cert_service = CertificateService(db)
    return await cert_service.create_certificate(
        ca_id=org_ca.id,
        common_name="perm.example.com",
        subject_dn="CN=perm.example.com",
        certificate_type=CertificateType.SERVER,
        organization_id=org.id,
        created_by_user_id=user_in_org.id,
    )


# --- check_ca_access tests ---


@pytest.mark.asyncio
async def test_superuser_always_allowed_ca(db: AsyncSession, su, org_ca):
    perm = PermissionService(db)
    for action in PermissionAction:
        ca = await perm.check_ca_access(su, org_ca.id, action)
        assert ca.id == org_ca.id


WRITE_ACTIONS = [a for a in PermissionAction if a != PermissionAction.READ]


@pytest.mark.asyncio
async def test_creator_without_capabilities_can_only_read_ca(
    db: AsyncSession, user_in_org, org_ca
):
    """Creator status must not bypass capability checks (issue #7)."""
    assert org_ca.created_by_user_id == user_in_org.id
    perm = PermissionService(db)

    ca = await perm.check_ca_access(user_in_org, org_ca.id, PermissionAction.READ)
    assert ca.id == org_ca.id

    for action in WRITE_ACTIONS:
        with pytest.raises(PermissionDeniedError):
            await perm.check_ca_access(user_in_org, org_ca.id, action)


@pytest.mark.asyncio
async def test_demoted_creator_loses_write_access_ca(db: AsyncSession, org, org_ca):
    """An admin who created a CA loses write access once demoted (issue #7)."""
    user_service = UserService(db)
    creator = await user_service.create_user(
        username="demotedcreator",
        email="demotedcreator@example.com",
        password="password123",
        role=UserRole.ADMIN,
        organization_id=org.id,
    )
    org_ca.created_by_user_id = creator.id
    db.add(org_ca)
    await db.commit()

    perm = PermissionService(db)
    await perm.check_ca_access(creator, org_ca.id, PermissionAction.DELETE_CA)

    creator.role = UserRole.USER
    db.add(creator)
    await db.commit()

    with pytest.raises(PermissionDeniedError):
        await perm.check_ca_access(creator, org_ca.id, PermissionAction.DELETE_CA)
    with pytest.raises(PermissionDeniedError):
        await perm.check_ca_access(
            creator, org_ca.id, PermissionAction.EXPORT_PRIVATE_KEY
        )


@pytest.mark.asyncio
async def test_creator_loses_access_after_org_move_ca(
    db: AsyncSession, other_org, org_ca
):
    """A creator moved to a different org loses access to their old resources."""
    user_service = UserService(db)
    moved_creator = await user_service.create_user(
        username="movedcreator",
        email="movedcreator@example.com",
        password="password123",
        role=UserRole.ADMIN,
        organization_id=other_org.id,
        can_export_private_key=True,
    )
    # Pretend this user originally created the CA before being moved.
    org_ca.created_by_user_id = moved_creator.id
    db.add(org_ca)
    await db.commit()

    perm = PermissionService(db)
    for action in PermissionAction:
        with pytest.raises(PermissionDeniedError):
            await perm.check_ca_access(moved_creator, org_ca.id, action)


@pytest.mark.asyncio
async def test_unowned_resource_denied_for_non_superuser_ca(
    db: AsyncSession, admin_in_org, unowned_ca
):
    perm = PermissionService(db)
    with pytest.raises(PermissionDeniedError):
        await perm.check_ca_access(admin_in_org, unowned_ca.id, PermissionAction.READ)


@pytest.mark.asyncio
async def test_superuser_can_access_unowned_ca(db: AsyncSession, su, unowned_ca):
    perm = PermissionService(db)
    ca = await perm.check_ca_access(su, unowned_ca.id, PermissionAction.READ)
    assert ca.id == unowned_ca.id


@pytest.mark.asyncio
async def test_wrong_org_denied_ca(db: AsyncSession, user_in_other_org, org_ca):
    perm = PermissionService(db)
    with pytest.raises(PermissionDeniedError):
        await perm.check_ca_access(user_in_other_org, org_ca.id, PermissionAction.READ)


@pytest.mark.asyncio
async def test_admin_full_access_in_org_ca(db: AsyncSession, admin_in_org, org_ca):
    perm = PermissionService(db)
    for action in PermissionAction:
        ca = await perm.check_ca_access(admin_in_org, org_ca.id, action)
        assert ca.id == org_ca.id


@pytest.mark.asyncio
async def test_user_read_allowed_in_org_ca(
    db: AsyncSession, org, org_ca, db_session_for_reader=None
):
    user_service = UserService(db)
    reader = await user_service.create_user(
        username="reader",
        email="reader@example.com",
        password="password123",
        role=UserRole.USER,
        organization_id=org.id,
    )
    perm = PermissionService(db)
    ca = await perm.check_ca_access(reader, org_ca.id, PermissionAction.READ)
    assert ca.id == org_ca.id


@pytest.mark.asyncio
async def test_user_write_denied_without_capability_ca(db: AsyncSession, org, org_ca):
    user_service = UserService(db)
    no_caps = await user_service.create_user(
        username="nocaps",
        email="nocaps@example.com",
        password="password123",
        role=UserRole.USER,
        organization_id=org.id,
    )
    perm = PermissionService(db)
    with pytest.raises(PermissionDeniedError):
        await perm.check_ca_access(no_caps, org_ca.id, PermissionAction.CREATE_CA)


@pytest.mark.asyncio
async def test_user_write_allowed_with_capability_ca(db: AsyncSession, org, org_ca):
    user_service = UserService(db)
    cap_user = await user_service.create_user(
        username="capuser",
        email="capuser@example.com",
        password="password123",
        role=UserRole.USER,
        organization_id=org.id,
        can_create_ca=True,
    )
    perm = PermissionService(db)
    ca = await perm.check_ca_access(cap_user, org_ca.id, PermissionAction.CREATE_CA)
    assert ca.id == org_ca.id


@pytest.mark.asyncio
async def test_ca_not_found(db: AsyncSession, su):
    perm = PermissionService(db)
    with pytest.raises(NotFoundError):
        await perm.check_ca_access(su, 9999, PermissionAction.READ)


# --- check_cert_access tests ---


@pytest.mark.asyncio
async def test_superuser_always_allowed_cert(db: AsyncSession, su, org_cert):
    perm = PermissionService(db)
    for action in PermissionAction:
        cert = await perm.check_cert_access(su, org_cert.id, action)
        assert cert.id == org_cert.id


@pytest.mark.asyncio
async def test_creator_without_capabilities_can_only_read_cert(
    db: AsyncSession, user_in_org, org_cert
):
    """Creator of a certificate cannot revoke/export it without the flag."""
    assert org_cert.created_by_user_id == user_in_org.id
    perm = PermissionService(db)

    cert = await perm.check_cert_access(user_in_org, org_cert.id, PermissionAction.READ)
    assert cert.id == org_cert.id

    for action in WRITE_ACTIONS:
        with pytest.raises(PermissionDeniedError):
            await perm.check_cert_access(user_in_org, org_cert.id, action)


@pytest.mark.asyncio
async def test_creator_without_capabilities_denied_export_key_cert(
    db: AsyncSession, user_in_org, org_cert
):
    """EXPORT_PRIVATE_KEY must require the explicit capability even for creators."""
    perm = PermissionService(db)
    with pytest.raises(PermissionDeniedError):
        await perm.check_cert_access(
            user_in_org, org_cert.id, PermissionAction.EXPORT_PRIVATE_KEY
        )


@pytest.mark.asyncio
async def test_creator_loses_access_after_org_move_cert(
    db: AsyncSession, other_org, org_cert
):
    user_service = UserService(db)
    moved_creator = await user_service.create_user(
        username="movedcertcreator",
        email="movedcertcreator@example.com",
        password="password123",
        role=UserRole.USER,
        organization_id=other_org.id,
        can_revoke_cert=True,
        can_export_private_key=True,
    )
    org_cert.created_by_user_id = moved_creator.id
    db.add(org_cert)
    await db.commit()

    perm = PermissionService(db)
    for action in PermissionAction:
        with pytest.raises(PermissionDeniedError):
            await perm.check_cert_access(moved_creator, org_cert.id, action)


@pytest.mark.asyncio
async def test_wrong_org_denied_cert(db: AsyncSession, user_in_other_org, org_cert):
    perm = PermissionService(db)
    with pytest.raises(PermissionDeniedError):
        await perm.check_cert_access(
            user_in_other_org, org_cert.id, PermissionAction.READ
        )


@pytest.mark.asyncio
async def test_cert_not_found(db: AsyncSession, su):
    perm = PermissionService(db)
    with pytest.raises(NotFoundError):
        await perm.check_cert_access(su, 9999, PermissionAction.READ)


# --- Capability flag mapping tests ---


@pytest.mark.asyncio
async def test_each_capability_maps_to_action(db: AsyncSession, org, org_ca):
    user_service = UserService(db)
    perm = PermissionService(db)

    caps_actions = [
        ("can_create_ca", PermissionAction.CREATE_CA),
        ("can_create_cert", PermissionAction.CREATE_CERT),
        ("can_revoke_cert", PermissionAction.REVOKE_CERT),
        ("can_export_private_key", PermissionAction.EXPORT_PRIVATE_KEY),
        ("can_delete_ca", PermissionAction.DELETE_CA),
    ]

    for i, (cap_field, action) in enumerate(caps_actions):
        kwargs = {cap_field: True}
        user = await user_service.create_user(
            username=f"cap_{i}",
            email=f"cap_{i}@example.com",
            password="password123",
            role=UserRole.USER,
            organization_id=org.id,
            **kwargs,
        )
        ca = await perm.check_ca_access(user, org_ca.id, action)
        assert ca.id == org_ca.id

        # Verify other write actions are denied
        for other_cap, other_action in caps_actions:
            if other_cap != cap_field:
                with pytest.raises(PermissionDeniedError):
                    await perm.check_ca_access(user, org_ca.id, other_action)
