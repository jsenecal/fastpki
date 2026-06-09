import pytest
import pytest_asyncio

from app.db.models import PermissionAction, UserRole
from app.services.ca import CAService
from app.services.exceptions import PermissionDeniedError
from app.services.organization import OrganizationService
from app.services.permission import PermissionService
from app.services.principal import Principal
from app.services.service_account import ServiceAccountService


@pytest_asyncio.fixture
async def org(db):
    return await OrganizationService(db).create_organization(name="PrincOrg")


@pytest_asyncio.fixture
async def other_org(db):
    return await OrganizationService(db).create_organization(name="PrincOther")


@pytest_asyncio.fixture
async def sa(db, org):
    return await ServiceAccountService(db).create_service_account(
        name="issuer-sa",
        organization_id=org.id,
        can_create_cert=True,
    )


async def _ca_owned_by_sa(db, org, sa):
    ca = await CAService(db).create_ca(
        name="SA CA", subject_dn="CN=SA CA", organization_id=org.id
    )
    ca.created_by_service_account_id = sa.id
    db.add(ca)
    await db.commit()
    await db.refresh(ca)
    return ca


async def _org_ca(db, org):
    return await CAService(db).create_ca(
        name="Org CA", subject_dn="CN=Org CA", organization_id=org.id
    )


def test_principal_from_service_account_is_user_role_scoped(sa):
    p = Principal.from_service_account(sa)
    assert p.kind == "service_account"
    assert p.id == sa.id
    assert p.organization_id == sa.organization_id
    assert p.role == UserRole.USER
    assert p.can_create_cert is True
    assert p.can_create_ca is False


@pytest.mark.asyncio
async def test_sa_is_creator_of_its_own_ca(db, org, sa):
    ca = await _ca_owned_by_sa(db, org, sa)
    perm = PermissionService(db)
    # Creator gets full access even for actions it lacks a capability flag for.
    result = await perm.check_ca_access(
        Principal.from_service_account(sa), ca.id, PermissionAction.DELETE_CA
    )
    assert result.id == ca.id


@pytest.mark.asyncio
async def test_sa_capability_flag_enforced_on_org_resource(db, org, sa):
    ca = await _org_ca(db, org)
    perm = PermissionService(db)
    p = Principal.from_service_account(sa)
    # READ allowed in same org
    assert (await perm.check_ca_access(p, ca.id, PermissionAction.READ)).id == ca.id
    # has can_create_cert -> CREATE_CERT allowed
    assert (
        await perm.check_ca_access(p, ca.id, PermissionAction.CREATE_CERT)
    ).id == ca.id
    # lacks can_delete_ca -> denied on a resource it does not own
    with pytest.raises(PermissionDeniedError):
        await perm.check_ca_access(p, ca.id, PermissionAction.DELETE_CA)


@pytest.mark.asyncio
async def test_sa_cross_org_denied(db, other_org, sa):
    ca = await CAService(db).create_ca(
        name="Other CA", subject_dn="CN=Other CA", organization_id=other_org.id
    )
    perm = PermissionService(db)
    with pytest.raises(PermissionDeniedError):
        await perm.check_ca_access(
            Principal.from_service_account(sa), ca.id, PermissionAction.READ
        )


@pytest.mark.asyncio
async def test_sa_denied_on_unowned_resource(db, org, sa):
    # org_id=None resource is superuser-only; an SA must never reach it.
    ca = await CAService(db).create_ca(name="Unowned", subject_dn="CN=Unowned")
    perm = PermissionService(db)
    with pytest.raises(PermissionDeniedError):
        await perm.check_ca_access(
            Principal.from_service_account(sa), ca.id, PermissionAction.READ
        )
