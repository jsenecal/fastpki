import pytest

from app.services.exceptions import AlreadyExistsError, NotFoundError
from app.services.service_account import ServiceAccountService


@pytest.mark.asyncio
async def test_create_service_account(db, test_org, admin_user):
    service = ServiceAccountService(db)

    sa = await service.create_service_account(
        name="ci-runner",
        organization_id=test_org.id,
        created_by_user_id=admin_user.id,
        description="CI pipeline",
        can_create_cert=True,
    )

    assert sa.id is not None
    assert sa.name == "ci-runner"
    assert sa.organization_id == test_org.id
    assert sa.created_by_user_id == admin_user.id
    assert sa.description == "CI pipeline"
    assert sa.can_create_cert is True
    assert sa.can_create_ca is False
    assert sa.disabled_at is None


@pytest.mark.asyncio
async def test_create_service_account_duplicate_name_in_org_rejected(
    db, test_org, admin_user
):
    service = ServiceAccountService(db)
    await service.create_service_account(
        name="dup",
        organization_id=test_org.id,
        created_by_user_id=admin_user.id,
    )

    with pytest.raises(AlreadyExistsError):
        await service.create_service_account(
            name="dup",
            organization_id=test_org.id,
            created_by_user_id=admin_user.id,
        )


@pytest.mark.asyncio
async def test_get_service_account_by_id(db, test_org, admin_user):
    service = ServiceAccountService(db)
    sa = await service.create_service_account(
        name="lookup",
        organization_id=test_org.id,
        created_by_user_id=admin_user.id,
    )

    fetched = await service.get_service_account_by_id(sa.id)
    assert fetched is not None
    assert fetched.id == sa.id


@pytest.mark.asyncio
async def test_list_service_accounts_scoped_to_org(db, test_org, admin_user):
    service = ServiceAccountService(db)
    await service.create_service_account(
        name="a", organization_id=test_org.id, created_by_user_id=admin_user.id
    )
    await service.create_service_account(
        name="b", organization_id=test_org.id, created_by_user_id=admin_user.id
    )

    accounts = await service.list_service_accounts(organization_id=test_org.id)
    assert {a.name for a in accounts} == {"a", "b"}


@pytest.mark.asyncio
async def test_disable_and_enable_service_account(db, test_org, admin_user):
    service = ServiceAccountService(db)
    sa = await service.create_service_account(
        name="toggle", organization_id=test_org.id, created_by_user_id=admin_user.id
    )

    disabled = await service.set_disabled(sa.id, disabled=True)
    assert disabled.disabled_at is not None

    enabled = await service.set_disabled(sa.id, disabled=False)
    assert enabled.disabled_at is None


@pytest.mark.asyncio
async def test_delete_service_account(db, test_org, admin_user):
    service = ServiceAccountService(db)
    sa = await service.create_service_account(
        name="gone", organization_id=test_org.id, created_by_user_id=admin_user.id
    )

    await service.delete_service_account(sa.id)

    assert await service.get_service_account_by_id(sa.id) is None
    with pytest.raises(NotFoundError):
        await service.delete_service_account(sa.id)
