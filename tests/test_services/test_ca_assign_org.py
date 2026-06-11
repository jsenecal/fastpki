"""Service-level tests for CAService.assign_organization (issue #51)."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import PermissionAction, UserRole
from app.services.ca import CAService
from app.services.cert import CertificateService, CertificateType
from app.services.exceptions import NotFoundError, PermissionDeniedError
from app.services.permission import PermissionService
from app.services.principal import Principal


def _service_account_principal(organization_id: int) -> Principal:
    """An org-scoped service account with issuance capability, as in issue #51."""
    return Principal(
        kind="service_account",
        id=1,
        organization_id=organization_id,
        role=UserRole.USER,
        is_active=True,
        display_name="gnmic-onboard",
        can_create_ca=False,
        can_create_cert=True,
        can_revoke_cert=False,
        can_export_private_key=False,
        can_delete_ca=False,
    )


async def _create_orgless_ca(
    db: AsyncSession, name: str, parent_ca_id: int | None = None
):
    return await CAService(db).create_ca(
        name=name,
        subject_dn=f"CN={name}",
        key_size=2048,
        valid_days=365,
        organization_id=None,
        parent_ca_id=parent_ca_id,
    )


@pytest.mark.asyncio
class TestAssignOrganizationService:
    async def test_assigns_organization_and_returns_counts(self, db, test_org):
        ca = await _create_orgless_ca(db, "Svc Legacy Root")
        ca_service = CAService(db)

        updated, cas_updated, certs_updated = await ca_service.assign_organization(
            ca.id, test_org.id
        )

        assert updated.organization_id == test_org.id
        assert cas_updated == 1
        assert certs_updated == 0

    async def test_cascade_counts_subtree_cas_and_certs(self, db, test_org):
        root = await _create_orgless_ca(db, "Svc Root")
        child = await _create_orgless_ca(db, "Svc Child", parent_ca_id=root.id)
        cert_service = CertificateService(db)
        await cert_service.create_certificate(
            ca_id=child.id,
            common_name="svc.example.com",
            subject_dn="CN=svc.example.com",
            certificate_type=CertificateType.SERVER,
            key_size=2048,
            valid_days=30,
            organization_id=None,
        )

        ca_service = CAService(db)
        _, cas_updated, certs_updated = await ca_service.assign_organization(
            root.id, test_org.id, cascade=True
        )

        assert cas_updated == 2
        assert certs_updated == 1

    async def test_missing_ca_raises_not_found(self, db, test_org):
        with pytest.raises(NotFoundError):
            await CAService(db).assign_organization(99999, test_org.id)

    async def test_org_service_account_can_issue_after_assignment(self, db, test_org):
        """The issue #51 scenario: an org-scoped SA blocked on an org-less CA
        gains CREATE_CERT access once the CA is assigned to its organization."""
        ca = await _create_orgless_ca(db, "NetOps Intermediate")
        principal = _service_account_principal(test_org.id)
        perm = PermissionService(db)

        with pytest.raises(PermissionDeniedError):
            await perm.check_ca_access(principal, ca.id, PermissionAction.CREATE_CERT)

        await CAService(db).assign_organization(ca.id, test_org.id)

        granted = await perm.check_ca_access(
            principal, ca.id, PermissionAction.CREATE_CERT
        )
        assert granted.id == ca.id
