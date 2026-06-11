"""Tests for PATCH /cas/{ca_id} — assigning a CA to an organization (issue #51)."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.db.models import AuditAction, AuditLog, Certificate, CertificateAuthority
from app.services.ca import CAService
from app.services.cert import CertificateService, CertificateType


async def _create_orgless_ca(
    db: AsyncSession, name: str = "Legacy Root", parent_ca_id: int | None = None
) -> CertificateAuthority:
    """Create a CA with no organization, as on pre-organization instances."""
    ca_service = CAService(db)
    return await ca_service.create_ca(
        name=name,
        subject_dn=f"CN={name}",
        key_size=2048,
        valid_days=365,
        organization_id=None,
        parent_ca_id=parent_ca_id,
    )


async def _create_orgless_cert(
    db: AsyncSession, ca_id: int, common_name: str = "legacy.example.com"
) -> Certificate:
    cert_service = CertificateService(db)
    return await cert_service.create_certificate(
        ca_id=ca_id,
        common_name=common_name,
        subject_dn=f"CN={common_name}",
        certificate_type=CertificateType.SERVER,
        key_size=2048,
        valid_days=30,
        organization_id=None,
    )


@pytest.mark.asyncio
class TestAssignOrganization:
    async def test_superuser_assigns_orgless_ca_to_org(
        self, superuser_client, db, test_org
    ):
        ca = await _create_orgless_ca(db)
        response = await superuser_client.patch(
            f"/api/v1/cas/{ca.id}",
            json={"organization_id": test_org.id},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == ca.id
        assert data["organization_id"] == test_org.id

    async def test_admin_cannot_assign_organization(self, admin_client, db, test_org):
        ca = await _create_orgless_ca(db)
        response = await admin_client.patch(
            f"/api/v1/cas/{ca.id}",
            json={"organization_id": test_org.id},
        )
        assert response.status_code == 403

    async def test_normal_user_cannot_assign_organization(
        self, normal_user_client, db, test_org
    ):
        ca = await _create_orgless_ca(db)
        response = await normal_user_client.patch(
            f"/api/v1/cas/{ca.id}",
            json={"organization_id": test_org.id},
        )
        assert response.status_code == 403

    async def test_unknown_ca_returns_404(self, superuser_client, test_org):
        response = await superuser_client.patch(
            "/api/v1/cas/99999",
            json={"organization_id": test_org.id},
        )
        assert response.status_code == 404

    async def test_unknown_organization_returns_400(self, superuser_client, db):
        ca = await _create_orgless_ca(db)
        response = await superuser_client.patch(
            f"/api/v1/cas/{ca.id}",
            json={"organization_id": 99999},
        )
        assert response.status_code == 400

    async def test_without_cascade_children_and_certs_untouched(
        self, superuser_client, db, test_org
    ):
        root = await _create_orgless_ca(db, name="Legacy Root NC")
        child = await _create_orgless_ca(
            db, name="Legacy Intermediate NC", parent_ca_id=root.id
        )
        cert = await _create_orgless_cert(db, child.id)

        response = await superuser_client.patch(
            f"/api/v1/cas/{root.id}",
            json={"organization_id": test_org.id},
        )
        assert response.status_code == 200

        await db.refresh(child)
        await db.refresh(cert)
        assert child.organization_id is None
        assert cert.organization_id is None

    async def test_cascade_adopts_descendant_cas_and_certificates(
        self, superuser_client, db, test_org
    ):
        root = await _create_orgless_ca(db, name="Legacy Root C")
        child = await _create_orgless_ca(
            db, name="Legacy Intermediate C", parent_ca_id=root.id
        )
        grandchild = await _create_orgless_ca(
            db, name="Legacy Issuing C", parent_ca_id=child.id
        )
        cert = await _create_orgless_cert(db, grandchild.id)

        response = await superuser_client.patch(
            f"/api/v1/cas/{root.id}",
            json={"organization_id": test_org.id, "cascade": True},
        )
        assert response.status_code == 200
        assert response.json()["organization_id"] == test_org.id

        await db.refresh(child)
        await db.refresh(grandchild)
        await db.refresh(cert)
        assert child.organization_id == test_org.id
        assert grandchild.organization_id == test_org.id
        assert cert.organization_id == test_org.id

    async def test_unauthenticated_returns_401(self, client, db, test_org):
        ca = await _create_orgless_ca(db, name="Unauthed Root")
        response = await client.patch(
            f"/api/v1/cas/{ca.id}",
            json={"organization_id": test_org.id},
        )
        assert response.status_code == 401

    async def test_assignment_is_audit_logged(self, superuser_client, db, test_org):
        ca = await _create_orgless_ca(db, name="Audited Root")
        response = await superuser_client.patch(
            f"/api/v1/cas/{ca.id}",
            json={"organization_id": test_org.id},
        )
        assert response.status_code == 200

        result = await db.execute(
            select(AuditLog).where(AuditLog.action == AuditAction.CA_UPDATE)
        )
        logs = list(result.scalars().all())
        assert len(logs) == 1
        assert logs[0].resource_type == "ca"
        assert logs[0].resource_id == ca.id
        assert logs[0].organization_id == test_org.id
        # The previous owner must be reconstructable from the audit trail.
        assert "unassigned" in (logs[0].detail or "")

    async def test_audit_detail_records_previous_organization(
        self, superuser_client, db, test_org
    ):
        from app.services.organization import OrganizationService

        other_org = await OrganizationService(db).create_organization(
            name="ApiPreviousOrg", description="previous owner"
        )
        ca = await _create_orgless_ca(db, name="Moved Root")
        first = await superuser_client.patch(
            f"/api/v1/cas/{ca.id}",
            json={"organization_id": other_org.id},
        )
        assert first.status_code == 200

        second = await superuser_client.patch(
            f"/api/v1/cas/{ca.id}",
            json={"organization_id": test_org.id},
        )
        assert second.status_code == 200

        result = await db.execute(
            select(AuditLog).where(AuditLog.action == AuditAction.CA_UPDATE)
        )
        logs = list(result.scalars().all())
        assert len(logs) == 2
        assert f"organization {other_org.id}" in (logs[1].detail or "")
