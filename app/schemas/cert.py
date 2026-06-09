from datetime import datetime

from pydantic import BaseModel

from app.db.models import CertificateStatus, CertificateType


class CertificateCreate(BaseModel):
    common_name: str
    subject_dn: str
    certificate_type: CertificateType
    key_size: int | None = None
    valid_days: int | None = None
    include_private_key: bool = True
    san_dns_names: list[str] | None = None
    san_ip_addresses: list[str] | None = None
    san_email_addresses: list[str] | None = None


class CSRSignRequest(BaseModel):
    csr: str
    ca_id: int | None = None
    ca_name: str | None = None
    certificate_type: CertificateType
    valid_days: int | None = None
    common_name: str | None = None
    subject_dn: str | None = None
    san_dns_names: list[str] | None = None
    san_ip_addresses: list[str] | None = None
    san_email_addresses: list[str] | None = None


class CertificateResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    common_name: str
    subject_dn: str
    certificate_type: CertificateType
    key_size: int
    valid_days: int
    status: CertificateStatus
    certificate: str
    serial_number: str
    not_before: datetime
    not_after: datetime
    revoked_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    issuer_id: int
    organization_id: int | None = None
    created_by_user_id: int | None = None
    created_by_service_account_id: int | None = None
    renewed_from_id: int | None = None
    renewed_to_ids: list[int] = []


class CertificateDetailResponse(CertificateResponse):
    private_key: str | None = None


class CertificateRevoke(BaseModel):
    reason: str | None = None


class CertificateRenewRequest(BaseModel):
    csr: str | None = None
