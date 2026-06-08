from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.db.models import CertificateType


class ServiceAccountCreate(BaseModel):
    name: str
    description: str | None = None
    # For superusers (who may have no organization of their own); ignored for
    # org-scoped admins, who always create within their own organization.
    organization_id: int | None = None
    can_create_ca: bool = False
    can_create_cert: bool = False
    can_revoke_cert: bool = False
    can_export_private_key: bool = False
    can_delete_ca: bool = False


class ServiceAccountUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    disabled: bool | None = None
    can_create_ca: bool | None = None
    can_create_cert: bool | None = None
    can_revoke_cert: bool | None = None
    can_export_private_key: bool | None = None
    can_delete_ca: bool | None = None


class ServiceAccountResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    organization_id: int
    created_by_user_id: int | None
    created_at: datetime
    updated_at: datetime
    disabled_at: datetime | None
    can_create_ca: bool
    can_create_cert: bool
    can_revoke_cert: bool
    can_export_private_key: bool
    can_delete_ca: bool


class ServiceAccountTokenCreate(BaseModel):
    name: str | None = None
    expires_at: datetime | None = None


class ServiceAccountTokenResponse(BaseModel):
    """Token metadata — never includes the plaintext or the digest."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    public_id: str
    name: str | None
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None
    revoked: bool


class ServiceAccountTokenCreateResponse(ServiceAccountTokenResponse):
    """Returned only at mint time — carries the one-time plaintext token."""

    token: str


class IssuancePolicyUpsert(BaseModel):
    cn_patterns: list[str] = []
    san_dns_patterns: list[str] = []
    san_ip_cidrs: list[str] = []
    san_email_domains: list[str] = []
    allowed_ca_ids: list[int] = []
    allowed_certificate_types: list[CertificateType] = []
    max_validity_days: int


class IssuancePolicyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    service_account_id: int
    cn_patterns: list[str]
    san_dns_patterns: list[str]
    san_ip_cidrs: list[str]
    san_email_domains: list[str]
    allowed_ca_ids: list[int]
    allowed_certificate_types: list[str]
    max_validity_days: int
    created_at: datetime
    updated_at: datetime
