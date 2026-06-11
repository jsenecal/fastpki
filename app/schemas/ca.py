from datetime import datetime

from pydantic import BaseModel, computed_field


class CACreate(BaseModel):
    name: str
    description: str | None = None
    subject_dn: str
    key_size: int | None = None
    valid_days: int | None = None
    parent_ca_id: int | None = None
    path_length: int | None = None
    allow_leaf_certs: bool | None = None
    crl_base_url: str | None = None


class CAAssignOrganization(BaseModel):
    organization_id: int
    cascade: bool = False


class CAResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    name: str
    description: str | None = None
    subject_dn: str
    key_size: int
    valid_days: int
    created_at: datetime
    updated_at: datetime
    certificate: str
    organization_id: int | None = None
    created_by_user_id: int | None = None
    parent_ca_id: int | None = None
    path_length: int | None = None
    allow_leaf_certs: bool
    crl_base_url: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_root(self) -> bool:
        return self.parent_ca_id is None


class CADetailResponse(CAResponse):
    private_key: str
