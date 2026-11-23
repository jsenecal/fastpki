from datetime import datetime

from pydantic import BaseModel, EmailStr, ValidationInfo, field_validator

from app.core.password_policy import validate_password
from app.db.models import UserRole


class UserBase(BaseModel):
    username: str
    email: EmailStr
    role: UserRole | None = UserRole.USER
    is_active: bool | None = True
    organization_id: int | None = None
    can_create_ca: bool = False
    can_create_cert: bool = False
    can_revoke_cert: bool = False
    can_export_private_key: bool = False
    can_delete_ca: bool = False


class UserCreate(UserBase):
    password: str

    @field_validator("password")
    @classmethod
    def password_policy(cls, v: str, info: ValidationInfo) -> str:
        # username/email are declared before password, so they are already
        # validated and available in info.data (absent if they failed).
        return validate_password(
            v, username=info.data.get("username"), email=info.data.get("email")
        )


class UserUpdate(BaseModel):
    email: EmailStr | None = None
    password: str | None = None
    role: UserRole | None = None
    is_active: bool | None = None
    organization_id: int | None = None
    can_create_ca: bool | None = None
    can_create_cert: bool | None = None
    can_revoke_cert: bool | None = None
    can_export_private_key: bool | None = None
    can_delete_ca: bool | None = None

    @field_validator("password")
    @classmethod
    def password_policy(cls, v: str | None, info: ValidationInfo) -> str | None:
        # The target user's username (and current email) are not known here;
        # the PATCH /users/{id} endpoint re-checks against them.
        if v is None:
            return v
        return validate_password(v, email=info.data.get("email"))


class UserInDBBase(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class User(UserInDBBase):
    pass


class UserInDB(UserInDBBase):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    refresh_token: str | None = None
    token_type: str


class TokenPayload(BaseModel):
    sub: str | None = None
    id: int | None = None
    role: str | None = None
    exp: int | None = None
    jti: str | None = None
    iat: int | None = None


class RefreshTokenRequest(BaseModel):
    refresh_token: str
