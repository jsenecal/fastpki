# mypy: disable-error-code="arg-type"
import hashlib
import hmac
import secrets
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import ServiceAccount, ServiceAccountToken
from app.services.exceptions import AlreadyExistsError, NotFoundError

UTC = ZoneInfo("UTC")

TOKEN_PREFIX = "fpki_sa_"
PEPPER_VERSION = 1

_SA_NOT_FOUND = "Service account not found"
_SA_EXISTS = "A service account with this name already exists in the organization"
_TOKEN_NOT_FOUND = "Service account token not found"


def _pepper(version: int = PEPPER_VERSION) -> str:
    """Return the pepper for the given version, falling back to SECRET_KEY."""
    return settings.SERVICE_ACCOUNT_TOKEN_PEPPER or settings.SECRET_KEY


def _digest(secret: str, version: int = PEPPER_VERSION) -> str:
    """HMAC-SHA256 the token secret with the server-side pepper."""
    return hmac.new(
        _pepper(version).encode("utf-8"), secret.encode("utf-8"), hashlib.sha256
    ).hexdigest()


class ServiceAccountService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_service_account(
        self,
        *,
        name: str,
        organization_id: int,
        created_by_user_id: int | None = None,
        description: str | None = None,
        can_create_ca: bool = False,
        can_create_cert: bool = False,
        can_revoke_cert: bool = False,
        can_export_private_key: bool = False,
        can_delete_ca: bool = False,
    ) -> ServiceAccount:
        """Create a service account scoped to an organization."""
        existing = await self.get_service_account_by_name(organization_id, name)
        if existing:
            raise AlreadyExistsError(_SA_EXISTS)

        sa = ServiceAccount(
            name=name,
            organization_id=organization_id,
            created_by_user_id=created_by_user_id,
            description=description,
            can_create_ca=can_create_ca,
            can_create_cert=can_create_cert,
            can_revoke_cert=can_revoke_cert,
            can_export_private_key=can_export_private_key,
            can_delete_ca=can_delete_ca,
        )
        self.db.add(sa)
        await self.db.commit()
        await self.db.refresh(sa)
        return sa

    async def get_service_account_by_id(self, sa_id: int) -> ServiceAccount | None:
        """Get a service account by ID."""
        result = await self.db.execute(
            select(ServiceAccount).where(ServiceAccount.id == sa_id)
        )
        return result.scalar_one_or_none()

    async def get_service_account_by_name(
        self, organization_id: int, name: str
    ) -> ServiceAccount | None:
        """Get a service account by name within an organization."""
        result = await self.db.execute(
            select(ServiceAccount).where(
                ServiceAccount.organization_id == organization_id,
                ServiceAccount.name == name,
            )
        )
        return result.scalar_one_or_none()

    async def list_service_accounts(
        self, *, organization_id: int | None = None
    ) -> list[ServiceAccount]:
        """List service accounts, optionally scoped to an organization."""
        query = select(ServiceAccount).order_by(ServiceAccount.name)
        if organization_id is not None:
            query = query.where(ServiceAccount.organization_id == organization_id)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def set_disabled(self, sa_id: int, *, disabled: bool) -> ServiceAccount:
        """Enable or disable a service account."""
        sa = await self._get_or_raise(sa_id)
        sa.disabled_at = datetime.now(UTC) if disabled else None
        self.db.add(sa)
        await self.db.commit()
        await self.db.refresh(sa)
        return sa

    async def update_service_account(
        self,
        sa_id: int,
        *,
        name: str | None = None,
        description: str | None = None,
        disabled: bool | None = None,
        can_create_ca: bool | None = None,
        can_create_cert: bool | None = None,
        can_revoke_cert: bool | None = None,
        can_export_private_key: bool | None = None,
        can_delete_ca: bool | None = None,
    ) -> ServiceAccount:
        """Update a service account. Only provided fields are changed."""
        sa = await self._get_or_raise(sa_id)

        if name is not None and name != sa.name:
            existing = await self.get_service_account_by_name(sa.organization_id, name)
            if existing is not None and existing.id != sa.id:
                raise AlreadyExistsError(_SA_EXISTS)
            sa.name = name

        if description is not None:
            sa.description = description
        if disabled is not None:
            sa.disabled_at = datetime.now(UTC) if disabled else None
        if can_create_ca is not None:
            sa.can_create_ca = can_create_ca
        if can_create_cert is not None:
            sa.can_create_cert = can_create_cert
        if can_revoke_cert is not None:
            sa.can_revoke_cert = can_revoke_cert
        if can_export_private_key is not None:
            sa.can_export_private_key = can_export_private_key
        if can_delete_ca is not None:
            sa.can_delete_ca = can_delete_ca

        self.db.add(sa)
        await self.db.commit()
        await self.db.refresh(sa)
        return sa

    async def delete_service_account(self, sa_id: int) -> None:
        """Delete a service account and all of its tokens."""
        sa = await self._get_or_raise(sa_id)
        await self.db.execute(
            delete(ServiceAccountToken).where(
                ServiceAccountToken.service_account_id == sa_id
            )
        )
        await self.db.delete(sa)
        await self.db.commit()

    async def _get_or_raise(self, sa_id: int) -> ServiceAccount:
        sa = await self.get_service_account_by_id(sa_id)
        if sa is None:
            raise NotFoundError(_SA_NOT_FOUND)
        return sa

    # --- token lifecycle -------------------------------------------------

    async def mint_token(
        self,
        service_account_id: int,
        *,
        name: str | None = None,
        expires_at: datetime | None = None,
    ) -> tuple[ServiceAccountToken, str]:
        """Mint a new token.

        Returns the persisted record and the full plaintext token. The
        plaintext is shown only here and is never stored; only an HMAC-SHA256
        digest of the secret is persisted.
        """
        await self._get_or_raise(service_account_id)

        public_id = secrets.token_urlsafe(12)
        secret = secrets.token_urlsafe(32)
        token = ServiceAccountToken(
            service_account_id=service_account_id,
            public_id=public_id,
            digest=_digest(secret),
            pepper_version=PEPPER_VERSION,
            name=name,
            expires_at=expires_at,
        )
        self.db.add(token)
        await self.db.commit()
        await self.db.refresh(token)

        plaintext = f"{TOKEN_PREFIX}{public_id}.{secret}"
        return token, plaintext

    async def get_token_by_id(self, token_id: int) -> ServiceAccountToken | None:
        """Get a token record by ID."""
        result = await self.db.execute(
            select(ServiceAccountToken).where(ServiceAccountToken.id == token_id)
        )
        return result.scalar_one_or_none()

    async def list_tokens(self, service_account_id: int) -> list[ServiceAccountToken]:
        """List token records for a service account (metadata only)."""
        result = await self.db.execute(
            select(ServiceAccountToken)
            .where(ServiceAccountToken.service_account_id == service_account_id)
            .order_by(ServiceAccountToken.created_at)
        )
        return list(result.scalars().all())

    async def revoke_token(self, token_id: int) -> ServiceAccountToken:
        """Revoke a token."""
        token = await self.get_token_by_id(token_id)
        if token is None:
            raise NotFoundError(_TOKEN_NOT_FOUND)
        token.revoked = True
        self.db.add(token)
        await self.db.commit()
        await self.db.refresh(token)
        return token

    async def resolve_token(self, raw_token: str) -> ServiceAccount | None:
        """Resolve a bearer token to its (enabled) service account, or None.

        Returns None for any failure mode (malformed, unknown, revoked,
        expired, bad secret, disabled account) so the caller can answer with a
        single uniform 401.
        """
        if not raw_token.startswith(TOKEN_PREFIX):
            return None
        body = raw_token[len(TOKEN_PREFIX) :]
        public_id, sep, secret = body.partition(".")
        if not sep or not public_id or not secret:
            return None

        result = await self.db.execute(
            select(ServiceAccountToken).where(
                ServiceAccountToken.public_id == public_id
            )
        )
        token = result.scalar_one_or_none()
        if token is None or token.revoked:
            return None

        if token.expires_at is not None:
            expires_at = token.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            if expires_at <= datetime.now(UTC):
                return None

        expected = _digest(secret, token.pepper_version)
        if not hmac.compare_digest(expected, token.digest):
            return None

        sa = await self.get_service_account_by_id(token.service_account_id)
        if sa is None or sa.disabled_at is not None:
            return None

        token.last_used_at = datetime.now(UTC)
        self.db.add(token)
        await self.db.commit()
        return sa
