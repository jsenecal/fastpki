from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.db.models import AuditAction, AuditLog
from app.services.principal import Principal


class AuditService:
    def __init__(self, db: AsyncSession):
        self.db = db

    @staticmethod
    def actor_fields(principal: Principal) -> dict[str, object | None]:
        """Map a principal to the audit actor columns for its kind."""
        if principal.kind == "service_account":
            return {
                "service_account_id": principal.id,
                "service_account_name": principal.display_name,
            }
        return {"user_id": principal.id, "username": principal.display_name}

    async def log_action(
        self,
        action: AuditAction,
        user_id: int | None = None,
        username: str | None = None,
        organization_id: int | None = None,
        resource_type: str | None = None,
        resource_id: int | None = None,
        detail: str | None = None,
        service_account_id: int | None = None,
        service_account_name: str | None = None,
    ) -> AuditLog:
        entry = AuditLog(
            action=action,
            user_id=user_id,
            username=username,
            organization_id=organization_id,
            resource_type=resource_type,
            resource_id=resource_id,
            detail=detail,
            service_account_id=service_account_id,
            service_account_name=service_account_name,
        )
        self.db.add(entry)
        await self.db.commit()
        await self.db.refresh(entry)
        return entry

    async def list_audit_logs(
        self,
        action: AuditAction | None = None,
        user_id: int | None = None,
        organization_id: int | None = None,
        resource_type: str | None = None,
        resource_id: int | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AuditLog]:
        query = select(AuditLog)
        if action is not None:
            query = query.where(AuditLog.action == action)
        if user_id is not None:
            query = query.where(AuditLog.user_id == user_id)
        if organization_id is not None:
            query = query.where(AuditLog.organization_id == organization_id)
        if resource_type is not None:
            query = query.where(AuditLog.resource_type == resource_type)
        if resource_id is not None:
            query = query.where(AuditLog.resource_id == resource_id)
        if since is not None:
            query = query.where(AuditLog.created_at >= since)
        if until is not None:
            query = query.where(AuditLog.created_at <= until)
        query = query.order_by(AuditLog.created_at.desc())  # type: ignore[attr-defined]
        query = query.offset(skip).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())
