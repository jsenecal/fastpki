from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_active_user
from app.db.models import AuditAction, User, UserRole
from app.db.session import get_session
from app.schemas.audit import AuditLogResponse
from app.services.audit import AuditService

router = APIRouter()


@router.get("/", response_model=list[AuditLogResponse])
async def list_audit_logs(
    action: AuditAction | None = Query(None),  # noqa: B008
    user_id: int | None = Query(None),
    resource_type: str | None = Query(None),
    resource_id: int | None = Query(None),
    since: datetime | None = Query(None),  # noqa: B008
    until: datetime | None = Query(None),  # noqa: B008
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_session, scope="function"),  # noqa: B008
    current_user: User = Depends(get_current_active_user),  # noqa: B008
) -> list[AuditLogResponse]:
    if current_user.role == UserRole.USER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )

    audit_service = AuditService(db)

    organization_id: int | None = None
    if current_user.role == UserRole.ADMIN:
        organization_id = current_user.organization_id

    logs = await audit_service.list_audit_logs(
        action=action,
        user_id=user_id,
        organization_id=organization_id,
        resource_type=resource_type,
        resource_id=resource_id,
        since=since,
        until=until,
        skip=skip,
        limit=limit,
    )
    return logs
