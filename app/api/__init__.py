from fastapi import APIRouter

from app.api import (
    audit,
    auth,
    ca,
    certs,
    export,
    organizations,
    service_accounts,
    users,
)

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(
    organizations.router, prefix="/organizations", tags=["organizations"]
)
api_router.include_router(ca.router, prefix="/cas", tags=["certificate-authorities"])
api_router.include_router(certs.router, prefix="/certificates", tags=["certificates"])
api_router.include_router(export.router, prefix="/export", tags=["export"])
api_router.include_router(
    service_accounts.router, prefix="/service-accounts", tags=["service-accounts"]
)
api_router.include_router(audit.router, prefix="/audit-logs", tags=["audit-logs"])
