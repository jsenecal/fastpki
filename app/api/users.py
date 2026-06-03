from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.api.deps import get_current_active_superuser, get_current_active_user
from app.core.config import logger, settings
from app.db.models import AuditAction, User, UserRole
from app.db.session import get_session
from app.schemas.user import User as UserSchema
from app.schemas.user import UserCreate, UserUpdate
from app.services.audit import AuditService
from app.services.user import UserService

# Optional OAuth2 scheme for endpoints that need to work with or without auth
oauth2_scheme_optional = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/token", auto_error=False
)

router = APIRouter()


@router.post("/", response_model=UserSchema, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_in: UserCreate,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    token: str | None = Depends(oauth2_scheme_optional),
) -> Any:
    logger.debug("Create user request for username: %s", user_in.username)
    user_service = UserService(db)

    # Check if username already exists
    db_user = await user_service.get_user_by_username(username=user_in.username)
    if db_user:
        logger.info("Create user failed: username already exists: %s", user_in.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    # Check if email already exists
    db_user = await user_service.get_user_by_email(email=user_in.email)
    if db_user:
        logger.info("Create user failed: email already exists: %s", user_in.email)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Set current_user if token is provided
    current_user = None
    if token:
        try:
            from app.api.deps import get_current_user

            current_user = await get_current_user(token=token, db=db)
            logger.debug(
                "Token provided by user: %s (role: %s)",
                current_user.username,
                current_user.role,
            )
        except Exception as e:
            logger.debug("Invalid token provided: %s", e)
            # Invalid token, current_user remains None
            pass

    # Check if there are any users in the system
    result = await db.execute(select(User).limit(1))
    first_user = result.scalar_one_or_none() is None

    logger.debug("First user check: %s", first_user)

    # If not the first user and not authenticated, check registration policy
    if not first_user and current_user is None:
        # Elevated roles always require authentication
        if user_in.role in [UserRole.ADMIN, UserRole.SUPERUSER]:
            logger.info(
                "Create user failed: insufficient permissions to create %s user",
                user_in.role,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions to create admin or superuser accounts",
            )
        # Regular user creation requires the setting to be enabled
        if not settings.ALLOW_UNAUTHENTICATED_REGISTRATION:
            logger.info("Create user failed: unauthenticated registration is disabled")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Unauthenticated registration is disabled",
            )

    # Only superusers can create users with admin/superuser roles
    # But for the first user in the system, we allow any role
    if (
        user_in.role in [UserRole.ADMIN, UserRole.SUPERUSER]
        and not first_user
        and (current_user is None or current_user.role != UserRole.SUPERUSER)
    ):
        logger.info(
            "Create user failed: insufficient permissions to create %s user",
            user_in.role,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to create admin or superuser accounts",
        )

    # Create the user
    user = await user_service.create_user(
        username=user_in.username,
        email=user_in.email,
        password=user_in.password,
        role=user_in.role,
        organization_id=user_in.organization_id,
    )

    audit_service = AuditService(db)
    await audit_service.log_action(
        action=AuditAction.USER_CREATE,
        user_id=current_user.id if current_user else None,
        username=current_user.username if current_user else None,
        organization_id=user.organization_id,
        resource_type="user",
        resource_id=user.id,
        detail=f"Created user '{user.username}'",
    )

    logger.info(
        "User created successfully: %s (ID: %s, Role: %s)",
        user.username,
        user.id,
        user.role,
    )
    return user


@router.get("/", response_model=list[UserSchema])
async def read_users(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    current_user: User = Depends(get_current_active_superuser),  # noqa: B008
) -> Any:
    """
    Retrieve users. Only superusers can access this endpoint.
    """
    result = await db.execute(select(User).offset(skip).limit(limit))
    users = result.scalars().all()
    return users


@router.get("/me", response_model=UserSchema)
async def read_user_me(
    current_user: User = Depends(get_current_active_user),  # noqa: B008
) -> Any:
    """
    Get current user.
    """
    return current_user


@router.get("/{user_id}", response_model=UserSchema)
async def read_user_by_id(
    user_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    current_user: User = Depends(get_current_active_user),  # noqa: B008
) -> Any:
    """
    Get a specific user by id.
    """
    user_service = UserService(db)
    user = await user_service.get_user_by_id(user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Organization-scoped visibility (issue #9):
    # - SUPERUSER: any user across all orgs.
    # - ADMIN: only users within their own organization (and they must have one).
    # - USER: only their own profile.
    is_self = current_user.id == user_id
    if current_user.role == UserRole.SUPERUSER:
        pass
    elif (
        current_user.role == UserRole.ADMIN and current_user.organization_id is not None
    ):
        if user.organization_id != current_user.organization_id and not is_self:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )
    elif not is_self:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    return user


@router.patch("/{user_id}", response_model=UserSchema)
async def update_user(
    user_id: int,
    user_in: UserUpdate,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    current_user: User = Depends(get_current_active_user),  # noqa: B008
) -> Any:
    """
    Update a user.
    """
    user_service = UserService(db)
    user = await user_service.get_user_by_id(user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Check permissions
    is_superuser = current_user.role == UserRole.SUPERUSER
    is_admin = current_user.role == UserRole.ADMIN
    is_self = current_user.id == user_id

    # Only superusers can change roles or activate/deactivate users
    if (user_in.role is not None or user_in.is_active is not None) and not is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to change role or activation status",
        )

    # Check capability field changes: require ADMIN in same org or SUPERUSER
    capability_fields = [
        user_in.can_create_ca,
        user_in.can_create_cert,
        user_in.can_revoke_cert,
        user_in.can_export_private_key,
        user_in.can_delete_ca,
    ]
    if any(f is not None for f in capability_fields):
        same_org = (
            is_admin
            and current_user.organization_id is not None
            and current_user.organization_id == user.organization_id
        )
        if not (is_superuser or same_org):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions to change capabilities",
            )

    # Only superusers can change organization assignments
    if (
        "organization_id" in user_in.model_fields_set
        and user_in.organization_id != user.organization_id
        and not is_superuser
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to change organization assignment",
        )

    # Regular users can only update their own profile
    if not (is_superuser or is_admin or is_self):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    # Don't allow downgrading superusers unless by another superuser
    if (
        user.role == UserRole.SUPERUSER
        and user_in.role is not None
        and user_in.role != UserRole.SUPERUSER
        and (not is_superuser or is_self)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superusers cannot downgrade themselves",
        )

    # Update the user
    updated_user = await user_service.update_user(
        user_id=user_id,
        email=user_in.email,
        password=user_in.password,
        role=user_in.role,
        is_active=user_in.is_active,
        organization_id=user_in.organization_id,
        can_create_ca=user_in.can_create_ca,
        can_create_cert=user_in.can_create_cert,
        can_revoke_cert=user_in.can_revoke_cert,
        can_export_private_key=user_in.can_export_private_key,
        can_delete_ca=user_in.can_delete_ca,
    )

    audit_service = AuditService(db)
    await audit_service.log_action(
        action=AuditAction.USER_UPDATE,
        user_id=current_user.id,
        username=current_user.username,
        organization_id=current_user.organization_id,
        resource_type="user",
        resource_id=user_id,
        detail=f"Updated user '{user.username}'",
    )

    return updated_user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_session),  # noqa: B008
    current_user: User = Depends(get_current_active_superuser),  # noqa: B008
) -> None:
    """
    Delete a user. Only superusers can delete users.
    """
    user_service = UserService(db)
    user = await user_service.get_user_by_id(user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Don't allow superusers to delete themselves
    if current_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Superusers cannot delete themselves",
        )

    # Delete the user
    await user_service.delete_user(user_id)
