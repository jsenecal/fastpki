import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import (
    get_current_active_admin_user,
    get_current_active_superuser,
    get_current_active_user,
    get_current_principal,
    get_current_user,
)
from app.db.models import User, UserRole
from app.db.session import get_session
from app.main import app
from app.services.organization import OrganizationService
from app.services.user import UserService
from tests.conftest import (
    TestAuth,
    TestPrincipalAuth,
    create_test_app,
    get_test_session,
)

real_app = app


def test_cors_default_is_empty():
    """CORS default should be empty list, not wildcard."""
    from app.core.config import Settings

    # Verify the class-level default (not the runtime value from .env)
    field = Settings.model_fields["BACKEND_CORS_ORIGINS"]
    assert field.default == [], (
        "BACKEND_CORS_ORIGINS default must be [] to prevent wildcard CORS"
    )


def test_cors_warns_on_wildcard_origin():
    """Setting BACKEND_CORS_ORIGINS to ['*'] should emit a warning."""
    import warnings

    from app.core.config import Settings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = Settings(
            BACKEND_CORS_ORIGINS=["*"],
            SECRET_KEY="a" * 32,
        )
        cors_warnings = [x for x in w if "BACKEND_CORS_ORIGINS" in str(x.message)]
        assert len(cors_warnings) == 1
        assert "production" in str(cors_warnings[0].message).lower()
        assert s.BACKEND_CORS_ORIGINS == ["*"]


@pytest.mark.asyncio
async def test_cors_no_middleware_when_origins_empty():
    """When CORS origins are empty, no CORS headers should be added."""
    # create_test_app() builds a fresh FastAPI without CORS middleware
    test_app = create_test_app()
    test_app.dependency_overrides[get_session] = get_test_session
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.options(
            "/api/v1/auth/token",
            headers={
                "Origin": "https://evil.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        allow_origin = response.headers.get("access-control-allow-origin", "")
        assert allow_origin != "https://evil.com", (
            "CORS must not reflect arbitrary origins when origins list is empty"
        )


def _app_for(user: User):
    test_app = create_test_app()
    test_app.dependency_overrides[get_session] = get_test_session
    test_app.dependency_overrides[get_current_user] = TestAuth(user)
    test_app.dependency_overrides[get_current_active_user] = TestAuth(user)
    test_app.dependency_overrides[get_current_active_superuser] = TestAuth(user)
    test_app.dependency_overrides[get_current_active_admin_user] = TestAuth(user)
    test_app.dependency_overrides[get_current_principal] = TestPrincipalAuth(user)
    return test_app


@pytest.mark.asyncio
async def test_normal_user_cannot_change_own_organization(setup_db, db):
    """A regular user must not be able to reassign themselves to another org."""
    org_service = OrganizationService(db)
    org_a = await org_service.create_organization(name="OrgA", description="")
    org_b = await org_service.create_organization(name="OrgB", description="")

    user_service = UserService(db)
    user = await user_service.create_user(
        username="orghopper",
        email="orghopper@example.com",
        password="password123",
        role=UserRole.USER,
        organization_id=org_a.id,
    )

    test_app = create_test_app()
    test_app.dependency_overrides[get_session] = get_test_session
    test_app.dependency_overrides[get_current_user] = TestAuth(user)
    test_app.dependency_overrides[get_current_active_user] = TestAuth(user)

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.patch(
            f"/api/v1/users/{user.id}",
            json={"organization_id": org_b.id},
        )
        assert response.status_code == 403


@pytest.mark.asyncio
async def test_admin_cannot_change_own_organization(setup_db, db):
    """An admin user must not be able to reassign themselves to another org."""
    org_service = OrganizationService(db)
    org_a = await org_service.create_organization(name="OrgC", description="")
    org_b = await org_service.create_organization(name="OrgD", description="")

    user_service = UserService(db)
    admin = await user_service.create_user(
        username="adminhopper",
        email="adminhopper@example.com",
        password="password123",
        role=UserRole.ADMIN,
        organization_id=org_a.id,
    )

    test_app = create_test_app()
    test_app.dependency_overrides[get_session] = get_test_session
    test_app.dependency_overrides[get_current_user] = TestAuth(admin)
    test_app.dependency_overrides[get_current_active_user] = TestAuth(admin)
    test_app.dependency_overrides[get_current_active_admin_user] = TestAuth(admin)

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.patch(
            f"/api/v1/users/{admin.id}",
            json={"organization_id": org_b.id},
        )
        assert response.status_code == 403


@pytest.mark.asyncio
async def test_superuser_can_change_user_organization(setup_db, db):
    """A superuser should be able to reassign a user's organization."""
    org_service = OrganizationService(db)
    org_a = await org_service.create_organization(name="OrgE", description="")
    org_b = await org_service.create_organization(name="OrgF", description="")

    user_service = UserService(db)
    target = await user_service.create_user(
        username="target_user",
        email="target@example.com",
        password="password123",
        role=UserRole.USER,
        organization_id=org_a.id,
    )
    su = await user_service.create_user(
        username="su_for_org_test",
        email="su_org@example.com",
        password="password123",
        role=UserRole.SUPERUSER,
    )

    test_app = _app_for(su)
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.patch(
            f"/api/v1/users/{target.id}",
            json={"organization_id": org_b.id},
        )
        assert response.status_code == 200
        assert response.json()["organization_id"] == org_b.id


@pytest.mark.asyncio
async def test_auth_rate_limiting(setup_db):
    """Auth endpoint should rate-limit after too many attempts."""
    from app.api.auth import limiter

    app.dependency_overrides[get_session] = get_test_session
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            responses = []
            for i in range(10):
                resp = await client.post(
                    "/api/v1/auth/token",
                    data={"username": f"brute_{i}", "password": "wrong"},
                )
                responses.append(resp.status_code)

            assert 429 in responses, (
                f"Expected at least one 429 response in {responses}"
            )
    finally:
        app.dependency_overrides.pop(get_session, None)
        limiter.reset()


@pytest.mark.asyncio
async def test_security_headers_present():
    """All responses should include security headers."""
    transport = ASGITransport(app=real_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

        assert response.headers.get("x-content-type-options") == "nosniff"
        assert response.headers.get("x-frame-options") == "DENY"
        assert (
            response.headers.get("referrer-policy") == "strict-origin-when-cross-origin"
        )
        assert response.headers.get("cache-control") == "no-store"
        assert "max-age=" in response.headers.get("strict-transport-security", "")


@pytest.mark.asyncio
async def test_security_headers_on_api_endpoint():
    """Security headers should be present on API endpoints too."""
    transport = ASGITransport(app=real_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/users/me")
        assert response.headers.get("x-content-type-options") == "nosniff"
        assert response.headers.get("x-frame-options") == "DENY"


def test_enable_docs_default_is_false():
    """ENABLE_DOCS must default to False so docs aren't exposed in production."""
    from app.core.config import Settings

    field = Settings.model_fields["ENABLE_DOCS"]
    assert field.default is False, (
        "ENABLE_DOCS default must be False to avoid leaking the OpenAPI schema"
    )


@pytest.mark.asyncio
async def test_docs_endpoints_disabled_by_default():
    """When ENABLE_DOCS is False, /docs, /redoc, and /openapi.json must 404."""
    from app.main import create_app

    test_app = create_app(enable_docs=False)
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for path in ("/docs", "/redoc", "/api/v1/openapi.json"):
            response = await client.get(path)
            assert response.status_code == 404, (
                f"{path} must return 404 when ENABLE_DOCS is False, "
                f"got {response.status_code}"
            )


@pytest.mark.asyncio
async def test_docs_endpoints_enabled_when_setting_is_true():
    """When ENABLE_DOCS is True, /docs, /redoc, and /openapi.json must respond."""
    from app.main import create_app

    test_app = create_app(enable_docs=True)
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for path in ("/docs", "/redoc", "/api/v1/openapi.json"):
            response = await client.get(path)
            assert response.status_code == 200, (
                f"{path} must be reachable when ENABLE_DOCS is True, "
                f"got {response.status_code}"
            )
