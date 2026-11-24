from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import status
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel

from app.api import api_router
from app.api.deps import (
    get_current_active_admin_user,
    get_current_active_superuser,
    get_current_active_user,
    get_current_user,
)
from app.core.config import settings
from app.db.models import User, UserRole
from app.db.session import get_session, unit_of_work
from tests.conftest import ArrangeSession

# Test database for this specific test
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test_organizations.db"

# Create async engine for tests
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    poolclass=NullPool,
    connect_args={"check_same_thread": False},
)

# Create test session
test_session_maker = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)


async def get_test_session() -> AsyncSession:
    async with unit_of_work(test_session_maker) as session:
        yield session


# App for testing
def create_test_app():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(api_router, prefix=settings.API_V1_STR)
    return app


# Test auth class that always returns the provided user
class TestAuth:
    def __init__(self, user: User):
        self.user = user

    async def __call__(self) -> User:
        return self.user


@pytest_asyncio.fixture(scope="function")
async def setup_test_db():
    # Create the test database and tables
    # Generate a unique database name for each test
    db_name = f"./test_organizations_{id(setup_test_db)}.db"

    # Remove test database if it exists
    if Path(db_name).exists():
        Path(db_name).unlink()

    # Create a connection engine for this specific test
    test_engine_local = create_async_engine(
        f"sqlite+aiosqlite:///{db_name}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )

    # Create test session maker for this test
    local_session_maker = async_sessionmaker(
        test_engine_local, class_=AsyncSession, expire_on_commit=False
    )

    # Provide the session maker
    global test_session_maker
    test_session_maker = local_session_maker

    async with test_engine_local.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield
    # Clean up after tests
    async with test_engine_local.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)

    # Remove test database file
    if Path(db_name).exists():
        Path(db_name).unlink()


@pytest_asyncio.fixture
async def test_db(setup_test_db):
    # Arrangement session: commits on each service flush (see conftest).
    async with ArrangeSession(
        bind=test_session_maker.kw["bind"], expire_on_commit=False
    ) as session:
        try:
            yield session
        finally:
            await session.close()


@pytest_asyncio.fixture
async def test_superuser(test_db):
    """Create and return a superuser for testing."""
    from app.services.user import UserService

    user_service = UserService(test_db)
    user = await user_service.create_user(
        username="testsuper",
        email="super@example.com",
        password="password123",
        role=UserRole.SUPERUSER,
    )
    return user


@pytest_asyncio.fixture
async def test_admin_user(test_db):
    """Create and return an admin user for testing."""
    from app.services.user import UserService

    user_service = UserService(test_db)
    user = await user_service.create_user(
        username="testadmin",
        email="admin@example.com",
        password="password123",
        role=UserRole.ADMIN,
    )
    return user


@pytest_asyncio.fixture
async def test_normal_user(test_db):
    """Create and return a normal user for testing."""
    from app.services.user import UserService

    user_service = UserService(test_db)
    user = await user_service.create_user(
        username="testuser",
        email="user@example.com",
        password="password123",
        role=UserRole.USER,
    )
    return user


@pytest_asyncio.fixture
async def superuser_client(setup_test_db, test_superuser):
    """Client with superuser authentication."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(api_router, prefix=settings.API_V1_STR)

    # Override the dependencies
    app.dependency_overrides[get_session] = get_test_session

    # Create a function that returns the superuser
    async def override_get_current_user():
        return test_superuser

    async def override_get_current_active_user():
        return test_superuser

    async def override_get_current_active_superuser():
        return test_superuser

    async def override_get_current_active_admin_user():
        return test_superuser

    # Apply the overrides
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_current_active_user] = override_get_current_active_user
    app.dependency_overrides[get_current_active_superuser] = (
        override_get_current_active_superuser
    )
    app.dependency_overrides[get_current_active_admin_user] = (
        override_get_current_active_admin_user
    )

    # Create test client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def admin_client(setup_test_db, test_admin_user):
    """Client with admin authentication."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(api_router, prefix=settings.API_V1_STR)

    # Override the dependencies
    app.dependency_overrides[get_session] = get_test_session

    # Create a function that returns the admin user
    async def override_get_current_user():
        return test_admin_user

    async def override_get_current_active_user():
        return test_admin_user

    async def override_get_current_active_admin_user():
        return test_admin_user

    # Apply the overrides
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_current_active_user] = override_get_current_active_user
    app.dependency_overrides[get_current_active_admin_user] = (
        override_get_current_active_admin_user
    )

    # Create test client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def normal_client(setup_test_db, test_normal_user):
    """Client with normal user authentication."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(api_router, prefix=settings.API_V1_STR)

    # Override the dependencies
    app.dependency_overrides[get_session] = get_test_session

    # Create a function that returns the normal user
    async def override_get_current_user():
        return test_normal_user

    async def override_get_current_active_user():
        return test_normal_user

    # Apply the overrides
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_current_active_user] = override_get_current_active_user

    # Create test client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_create_organization(superuser_client):
    # Generate a unique organization name for each test run
    import time

    org_name = f"Test Organization {time.time()}"

    # Test creating a new organization with superuser
    response = await superuser_client.post(
        "/api/v1/organizations/",
        json={"name": org_name, "description": "This is a test organization"},
    )

    # Print the response for debugging
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.text}")

    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["name"] == org_name
    assert data["description"] == "This is a test organization"
    assert "id" in data
    assert "created_at" in data
    assert "updated_at" in data


@pytest.mark.asyncio
async def test_regular_user_cannot_create_organization(normal_client):
    # Regular users should not be able to create organizations
    response = await normal_client.post(
        "/api/v1/organizations/",
        json={"name": "Unauthorized Org", "description": "This should not be created"},
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_get_organization(superuser_client):
    # Generate a unique organization name
    import time

    org_name = f"Get Org {time.time()}"

    # First create an organization
    create_response = await superuser_client.post(
        "/api/v1/organizations/",
        json={"name": org_name, "description": "Organization to retrieve"},
    )

    org_id = create_response.json()["id"]

    # Now get the organization
    response = await superuser_client.get(f"/api/v1/organizations/{org_id}")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["id"] == org_id
    assert data["name"] == org_name
    assert data["description"] == "Organization to retrieve"


@pytest.mark.asyncio
async def test_get_nonexistent_organization(superuser_client):
    # Try to get an organization that doesn't exist
    response = await superuser_client.get("/api/v1/organizations/99999")

    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_get_all_organizations(superuser_client):
    # Generate unique organization names
    import time

    timestamp = time.time()
    org_name1 = f"Org List 1 {timestamp}"
    org_name2 = f"Org List 2 {timestamp}"

    # Create multiple organizations
    await superuser_client.post(
        "/api/v1/organizations/",
        json={"name": org_name1, "description": "First org for list test"},
    )

    await superuser_client.post(
        "/api/v1/organizations/",
        json={"name": org_name2, "description": "Second org for list test"},
    )

    # Get all organizations
    response = await superuser_client.get("/api/v1/organizations/")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)

    # Check if our organizations are in the list
    org_names = [org["name"] for org in data]
    assert org_name1 in org_names
    assert org_name2 in org_names


@pytest.mark.asyncio
async def test_update_organization(superuser_client):
    # Generate unique organization names
    import time

    timestamp = time.time()
    org_name = f"Update Test Org {timestamp}"
    updated_name = f"Updated Org {timestamp}"

    # First create an organization
    create_response = await superuser_client.post(
        "/api/v1/organizations/",
        json={"name": org_name, "description": "Will be updated"},
    )

    org_id = create_response.json()["id"]

    # Now update the organization
    response = await superuser_client.put(
        f"/api/v1/organizations/{org_id}",
        json={"name": updated_name, "description": "Has been updated"},
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["id"] == org_id
    assert data["name"] == updated_name
    assert data["description"] == "Has been updated"


@pytest.mark.asyncio
async def test_delete_organization(superuser_client):
    # Generate a unique organization name
    import time

    org_name = f"Delete Test Org {time.time()}"

    # First create an organization
    create_response = await superuser_client.post(
        "/api/v1/organizations/",
        json={"name": org_name, "description": "Will be deleted"},
    )

    org_id = create_response.json()["id"]

    # Now delete the organization
    response = await superuser_client.delete(f"/api/v1/organizations/{org_id}")

    assert response.status_code == status.HTTP_204_NO_CONTENT

    # Verify the organization is gone
    get_response = await superuser_client.get(f"/api/v1/organizations/{org_id}")

    assert get_response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_add_user_to_organization(setup_test_db):
    """Test adding a user to an organization."""
    import time

    from fastapi import FastAPI

    from app.services.user import UserService

    # Create a test app with consistent database session
    app = FastAPI()
    app.include_router(api_router, prefix=settings.API_V1_STR)

    # Create a session for this test
    async with test_session_maker() as session:
        # Create a superuser in the test database
        user_service = UserService(session)
        superuser = await user_service.create_user(
            username=f"test_super_{time.time()}",
            email=f"test_super_{time.time()}@example.com",
            password="password123",
            role=UserRole.SUPERUSER,
        )

        # Create a normal user in the same session
        normal_user = await user_service.create_user(
            username=f"test_normal_{time.time()}",
            email=f"test_normal_{time.time()}@example.com",
            password="password123",
            role=UserRole.USER,
        )

        # Use the session in our app
        async def override_get_session():
            try:
                yield session
            finally:
                pass  # Don't close the session here as it's managed by the test

        # Override auth dependencies
        async def override_get_current_user():
            return superuser

        async def override_get_current_active_user():
            return superuser

        async def override_get_current_active_superuser():
            return superuser

        app.dependency_overrides[get_session] = override_get_session
        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[get_current_active_user] = (
            override_get_current_active_user
        )
        app.dependency_overrides[get_current_active_superuser] = (
            override_get_current_active_superuser
        )
        app.dependency_overrides[get_current_active_admin_user] = (
            override_get_current_active_superuser
        )

        # Create test client
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create an organization
            org_name = f"User Org {time.time()}"
            org_response = await client.post(
                "/api/v1/organizations/",
                json={
                    "name": org_name,
                    "description": "Organization for user assignment",
                },
            )

            assert org_response.status_code == status.HTTP_201_CREATED
            org_id = org_response.json()["id"]

            # Add the user to the organization
            add_response = await client.post(
                f"/api/v1/organizations/{org_id}/users/{normal_user.id}"
            )

            # Check response
            assert add_response.status_code == status.HTTP_200_OK
            data = add_response.json()
            assert data["id"] == normal_user.id
            assert data["organization_id"] == org_id


@pytest.mark.asyncio
async def test_remove_user_from_organization(setup_test_db):
    """Test removing a user from an organization."""
    import time

    from fastapi import FastAPI

    from app.services.user import UserService

    # Create a test app with consistent database session
    app = FastAPI()
    app.include_router(api_router, prefix=settings.API_V1_STR)

    # Create a session for this test
    async with test_session_maker() as session:
        # Create a superuser in the test database
        user_service = UserService(session)
        superuser = await user_service.create_user(
            username=f"test_super_remove_{time.time()}",
            email=f"test_super_remove_{time.time()}@example.com",
            password="password123",
            role=UserRole.SUPERUSER,
        )

        # Create a normal user in the same session
        normal_user = await user_service.create_user(
            username=f"test_normal_remove_{time.time()}",
            email=f"test_normal_remove_{time.time()}@example.com",
            password="password123",
            role=UserRole.USER,
        )

        # Use the session in our app
        async def override_get_session():
            try:
                yield session
            finally:
                pass

        # Override auth dependencies
        async def override_get_current_user():
            return superuser

        async def override_get_current_active_user():
            return superuser

        async def override_get_current_active_superuser():
            return superuser

        app.dependency_overrides[get_session] = override_get_session
        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[get_current_active_user] = (
            override_get_current_active_user
        )
        app.dependency_overrides[get_current_active_superuser] = (
            override_get_current_active_superuser
        )
        app.dependency_overrides[get_current_active_admin_user] = (
            override_get_current_active_superuser
        )

        # Create test client
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create an organization
            org_name = f"User Remove Org {time.time()}"
            org_response = await client.post(
                "/api/v1/organizations/",
                json={"name": org_name, "description": "Organization for user removal"},
            )

            assert org_response.status_code == status.HTTP_201_CREATED
            org_id = org_response.json()["id"]

            # Add the user to the organization
            add_response = await client.post(
                f"/api/v1/organizations/{org_id}/users/{normal_user.id}"
            )
            assert add_response.status_code == status.HTTP_200_OK

            # Now remove the user from the organization
            remove_response = await client.delete(
                f"/api/v1/organizations/{org_id}/users/{normal_user.id}"
            )

            # Check response
            assert remove_response.status_code == status.HTTP_200_OK
            data = remove_response.json()
            assert data["id"] == normal_user.id
            assert data["organization_id"] is None


@pytest.mark.asyncio
async def test_get_organization_users(setup_test_db):
    """Test getting all users in an organization."""
    import time

    from fastapi import FastAPI

    from app.services.user import UserService

    # Create a test app with consistent database session
    app = FastAPI()
    app.include_router(api_router, prefix=settings.API_V1_STR)

    # Create a session for this test
    async with test_session_maker() as session:
        # Create a superuser in the test database
        user_service = UserService(session)
        superuser = await user_service.create_user(
            username=f"test_super_list_{time.time()}",
            email=f"test_super_list_{time.time()}@example.com",
            password="password123",
            role=UserRole.SUPERUSER,
        )

        # Create a normal user in the same session
        normal_user = await user_service.create_user(
            username=f"test_normal_list_{time.time()}",
            email=f"test_normal_list_{time.time()}@example.com",
            password="password123",
            role=UserRole.USER,
        )

        # Use the session in our app
        async def override_get_session():
            try:
                yield session
            finally:
                pass

        # Override auth dependencies
        async def override_get_current_user():
            return superuser

        async def override_get_current_active_user():
            return superuser

        async def override_get_current_active_superuser():
            return superuser

        app.dependency_overrides[get_session] = override_get_session
        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[get_current_active_user] = (
            override_get_current_active_user
        )
        app.dependency_overrides[get_current_active_superuser] = (
            override_get_current_active_superuser
        )
        app.dependency_overrides[get_current_active_admin_user] = (
            override_get_current_active_superuser
        )

        # Create test client
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create an organization
            timestamp = time.time()
            org_name = f"Multi User Org {timestamp}"
            org_response = await client.post(
                "/api/v1/organizations/",
                json={
                    "name": org_name,
                    "description": "Organization with multiple users",
                },
            )

            assert org_response.status_code == status.HTTP_201_CREATED
            org_id = org_response.json()["id"]

            # Add a user to the organization
            add_response = await client.post(
                f"/api/v1/organizations/{org_id}/users/{normal_user.id}"
            )
            assert add_response.status_code == status.HTTP_200_OK

            # Now get all users in the organization
            users_response = await client.get(f"/api/v1/organizations/{org_id}/users")

            # Check response
            assert users_response.status_code == status.HTTP_200_OK
            users_data = users_response.json()
            assert isinstance(users_data, list)
            assert len(users_data) >= 1

            # Check if our test user is in the list
            user_ids = [user["id"] for user in users_data]
            assert normal_user.id in user_ids


@pytest.mark.asyncio
async def test_user_access_own_organization(setup_test_db):
    """Test a user accessing their own organization."""
    import time

    from fastapi import FastAPI

    from app.services.organization import OrganizationService
    from app.services.user import UserService

    # Create a test app with consistent database session
    app = FastAPI()
    app.include_router(api_router, prefix=settings.API_V1_STR)

    # Create a session for this test
    async with test_session_maker() as session:
        # Create a superuser in the test database to set everything up
        user_service = UserService(session)
        superuser = await user_service.create_user(
            username=f"test_super_access_{time.time()}",
            email=f"test_super_access_{time.time()}@example.com",
            password="password123",
            role=UserRole.SUPERUSER,
        )

        # Create an admin user in the same session
        admin_user = await user_service.create_user(
            username=f"test_admin_access_{time.time()}",
            email=f"test_admin_access_{time.time()}@example.com",
            password="password123",
            role=UserRole.ADMIN,
        )

        # Create the organization and add the admin to it directly
        # (bypassing API permission checks)
        org_service = OrganizationService(session)
        org_name = f"Admin Org {time.time()}"
        org = await org_service.create_organization(
            name=org_name, description="Organization for admin access test"
        )

        # Directly assign the user to the organization via the service
        await org_service.add_user_to_organization(
            user_id=admin_user.id,
            org_id=org.id,
            admin_user_id=superuser.id,  # Use the superuser to bypass permission checks
        )

        # Use the session in our app
        async def override_get_session():
            try:
                yield session
            finally:
                pass

        # Override auth dependencies to use our admin user
        async def override_get_current_user():
            return admin_user

        async def override_get_current_active_user():
            return admin_user

        async def override_get_current_active_admin_user():
            return admin_user

        app.dependency_overrides[get_session] = override_get_session
        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[get_current_active_user] = (
            override_get_current_active_user
        )
        app.dependency_overrides[get_current_active_admin_user] = (
            override_get_current_active_admin_user
        )

        # Create test client
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Admin should be able to access their own organization
            access_response = await client.get(f"/api/v1/organizations/{org.id}")

            # Check response
            assert access_response.status_code == status.HTTP_200_OK
            org_data = access_response.json()
            assert org_data["id"] == org.id
            assert org_data["name"] == org_name
