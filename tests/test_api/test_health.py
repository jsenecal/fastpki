"""Tests for the DB-aware /health endpoint (issue #57).

The K8s probes must fail when the database is unreachable; probing a static
route (like `/`) hides connection failures and leaves the pod wedged.
"""

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.db.session import get_session
from tests.conftest import get_test_session


@pytest_asyncio.fixture
async def health_client(setup_db):
    from app.main import create_app

    app = create_app()
    app.dependency_overrides[get_session] = get_test_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health_ok_when_db_reachable(health_client):
    resp = await health_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_health_503_when_db_unavailable(setup_db):
    from app.main import create_app

    class _DeadSession:
        async def execute(self, *args, **kwargs):
            raise OSError("connection reset by peer")  # noqa: TRY003

    async def dead_session():
        yield _DeadSession()

    app = create_app()
    app.dependency_overrides[get_session] = dead_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/health")
    assert resp.status_code == 503
