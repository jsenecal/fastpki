"""A mutation and its audit row commit together or not at all (issue #55).

Services only add/flush; the request unit of work (``get_session``) commits
once, before the response is sent. These tests inject a failure into the
audit insert (or the commit itself) and assert that the mutation was not
persisted and the client saw an error rather than a success.
"""

from collections.abc import AsyncGenerator, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.dependencies.models import Dependant
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.core.config import settings
from app.db.models import (
    AuditAction,
    AuditLog,
    Certificate,
    CertificateAuthority,
    CertificateStatus,
    CRLEntry,
    ServiceAccount,
    User,
)
from app.db.session import get_session, unit_of_work
from app.main import create_app
from tests.conftest import (
    _make_client_app,
    create_test_app,
    get_test_session,
)
from tests.conftest import (
    test_session_maker as fresh_session_maker,
)

API = settings.API_V1_STR


class AuditInsertError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("simulated audit insert failure")


def _boom(*_args: Any) -> None:
    raise AuditInsertError


@contextmanager
def audit_insert_fails() -> Iterator[None]:
    """Make every AuditLog INSERT fail at flush time."""
    event.listen(AuditLog, "before_insert", _boom)
    try:
        yield
    finally:
        event.remove(AuditLog, "before_insert", _boom)


@pytest.fixture
def failing_audit_insert() -> Iterator[None]:
    with audit_insert_fails():
        yield


def _client(app: FastAPI) -> AsyncClient:
    # raise_app_exceptions=False: observe the 500 the client would really get.
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest_asyncio.fixture
async def su_client(setup_db, superuser) -> AsyncGenerator[AsyncClient, None]:
    async with _client(_make_client_app(superuser)) as c:
        yield c


@pytest_asyncio.fixture
async def admin_c(setup_db, admin_user) -> AsyncGenerator[AsyncClient, None]:
    async with _client(_make_client_app(admin_user)) as c:
        yield c


async def _count(model: Any, *where: Any) -> int:
    """Count rows from a fresh session, i.e. only what was committed."""
    async with fresh_session_maker() as s:
        stmt = select(func.count()).select_from(model)
        for clause in where:
            stmt = stmt.where(clause)
        return int((await s.execute(stmt)).scalar_one())


async def _create_ca(c: AsyncClient, name: str) -> int:
    resp = await c.post(f"{API}/cas/", json={"name": name, "subject_dn": f"CN={name}"})
    assert resp.status_code == 201, resp.text
    return int(resp.json()["id"])


async def _create_cert(c: AsyncClient, ca_id: int, cn: str) -> int:
    resp = await c.post(
        f"{API}/certificates/?ca_id={ca_id}",
        json={
            "common_name": cn,
            "subject_dn": f"CN={cn}",
            "certificate_type": "server",
            "key_size": 2048,
            "valid_days": 30,
        },
    )
    assert resp.status_code == 201, resp.text
    return int(resp.json()["id"])


async def test_user_create_rolled_back_when_audit_insert_fails(
    setup_db, failing_audit_insert
):
    app = create_test_app()
    app.dependency_overrides[get_session] = get_test_session
    async with _client(app) as c:
        resp = await c.post(
            f"{API}/users/",
            json={
                "username": "atomic",
                "email": "atomic@example.com",
                "password": "Kestrel-Meadow-4821",
            },
        )
    assert resp.status_code == 500, resp.text
    assert await _count(User, User.username == "atomic") == 0
    assert await _count(AuditLog) == 0


async def test_service_account_create_rolled_back_when_audit_insert_fails(
    admin_c, failing_audit_insert
):
    resp = await admin_c.post(f"{API}/service-accounts/", json={"name": "ci"})
    assert resp.status_code == 500, resp.text
    assert await _count(ServiceAccount, ServiceAccount.name == "ci") == 0


async def test_service_account_create_retry_succeeds_after_failed_audit(
    admin_c,
):
    """The issue's symptom: a retry after a failed audit must not 400."""

    with audit_insert_fails():
        first = await admin_c.post(f"{API}/service-accounts/", json={"name": "ci"})
    assert first.status_code == 500

    retry = await admin_c.post(f"{API}/service-accounts/", json={"name": "ci"})
    assert retry.status_code == 201, retry.text
    assert await _count(ServiceAccount, ServiceAccount.name == "ci") == 1
    assert (
        await _count(AuditLog, AuditLog.action == AuditAction.SERVICE_ACCOUNT_CREATE)
        == 1
    )


async def test_ca_create_rolled_back_when_audit_insert_fails(
    su_client, failing_audit_insert
):
    resp = await su_client.post(
        f"{API}/cas/", json={"name": "Atomic CA", "subject_dn": "CN=Atomic CA"}
    )
    assert resp.status_code == 500, resp.text
    assert await _count(CertificateAuthority) == 0


async def test_cert_create_rolled_back_when_audit_insert_fails(su_client):
    ca_id = await _create_ca(su_client, "Issuing CA")

    with audit_insert_fails():
        resp = await su_client.post(
            f"{API}/certificates/?ca_id={ca_id}",
            json={
                "common_name": "atomic.example.com",
                "subject_dn": "CN=atomic.example.com",
                "certificate_type": "server",
                "key_size": 2048,
                "valid_days": 30,
            },
        )

    assert resp.status_code == 500, resp.text
    assert await _count(Certificate) == 0
    # The CA and its audit row from the earlier, successful request remain.
    assert await _count(CertificateAuthority) == 1
    assert await _count(AuditLog, AuditLog.action == AuditAction.CA_CREATE) == 1


async def test_cert_revoke_rolled_back_when_audit_insert_fails(su_client):
    ca_id = await _create_ca(su_client, "Revoking CA")
    cert_id = await _create_cert(su_client, ca_id, "revoke.example.com")

    with audit_insert_fails():
        resp = await su_client.post(
            f"{API}/certificates/{cert_id}/revoke", json={"reason": "test"}
        )

    assert resp.status_code == 500, resp.text
    assert await _count(CRLEntry) == 0
    assert (
        await _count(
            Certificate,
            Certificate.id == cert_id,
            Certificate.status == CertificateStatus.VALID,
        )
        == 1
    )


async def test_commit_failure_is_reported_before_response(setup_db, superuser):
    """A failing commit must reach the client as a 500, never as a 201.

    Guards the ``scope="function"`` contract on ``get_session``: with the
    default request scope the commit would run after the 201 was sent.
    """

    async def failing_commit_session() -> AsyncGenerator[AsyncSession, None]:
        async with unit_of_work(fresh_session_maker) as session:
            session.commit = AsyncMock(  # type: ignore[method-assign]
                side_effect=RuntimeError("simulated commit failure")
            )
            yield session

    app = _make_client_app(superuser)
    app.dependency_overrides[get_session] = failing_commit_session
    async with _client(app) as c:
        resp = await c.post(
            f"{API}/cas/", json={"name": "Never CA", "subject_dn": "CN=Never CA"}
        )
    assert resp.status_code == 500, resp.text
    assert await _count(CertificateAuthority) == 0


async def test_login_failure_audit_persists(client: AsyncClient):
    resp = await client.post(
        f"{API}/auth/token", data={"username": "ghost", "password": "wrong-password"}
    )
    assert resp.status_code == 401
    assert (
        await _count(
            AuditLog,
            AuditLog.action == AuditAction.LOGIN_FAILURE,
            AuditLog.username == "ghost",
        )
        == 1
    )


def _session_dependants(dependant: Dependant) -> Iterator[Dependant]:
    for sub in dependant.dependencies:
        if sub.call is get_session:
            yield sub
        yield from _session_dependants(sub)


def test_every_get_session_dependency_is_function_scoped():
    """Request-scoped get_session would commit after the response is sent."""
    app = create_app(enable_docs=False)
    found = 0
    for route in app.routes:
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            continue
        for sub in _session_dependants(dependant):
            found += 1
            assert sub.scope == "function", (
                f"{getattr(route, 'path', route)}: get_session must be declared "
                'as Depends(get_session, scope="function")'
            )
    assert found > 0
