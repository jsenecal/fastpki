"""Tests for database engine connection-resilience configuration (issue #57)."""

from app.core.config import settings
from app.db.session import build_engine_kwargs


def test_settings_have_db_resilience_defaults():
    assert settings.DB_POOL_PRE_PING is True
    assert settings.DB_POOL_RECYCLE == 300
    assert settings.DB_COMMAND_TIMEOUT == 30.0


def test_engine_kwargs_postgres_sets_command_timeout_and_pool_resilience():
    kw = build_engine_kwargs("postgresql+asyncpg://user:pw@host/db")
    assert kw["pool_pre_ping"] is True
    assert kw["pool_recycle"] == 300
    # command_timeout bounds every asyncpg query so a dead socket raises
    # instead of hanging forever.
    assert kw["connect_args"]["command_timeout"] == 30.0


def test_engine_kwargs_sqlite_keeps_check_same_thread_and_no_command_timeout():
    kw = build_engine_kwargs("sqlite+aiosqlite:///./fastpki.db")
    assert kw["pool_pre_ping"] is True
    assert kw["connect_args"]["check_same_thread"] is False
    # command_timeout is asyncpg-only; passing it to aiosqlite would error.
    assert "command_timeout" not in kw["connect_args"]


def test_engine_kwargs_command_timeout_disabled_when_not_positive(monkeypatch):
    monkeypatch.setattr(settings, "DB_COMMAND_TIMEOUT", 0)
    kw = build_engine_kwargs("postgresql+asyncpg://user:pw@host/db")
    assert "command_timeout" not in kw["connect_args"]


def test_engine_kwargs_pool_recycle_passthrough(monkeypatch):
    monkeypatch.setattr(settings, "DB_POOL_RECYCLE", -1)
    kw = build_engine_kwargs("postgresql+asyncpg://user:pw@host/db")
    assert kw["pool_recycle"] == -1


def test_engine_kwargs_preserves_user_connect_args_without_override(monkeypatch):
    # User-provided connect args are kept; an explicit command_timeout wins
    # over the default (setdefault must not clobber it).
    monkeypatch.setattr(
        settings,
        "DATABASE_CONNECT_ARGS",
        {"command_timeout": 5, "server_settings": {"application_name": "fastpki"}},
    )
    kw = build_engine_kwargs("postgresql+asyncpg://user:pw@host/db")
    assert kw["connect_args"]["command_timeout"] == 5
    assert kw["connect_args"]["server_settings"] == {"application_name": "fastpki"}
