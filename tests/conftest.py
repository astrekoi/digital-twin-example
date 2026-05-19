"""Shared pytest fixtures.

The migration to Taipy/Celery archived legacy tests under
tests/_archive_reflex/. We exclude that directory from collection.
"""
from __future__ import annotations

from pathlib import Path

import pytest

collect_ignore = ["_archive_reflex"]


@pytest.fixture
def tmp_sqlite(tmp_path) -> Path:
    """Empty SQLite DB path; the store creates the file on first connection."""
    return tmp_path / "telemetry.db"


@pytest.fixture
def sqlite_store(tmp_sqlite):
    from src.storage.factory import create_telemetry_store

    store = create_telemetry_store(backend="sqlite", sqlite_db_path=tmp_sqlite)
    store.init_schema()
    yield store
    store.close()


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Strip integration creds from env so tests never accidentally reach the network."""
    for k in (
        "ALLOW_EXTERNAL_API_CALLS",
        "PINATA_JWT",
        "HEDERA_OPERATOR_KEY",
        "HEDERA_OPERATOR_ID",
        "HEDERA_TOPIC_ID",
        "HEDERA_ENABLED",
        "IPFS_ENABLED",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ALLOW_EXTERNAL_API_CALLS", "false")
    monkeypatch.setenv("HEDERA_ENABLED", "false")
    monkeypatch.setenv("IPFS_ENABLED", "false")
    monkeypatch.setenv("APP_ENV", "test")
    yield
