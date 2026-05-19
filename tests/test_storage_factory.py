"""Factory: backend switching and DSN validation."""
from __future__ import annotations

import pytest

from src.storage.factory import UnsupportedBackendError, create_telemetry_store


def test_default_sqlite(tmp_path):
    s = create_telemetry_store(backend="sqlite", sqlite_db_path=tmp_path / "t.db")
    s.init_schema()
    assert s.fetch_latest_reading() is None
    s.close()


def test_postgres_empty_dsn_fast_fails():
    with pytest.raises(ValueError, match="POSTGRES_DSN"):
        create_telemetry_store(backend="postgres", postgres_dsn="")


def test_unknown_backend_raises():
    with pytest.raises(UnsupportedBackendError):
        create_telemetry_store(backend="mongodb")
