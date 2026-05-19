"""Factory for telemetry storage backends.

DB_BACKEND=sqlite (default) -> SQLiteTelemetryStore.
DB_BACKEND=postgres + POSTGRES_DSN -> PostgresTelemetryStore (role-aware pool).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from src.config import DatabaseBackendConfig, load_env_config

from .sqlite_backend import SQLiteTelemetryStore

TelemetryStoreInstance = Union[SQLiteTelemetryStore, "PostgresTelemetryStoreType"]  # noqa: F821


class UnsupportedBackendError(ValueError):
    """Raised when DB_BACKEND is unknown."""


def create_telemetry_store(
    config: DatabaseBackendConfig | None = None,
    *,
    backend: str | None = None,
    sqlite_db_path: Path | str | None = None,
    postgres_dsn: str | None = None,
):
    """Create a telemetry store based on env config or explicit args."""
    if config is None:
        config = load_env_config().database

    selected = backend or config.backend
    if selected == "sqlite":
        path = Path(sqlite_db_path) if sqlite_db_path is not None else config.sqlite_db_path
        return SQLiteTelemetryStore(path)
    if selected == "postgres":
        if postgres_dsn is not None:
            dsn = postgres_dsn
        else:
            dsn = config.postgres_dsn or ""
        if not dsn:
            raise ValueError(
                "DB_BACKEND=postgres requires POSTGRES_DSN to be set in .env"
            )
        from .postgres_backend import PostgresTelemetryStore
        return PostgresTelemetryStore(dsn)
    raise UnsupportedBackendError(f"Unsupported DB_BACKEND: {selected!r}")
