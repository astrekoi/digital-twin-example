"""Taipy data_access wrappers - smoke tests over a clean SQLite store."""
from __future__ import annotations

import time

from app import data_access


def test_get_latest_reading_empty(monkeypatch, tmp_path):
    db = tmp_path / "telemetry.db"
    monkeypatch.setattr(data_access, "_store", None)
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", str(db))
    monkeypatch.setattr(data_access, "_LATEST_CACHE", {"ts": 0.0, "row": None})
    row = data_access.get_latest_reading()
    assert row == {}


def test_build_live_snapshot_with_one_row(monkeypatch, tmp_path):
    from src.storage.factory import create_telemetry_store

    db = tmp_path / "telemetry.db"
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", str(db))
    monkeypatch.setattr(data_access, "_store", None)
    monkeypatch.setattr(data_access, "_LATEST_CACHE", {"ts": 0.0, "row": None})

    seed = create_telemetry_store(backend="sqlite", sqlite_db_path=db)
    seed.init_schema()
    now = time.time()
    seed.insert_reading({
        "ts_iso": "2026-04-29T22:00:00+00:00",
        "ts_unix": now,
        "temp_c": 22.5,
        "humidity_pct": 50.0,
        "pressure_hpa": 1013.0,
        "lux": None,
        "current_ma": None,
        "bus_voltage": None,
        "shunt_mv": None,
        "power_mw": None,
        "ds18b20_c": None,
        "mq2_raw": None,
        "mq2_voltage": None,
        "pir_state": None,
    })
    seed.close()

    snap = data_access.build_live_snapshot()
    assert snap.latest["temp_c"] == 22.5
    assert snap.last_age_s is not None
    assert snap.last_age_s < 5.0
    assert snap.last_prediction is None
    assert snap.events_recent == []
    assert snap.proofs_recent == []
