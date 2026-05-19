"""Tests for the multi-horizon inference path (artifacts schema, loader bypass,
predictor dispatch, predict_task wiring).

All tests are mock-based - no real torch / darts / lightgbm import is required.
The PyTorch / Darts branches are NOT covered here; their integration is verified
separately via the manual smoke loop on the Pi.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Schema: validate_metadata_dict( horizons_min )
# ---------------------------------------------------------------------------


def _base_meta(**overrides) -> dict:
    """Minimal valid meta dict for the validator. Override fields per test."""
    base = {
        "model_id": "test_model",
        "algo": "xgboost",
        "trained_at": "2026-05-01T00:00:00Z",
        "features": ["temp_c", "humidity_pct"],
        "target": "temp_c",
        "metrics": {"rmse": 0.42},
        "train_rows": 1000,
    }
    base.update(overrides)
    return base


def test_artifacts_horizons_min_valid_sorted():
    from src.ai.artifacts import validate_metadata_dict

    msgs = validate_metadata_dict(_base_meta(horizons_min=[1, 5, 10, 15, 30, 60]))
    errors = [m for m in msgs if m.startswith("ERROR:")]
    assert errors == [], f"unexpected errors: {errors}"


def test_artifacts_horizons_min_unsorted_rejected():
    from src.ai.artifacts import validate_metadata_dict

    msgs = validate_metadata_dict(_base_meta(horizons_min=[60, 5, 15]))
    errors = [m for m in msgs if m.startswith("ERROR:")]
    assert any("sorted ascending" in e for e in errors), f"got: {errors}"


def test_artifacts_horizons_min_empty_rejected():
    from src.ai.artifacts import validate_metadata_dict

    msgs = validate_metadata_dict(_base_meta(horizons_min=[]))
    errors = [m for m in msgs if m.startswith("ERROR:")]
    assert any("non-empty list" in e for e in errors), f"got: {errors}"


def test_artifacts_horizons_min_out_of_range_rejected():
    from src.ai.artifacts import validate_metadata_dict

    msgs = validate_metadata_dict(_base_meta(horizons_min=[1, 5, 9999]))
    errors = [m for m in msgs if m.startswith("ERROR:")]
    assert any("out of range" in e for e in errors), f"got: {errors}"


def test_artifacts_horizons_min_mutex_with_horizon_min():
    from src.ai.artifacts import validate_metadata_dict

    msgs = validate_metadata_dict(
        _base_meta(horizons_min=[1, 5, 15], horizon_min=15)
    )
    errors = [m for m in msgs if m.startswith("ERROR:")]
    assert any("mutually exclusive" in e for e in errors), f"got: {errors}"


# ---------------------------------------------------------------------------
# Loader: persistence bypass
# ---------------------------------------------------------------------------


def test_loader_persistence_bypass_returns_none_model():
    """algo='persistence' must skip joblib.load entirely and return (None, meta)."""
    from src.ai import loader as loader_mod

    fake_entry = {
        "model_id": "persistence_temp_real_v1",
        "model_path": loader_mod.Path("models/persistence_temp_real_v1.joblib"),
        "joblib_path": loader_mod.Path("models/persistence_temp_real_v1.joblib"),
        "meta_path": loader_mod.Path("models/persistence_temp_real_v1.meta.json"),
        "meta": {
            "model_id": "persistence_temp_real_v1",
            "algo": "persistence",
            "target": "temp_c",
            "horizons_min": [1, 5, 10, 15, 30, 60],
        },
        "trained_at": "2026-04-30T16:15:10Z",
        "mtime": 1.0,
    }

    load_joblib_calls: list = []

    def fake_load_joblib(path):
        load_joblib_calls.append(path)
        raise AssertionError("_load_joblib must NOT be called for persistence")

    with patch.object(loader_mod, "list_available_models", return_value=[fake_entry]):
        with patch.object(loader_mod, "_load_joblib", side_effect=fake_load_joblib):
            model, meta = loader_mod.load_model(
                model_id="persistence_temp_real_v1", models_dir="models"
            )

    assert model is None
    assert meta["algo"] == "persistence"
    assert load_joblib_calls == [], "joblib.load was called for persistence"


# ---------------------------------------------------------------------------
# Predictor.predict_multi() dispatch
# ---------------------------------------------------------------------------


def _make_predictor(model, meta, payload=None):
    """Build a Predictor instance bypassing reload() (no real model file)."""
    from src.ai.predictor import Predictor

    p = Predictor.__new__(Predictor)
    p._models_dir = None  # type: ignore[attr-defined]
    p._model_id = None  # type: ignore[attr-defined]
    p.model = model
    p.meta = meta
    p._payload = payload if payload is not None else model
    p._torch_module = None
    p._darts_module = None
    return p


def test_predict_multi_persistence_echoes_last_observed():
    horizons = [1, 5, 10, 15, 30, 60]
    meta = {
        "model_id": "persistence_temp_real_v1",
        "algo": "persistence",
        "target": "temp_c",
        "horizons_min": horizons,
    }
    p = _make_predictor(model=None, meta=meta)

    df = pd.DataFrame({
        "ts_unix": np.arange(100, dtype=float),
        "temp_c": np.linspace(20.0, 23.456, 100),
    })
    fc = p.predict_multi(df)

    assert len(fc) == len(horizons)
    expected_value = pytest.approx(23.456)
    for f, h in zip(fc, horizons, strict=True):
        assert f["horizon_min"] == h
        assert f["value"] == expected_value


def test_predict_multi_xgboost_dict_dispatch_calls_each_horizon():
    horizons = [1, 5, 15, 60]
    feature_columns = ["temp_c", "humidity_pct"]
    meta = {
        "model_id": "xgb_temp_real_v2",
        "algo": "xgboost",
        "target": "temp_c",
        "horizons_min": horizons,
        "feature_columns": feature_columns,
        "features": feature_columns,
    }
    # Per-horizon mock regressors returning a horizon-specific value so we can
    # verify dispatch picked the right model for the right horizon.
    mocks: dict = {}
    for h in horizons:
        m = MagicMock(name=f"reg_h{h}")
        m.predict.return_value = np.array([float(h) + 0.1])
        mocks[h] = m

    p = _make_predictor(model=mocks, meta=meta, payload=mocks)

    df = pd.DataFrame({
        "ts_unix": np.arange(10, dtype=float),
        "temp_c": np.linspace(20.0, 22.0, 10),
        "humidity_pct": np.linspace(40.0, 50.0, 10),
    })
    fc = p.predict_multi(df)

    assert [f["horizon_min"] for f in fc] == horizons
    for f, h in zip(fc, horizons, strict=True):
        assert f["value"] == pytest.approx(float(h) + 0.1), f"wrong dispatch for h={h}"
    for h in horizons:
        mocks[h].predict.assert_called_once()


def test_predict_multi_prophet_uses_make_future_dataframe():
    horizons = [1, 5, 15, 30, 60]
    meta = {
        "model_id": "prophet_temp_real_v2",
        "algo": "prophet",
        "target": "temp_c",
        "horizons_min": horizons,
        "features": ["ds"],
    }
    fake_forecast = pd.DataFrame({
        "yhat": [10.0 + i * 0.5 for i in range(60)],
    })
    prophet_model = MagicMock()
    prophet_model.make_future_dataframe = MagicMock(return_value=pd.DataFrame())
    prophet_model.predict.return_value = fake_forecast

    p = _make_predictor(model=prophet_model, meta=meta)
    df = pd.DataFrame({"ts_unix": [1.0, 2.0, 3.0]})
    fc = p.predict_multi(df)

    prophet_model.make_future_dataframe.assert_called_once_with(
        periods=60, freq="min", include_history=False,
    )
    assert [f["horizon_min"] for f in fc] == horizons
    for f, h in zip(fc, horizons, strict=True):
        assert f["value"] == pytest.approx(fake_forecast.iloc[h - 1]["yhat"])


def test_predict_multi_returns_empty_on_unsupported_algo():
    meta = {
        "model_id": "weirdo",
        "algo": "magic_brand_new_model",
        "target": "temp_c",
        "horizons_min": [1, 5, 15],
    }
    p = _make_predictor(model=MagicMock(), meta=meta)
    df = pd.DataFrame({"temp_c": [20.0, 21.0]})
    assert p.predict_multi(df) == []


# ---------------------------------------------------------------------------
# Task: predict_task writes full forecast_list when horizons_min present
# ---------------------------------------------------------------------------


def test_predict_task_writes_full_forecast_list(monkeypatch, sqlite_store):
    """predict_task must persist the multi-point forecast as JSON."""
    from src.tasks import ai as tasks_ai

    horizons = [1, 5, 10, 15, 30, 60]
    fake_forecast = [
        {"horizon_min": h, "value": 21.0 + i * 0.1}
        for i, h in enumerate(horizons)
    ]

    fake_predictor = MagicMock()
    fake_predictor.meta = {
        "model_id": "fake_multi_v1",
        "algo": "xgboost",
        "target": "temp_c",
        "horizons_min": horizons,
        "features": ["temp_c"],
        "feature_columns": ["temp_c"],
    }
    fake_predictor.predict_multi = MagicMock(return_value=fake_forecast)

    # Inject readings so fetch_recent_readings(minutes=180) returns non-empty.
    # Schema requires every reading column to be bound - provide all 14.
    import time as _time
    now = _time.time()
    for i in range(120):
        sqlite_store.insert_reading({
            "ts_iso": f"2026-05-01T00:00:{i:02d}Z",
            "ts_unix": now - 120 + i,  # last 2 min of "now" so fetch_recent picks them up
            "temp_c": 20.0 + i * 0.01,
            "humidity_pct": 50.0,
            "pressure_hpa": 1013.0,
            "lux": 100.0,
            "current_ma": 0.0,
            "bus_voltage": 5.0,
            "shunt_mv": 0.0,
            "power_mw": 0.0,
            "ds18b20_c": None,
            "mq2_raw": 0.2,
            "mq2_voltage": 1.0,
            "pir_state": 0,
        })

    monkeypatch.setitem(tasks_ai._state, "predictor", fake_predictor)
    monkeypatch.setitem(tasks_ai._state, "store", sqlite_store)
    monkeypatch.setitem(tasks_ai._state, "model_id", "fake_multi_v1")

    result = tasks_ai.predict_task()

    assert result["status"] == "ok"
    assert result["points"] == len(horizons)
    assert result["primary_horizon"] == 15

    fake_predictor.predict_multi.assert_called_once()
    latest = sqlite_store.fetch_latest_prediction()
    assert latest is not None
    forecast_json = latest.get("forecast") or latest.get("forecast_json")
    if isinstance(forecast_json, str):
        import json
        forecast_json = json.loads(forecast_json)
    assert isinstance(forecast_json, list)
    assert len(forecast_json) == len(horizons)
    assert [item["horizon_min"] for item in forecast_json] == horizons
