"""Read-only/write-thin wrappers around the telemetry store for the Taipy GUI.

The GUI never opens GPIO or runs Celery tasks synchronously. All hot-path
data comes via this module. A short TTL cache on the latest reading reduces
DB hits when many sessions tick at the same time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.storage.factory import create_telemetry_store

# Lazy-singleton store for the GUI process.
_store: Any = None
_LATEST_CACHE: dict[str, Any] = {"ts": 0.0, "row": None}
_LATEST_CACHE_TTL_S = 0.5


def _get_store():
    global _store
    if _store is None:
        _store = create_telemetry_store()
        _store.init_schema()
    return _store


@dataclass
class LiveSnapshot:
    latest: dict[str, Any] = field(default_factory=dict)
    last_age_s: float | None = None
    last_prediction: dict[str, Any] | None = None
    events_recent: dict[str, list] = field(default_factory=dict)       # dict-of-arrays for Taipy table
    proofs_recent: dict[str, list] = field(default_factory=dict)
    recommendations: dict[str, list] = field(default_factory=dict)


def get_latest_reading() -> dict[str, Any]:
    now = time.time()
    if now - _LATEST_CACHE["ts"] < _LATEST_CACHE_TTL_S and _LATEST_CACHE["row"] is not None:
        return _LATEST_CACHE["row"]
    store = _get_store()
    row = store.fetch_latest_reading() or {}
    _LATEST_CACHE["ts"] = now
    _LATEST_CACHE["row"] = row
    return row


def get_recent_readings_df(minutes: int = 60) -> pd.DataFrame:
    store = _get_store()
    rows = store.fetch_recent_readings(minutes=minutes)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ts_unix" in df.columns:
        df["ts_dt"] = pd.to_datetime(df["ts_unix"], unit="s", utc=True).dt.tz_convert(None)
    return df


def get_readings_range_df(
    from_unix: float, to_unix: float, max_points: int = 2000
) -> pd.DataFrame:
    """Return a readings DataFrame for the given range with auto-downsampling.

    max_points protects the frontend from rendering 100k+ points: if span/1s
    exceeds max_points, step_s is chosen so the result has <= max_points rows.
    """
    span = max(1.0, float(to_unix) - float(from_unix))
    step_s = 0.0
    if span > max_points:
        step_s = span / max_points
    store = _get_store()
    rows = store.fetch_readings_range(float(from_unix), float(to_unix), step_s)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ts_unix" in df.columns:
        df["ts_dt"] = pd.to_datetime(df["ts_unix"], unit="s", utc=True).dt.tz_convert(None)
    return df


def get_latest_prediction() -> dict[str, Any] | None:
    store = _get_store()
    pred = store.fetch_latest_prediction()
    if not pred:
        return None
    forecast = pred.get("forecast") or pred.get("forecast_json")
    if isinstance(forecast, str):
        import json
        try:
            forecast = json.loads(forecast)
        except Exception:
            forecast = []
    if isinstance(forecast, list) and forecast:
        head = forecast[0]
        pred["display_value"] = head.get("value")
        # Support both the old (`ts_offset_min`) and the new (`horizon_min`) format.
        pred["display_horizon_min"] = (
            head.get("horizon_min")
            or head.get("ts_offset_min")
            or pred.get("horizon_min")
        )
    pred["forecast_list"] = forecast if isinstance(forecast, list) else []
    return pred


def get_recent_events(limit: int = 50, event_type: str | None = None) -> list[dict[str, Any]]:
    store = _get_store()
    return store.fetch_events(event_type=event_type, limit=limit)


def get_recent_proofs(limit: int = 20) -> list[dict[str, Any]]:
    store = _get_store()
    return store.fetch_proofs(limit=limit)


def rows_to_table(rows: list[dict[str, Any]] | None, columns: tuple[str, ...]) -> dict[str, list]:
    """Taipy GUI table expects dict-of-arrays: {col: [v1, v2, ...]}.

    `<|{var}|table|columns=...|>` calls pd.DataFrame(var) which requires either
    a DataFrame or dict-of-arrays; list-of-dicts is rejected with
    "If using all scalar values, you must pass an index".

    Always emits at least one row (placeholder "-") so Taipy never sees an empty
    table - empty dict-of-empty-lists also crashes the renderer.
    """
    out: dict[str, list] = {c: [] for c in columns}
    if rows:
        for row in rows:
            for c in columns:
                v = row.get(c) if isinstance(row, dict) else None
                out[c].append("" if v is None else v)
    if not out[columns[0]]:
        for c in columns:
            out[c].append("-")
    return out


# Column whitelists used by the GUI (keep in sync with table markdown columns)
EVENT_COLS = ("ts_iso", "event_type", "source", "message", "value")
PROOF_COLS = ("idempotency_key", "pinata_cid", "hedera_sequence_number", "hedera_consensus_ts")
RECOMMENDATION_COLS = ("icon", "severity", "text")


def build_live_snapshot() -> LiveSnapshot:
    """One snapshot for the broadcast tick - bundles everything the overview shows."""
    latest = get_latest_reading()
    last_age: float | None = None
    if latest and "ts_unix" in latest:
        last_age = max(0.0, time.time() - float(latest["ts_unix"]))
    last_prediction = get_latest_prediction()
    events = get_recent_events(limit=20)
    proofs = get_recent_proofs(limit=10)
    recs = build_recommendations(latest, last_prediction)
    return LiveSnapshot(
        latest=latest,
        last_age_s=last_age,
        last_prediction=last_prediction,
        events_recent=rows_to_table(events, EVENT_COLS),
        proofs_recent=rows_to_table(proofs, PROOF_COLS),
        recommendations=rows_to_table(recs, RECOMMENDATION_COLS),
    )


def build_recommendations(
    latest: dict[str, Any] | None,
    prediction: dict[str, Any] | None,
    runtime_cfg: Any = None,
) -> list[dict[str, Any]]:
    """Rule-based, advisory recommendations from latest readings + prediction.

    Reads thresholds from configs/runtime.yml (RuntimeConfig). Returns a list
    of {severity, icon, text} dicts. NEVER triggers actions - UI displays only;
    the operator decides. Severity levels: alert > warn > info.
    """
    if runtime_cfg is None:
        try:
            from pathlib import Path

            from src.config import load_runtime_config
            runtime_cfg = load_runtime_config(Path("configs/runtime.yml"))
        except Exception:
            return []

    thr = runtime_cfg.thresholds
    out: list[dict[str, Any]] = []
    latest = latest or {}

    # --- temp_c (current + predicted) ---
    cur_temp = latest.get("temp_c")
    if cur_temp is not None:
        try:
            cur_temp = float(cur_temp)
        except (TypeError, ValueError):
            cur_temp = None

    pred_value = None
    if prediction is not None:
        pv = prediction.get("display_value")
        if pv is not None:
            try:
                pred_value = float(pv)
            except (TypeError, ValueError):
                pred_value = None

    if cur_temp is not None and cur_temp > thr.temp_c_max:
        out.append({
            "severity": "alert", "icon": "🔥",
            "text": f"Overheating: {cur_temp:.1f} C > limit {thr.temp_c_max:.0f} C - turn the fan on (relay 1)",
        })
    elif pred_value is not None and pred_value > thr.temp_c_max - 2:
        out.append({
            "severity": "warn", "icon": "🌡️",
            "text": f"Forecast {pred_value:.1f} C approaching limit {thr.temp_c_max:.0f} C - turn fan on ch1 preventively",
        })
    if cur_temp is not None and cur_temp < thr.temp_c_min:
        out.append({
            "severity": "warn", "icon": "❄️",
            "text": f"Undercooling: {cur_temp:.1f} C < limit {thr.temp_c_min:.0f} C - check heating",
        })

    # --- humidity_pct (BME280 only; BMP280 returns None) ---
    hum = latest.get("humidity_pct")
    if hum is not None:
        try:
            hum = float(hum)
            if hum > thr.humidity_pct_max:
                out.append({
                    "severity": "warn", "icon": "💧",
                    "text": f"Humidity {hum:.0f}% > {thr.humidity_pct_max:.0f}% - condensation risk",
                })
        except (TypeError, ValueError):
            pass

    # --- mq2 (gas) ---
    mq2 = latest.get("mq2_raw")
    if mq2 is not None:
        try:
            mq2 = float(mq2)
            if mq2 > thr.mq2_raw_alert:
                out.append({
                    "severity": "alert", "icon": "💨",
                    "text": f"MQ-2={mq2:.2f} > alert limit {thr.mq2_raw_alert:.2f} - evacuate / cut loads",
                })
            elif mq2 > thr.mq2_raw_warn:
                out.append({
                    "severity": "warn", "icon": "🟡",
                    "text": f"MQ-2={mq2:.2f} > warning limit {thr.mq2_raw_warn:.2f} - increase ventilation",
                })
        except (TypeError, ValueError):
            pass

    # --- current ---
    cur_ma = latest.get("current_ma")
    if cur_ma is not None:
        try:
            cur_ma = float(cur_ma)
            if cur_ma > thr.current_ma_max:
                out.append({
                    "severity": "alert", "icon": "⚡",
                    "text": f"Current {cur_ma:.0f} mA > limit {thr.current_ma_max:.0f} mA - circuit overload",
                })
        except (TypeError, ValueError):
            pass

    # --- lux ---
    lux = latest.get("lux")
    if lux is not None:
        try:
            lux = float(lux)
            if lux < thr.lux_min:
                out.append({
                    "severity": "info", "icon": "💡",
                    "text": f"Illuminance {lux:.0f} lx < {thr.lux_min:.0f} lx - dim, check source",
                })
        except (TypeError, ValueError):
            pass

    if not out:
        out.append({
            "severity": "info", "icon": "✅",
            "text": "All parameters within normal range",
        })

    return out


# --- write-side: enqueue commands; the Celery hw-worker applies them ---


def enqueue_command(actuator: str, action: str, channel: int | None = None, value: float | None = None) -> int:
    store = _get_store()
    return store.create_command(actuator=actuator, action=action, channel=channel, value=value)


def send_kill_switch() -> dict[str, Any]:
    """Send kill switch via Celery - high-priority queue, no commands-table delay."""
    from src.tasks.celery_app import celery_app

    res = celery_app.send_task(
        "src.tasks.hardware.kill_switch_task",
        queue="hardware_priority",
        priority=9,
    )
    return {"task_id": res.id}
