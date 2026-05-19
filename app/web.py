"""Taipy GUI - Grafana-inspired dashboard.

Principles:
- Live data is pushed via broadcast_callback every second, but ONLY KPI
  numbers and tables. Charts refresh manually or once every 30 s.
- uirevision="static" in the plotly layout preserves zoom/pan across replots.
- Settings actionable: thresholds + AI/integrations toggles -> runtime.yml.
- All technical terms (queue/broker/task) are hidden from the operator.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import plotly.graph_objects as go
import yaml
from taipy.gui import Gui, Markdown, navigate

from app import data_access, refresh, theme
from src.config import load_env_config, load_runtime_config
from src.env import load_project_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_project_env(PROJECT_ROOT / ".env")
_env = load_env_config(PROJECT_ROOT / ".env")
_runtime_path = PROJECT_ROOT / "configs" / "runtime.yml"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State variables (module-level)
# ---------------------------------------------------------------------------

# Live KPIs - pre-formatted strings + status colour
data_status: str = "loading..."
data_status_color: str = theme.PALETTE["muted"]
age_display: str = "-"

temp_value: str = "-"
temp_color: str = theme.PALETTE["info"]
humidity_value: str = "-"
humidity_color: str = theme.PALETTE["info"]
pressure_value: str = "-"
pressure_color: str = theme.PALETTE["info"]
lux_value: str = "-"
lux_color: str = theme.PALETTE["info"]
current_value: str = "-"
current_color: str = theme.PALETTE["info"]
mq2_value: str = "-"
mq2_color: str = theme.PALETTE["info"]
pir_value: str = "-"
pir_color: str = theme.PALETTE["info"]

# Forecast
prediction_value_display: str = "-"
prediction_meta: str = "Forecast not built yet"

# Tables (Taipy expects dict-of-arrays)
events_recent: dict = {"ts_iso": ["-"], "event_type": ["-"], "source": ["-"], "message": ["loading..."], "value": [""]}
proofs_recent: dict = {"idempotency_key": ["-"], "pinata_cid": ["-"], "hedera_sequence_number": ["-"], "hedera_consensus_ts": ["-"]}
recommendations: dict = {"icon": ["OK"], "severity": ["info"], "text": ["All parameters within normal range"]}

# Charts (Grafana-style: 5 separate panels per metric)
chart_range_minutes: int = 60
# Custom date-range mode (Grafana absolute range). When chart_use_custom_range=True,
# chart_range_minutes is ignored and chart_date_from/to are used instead.
chart_use_custom_range: bool = False
chart_date_from: datetime = datetime.now() - timedelta(hours=1)
chart_date_to: datetime = datetime.now()
chart_range_label: str = "last 60 min"
chart_temp:     go.Figure = go.Figure(layout=theme.plotly_theme())
chart_humidity: go.Figure = go.Figure(layout=theme.plotly_theme())
chart_pressure: go.Figure = go.Figure(layout=theme.plotly_theme())
chart_lux:      go.Figure = go.Figure(layout=theme.plotly_theme())
chart_current:  go.Figure = go.Figure(layout=theme.plotly_theme())
chart_mq2:      go.Figure = go.Figure(layout=theme.plotly_theme())
forecast_chart: go.Figure = go.Figure(layout=theme.plotly_theme())

# Control
relay_1: bool = False
relay_2: bool = False
relay_3: bool = False
relay_4: bool = False
servo_angle: float = 90.0
servo_channel: int = 1
last_command_at: float = 0.0

# Settings (loaded from runtime.yml - editable!)
def _list_available_models() -> list[str]:
    """Discover model_ids from models/*.meta.json (sorted by trained_at desc)."""
    try:
        from src.ai.registry import list_available_models
        entries = list_available_models("models")
        return [e["model_id"] for e in entries]
    except Exception:
        return []


available_models: list = _list_available_models() or [_env.ai.model_id or "-"]
selected_model: str = _env.ai.model_id or available_models[0]


_runtime = load_runtime_config(_runtime_path)
threshold_temp_max: float = float(_runtime.thresholds.temp_c_max)
threshold_temp_min: float = float(_runtime.thresholds.temp_c_min)
threshold_humidity_max: float = float(_runtime.thresholds.humidity_pct_max)
threshold_mq2_warn: float = float(_runtime.thresholds.mq2_raw_warn)
threshold_mq2_alert: float = float(_runtime.thresholds.mq2_raw_alert)
threshold_current_max: float = float(_runtime.thresholds.current_ma_max)
threshold_lux_min: float = float(_runtime.thresholds.lux_min)

# Integrations status (read-only display)
ai_enabled_display: str = "enabled" if _env.ai.enabled else "disabled"
ipfs_provider_display: str = _env.integrations.ipfs_provider
hedera_enabled_display: str = "enabled" if _env.hedera.enabled else "disabled"
hedera_topic_id: str = _env.hedera.topic_id or "-"
external_calls_display: str = "allowed" if _env.integrations.allow_external_api_calls else "BLOCKED"

# Hedera info panel
hedera_last_publish: str = "-"
hedera_last_status: str = "-"
hedera_next_publish: str = "-"

last_error: str = ""

_RATE_LIMIT_S = 2.0
_DIGEST_PERIOD_S = float(_env.celery.digest_period_s)


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------


def _fmt(value, fmt: str = ".1f", suffix: str = "") -> str:
    if value is None:
        return "-"
    try:
        return f"{format(float(value), fmt)}{suffix}"
    except (TypeError, ValueError):
        return "-"


def _temp_severity(v: float | None) -> str:
    if v is None:
        return "info"
    if v >= threshold_temp_max or v <= threshold_temp_min:
        return "alert"
    if v >= threshold_temp_max - 2 or v <= threshold_temp_min + 2:
        return "warn"
    return "ok"


def _humidity_severity(v: float | None) -> str:
    if v is None:
        return "info"
    if v >= threshold_humidity_max:
        return "alert"
    if v >= threshold_humidity_max - 5:
        return "warn"
    return "ok"


def _mq2_severity(v: float | None) -> str:
    if v is None:
        return "info"
    if v >= threshold_mq2_alert:
        return "alert"
    if v >= threshold_mq2_warn:
        return "warn"
    return "ok"


def _current_severity(v: float | None) -> str:
    if v is None:
        return "info"
    if v >= threshold_current_max:
        return "alert"
    if v >= threshold_current_max * 0.8:
        return "warn"
    return "ok"


def _lux_severity(v: float | None) -> str:
    if v is None:
        return "info"
    if v < threshold_lux_min:
        return "warn"
    return "ok"


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Charts builders
# ---------------------------------------------------------------------------


def _single_metric_chart(df, col: str, label: str, color: str, suffix: str = "") -> go.Figure:
    """One metric, one panel (Grafana-style). Compact layout for stacked grid."""
    fig = go.Figure(layout={**theme.plotly_theme(), "margin": {"t": 30, "r": 20, "b": 30, "l": 50},
                            "showlegend": False, "title": {"text": f"{label}{suffix}", "font": {"size": 14, "color": theme.PALETTE["text"]}}})
    if df is None or df.empty or "ts_dt" not in df.columns or col not in df.columns or not df[col].notna().any():
        fig.update_layout(title=f"{label} - no data")
        return fig
    fig.add_trace(go.Scatter(
        x=df["ts_dt"], y=df[col], mode="lines", name=label,
        line={"color": color, "width": 2},
        fill="tozeroy" if col in ("lux", "mq2_raw", "current_ma") else None,
        fillcolor=f"rgba({_hex_to_rgb(color)},0.08)" if col in ("lux", "mq2_raw", "current_ma") else None,
    ))
    return fig


def _hex_to_rgb(hex_color: str) -> str:
    hc = hex_color.lstrip("#")
    return f"{int(hc[0:2],16)},{int(hc[2:4],16)},{int(hc[4:6],16)}"


def _charts_from_df(df) -> dict[str, go.Figure]:
    return {
        "temp":     _single_metric_chart(df, "temp_c",       "Temperature",   theme.PALETTE["series_temp"],     " (C)"),
        "humidity": _single_metric_chart(df, "humidity_pct", "Humidity",      theme.PALETTE["series_humidity"], " (%)"),
        "pressure": _single_metric_chart(df, "pressure_hpa", "Pressure",      theme.PALETTE["series_pressure"], " (hPa)"),
        "lux":      _single_metric_chart(df, "lux",          "Illuminance",   theme.PALETTE["series_lux"],      " (lx)"),
        "current":  _single_metric_chart(df, "current_ma",   "Load current",  theme.PALETTE["series_current"],  " (mA)"),
        "mq2":      _single_metric_chart(df, "mq2_raw",      "MQ-2 gas",      "#ff8c00",                        " (raw)"),
    }


def _build_all_metric_charts(minutes: int) -> dict[str, go.Figure]:
    """Quick-range variant: last N minutes."""
    df = data_access.get_recent_readings_df(minutes=minutes)
    return _charts_from_df(df)


def _build_charts_from_state(state) -> dict[str, go.Figure]:
    """Single source of truth for building per-state charts.

    If state.chart_use_custom_range=True - use chart_date_from..to; otherwise
    fall back to last chart_range_minutes minutes. This function is called
    from user callbacks (buttons) and from _broadcast_slow (auto-refresh) -
    so auto-refresh no longer resets the user's selected range.
    """
    use_custom = bool(getattr(state, "chart_use_custom_range", False))
    if use_custom:
        df = _df_for_custom_range(
            getattr(state, "chart_date_from", None),
            getattr(state, "chart_date_to", None),
        )
    else:
        df = data_access.get_recent_readings_df(
            minutes=int(getattr(state, "chart_range_minutes", 60))
        )
    return _charts_from_df(df)


def _to_unix(value) -> float | None:
    """Accept datetime/date/str(ISO)/number -> unix seconds, or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).timestamp()
        return value.timestamp()
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except Exception:
            return None
    # date (no time component)
    try:
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc).timestamp()
    except Exception:
        return None


def _df_for_custom_range(date_from, date_to):
    """Load readings for the given range, guarding against empty or inverted values."""
    f = _to_unix(date_from)
    t = _to_unix(date_to)
    if f is None or t is None:
        return data_access.get_recent_readings_df(minutes=60)
    if t < f:
        f, t = t, f
    if (t - f) < 60:
        t = f + 60
    return data_access.get_readings_range_df(f, t)


# Backwards-compat alias for places that still call _build_history_chart
def _build_history_chart(minutes: int) -> go.Figure:
    return _build_all_metric_charts(minutes)["temp"]


def _build_forecast_chart(prediction: dict | None) -> go.Figure:
    """Temperature history (last 60 min) + multi-point interpolated forecast."""
    df = data_access.get_recent_readings_df(minutes=60)
    fig = go.Figure(layout=theme.plotly_theme())
    if df.empty or "ts_dt" not in df.columns or "temp_c" not in df.columns:
        fig.update_layout(title="No data")
        return fig

    # History
    fig.add_trace(go.Scatter(
        x=df["ts_dt"], y=df["temp_c"], mode="lines",
        name="Temperature (history)",
        line={"color": theme.PALETTE["series_temp"], "width": 2.4},
    ))

    if not prediction:
        fig.update_layout(title="Forecast not built yet")
        return fig

    forecast_list = prediction.get("forecast_list") or []
    gen_at = prediction.get("generated_at")
    if not (gen_at and forecast_list):
        return fig

    try:
        base_dt = datetime.fromtimestamp(float(gen_at), tz=timezone.utc).replace(tzinfo=None)
    except Exception:
        return fig

    from datetime import timedelta

    # Multi-point timeline: split [now -> +max_horizon] into 6 points,
    # linearly interpolating between current temp and forecast value.
    cur_temp = _safe_float(df["temp_c"].iloc[-1]) if not df.empty else None
    cur_dt = df["ts_dt"].iloc[-1] if not df.empty else base_dt

    points_x: list = [cur_dt]
    points_y: list = [cur_temp]

    max_horizon = max(
        int(f.get("horizon_min") or f.get("ts_offset_min") or 0)
        for f in forecast_list
    ) or 15

    # When several forecast points exist - plot each and fill in by
    # interpolating between them.
    sorted_forecast = sorted(
        forecast_list,
        key=lambda f: int(f.get("horizon_min") or f.get("ts_offset_min") or 0),
    )
    for f in sorted_forecast:
        h = int(f.get("horizon_min") or f.get("ts_offset_min") or 0)
        v = _safe_float(f.get("value"))
        if v is None:
            continue
        points_x.append(base_dt + timedelta(minutes=h))
        points_y.append(v)

    fig.add_trace(go.Scatter(
        x=points_x, y=points_y, mode="lines+markers",
        name=f"Forecast +{max_horizon} min",
        line={"color": theme.PALETTE["series_forecast"], "width": 2.6, "dash": "dot"},
        marker={"size": 12, "color": theme.PALETTE["series_forecast"], "symbol": "diamond"},
    ))

    # Confidence band (+/-1 C, a simple visual error estimate)
    if len(points_x) >= 2:
        upper_y = [(y + 1.0) if y is not None else None for y in points_y]
        lower_y = [(y - 1.0) if y is not None else None for y in points_y]
        fig.add_trace(go.Scatter(
            x=points_x + points_x[::-1],
            y=upper_y + lower_y[::-1],
            fill="toself",
            fillcolor="rgba(115, 224, 169, 0.15)",
            line={"color": "rgba(0,0,0,0)"},
            showlegend=False,
            hoverinfo="skip",
        ))
    return fig


# ---------------------------------------------------------------------------
# Live tick (1 Hz) - KPI + tables only; charts updated separately
# ---------------------------------------------------------------------------


_SEVERITY_ICON = {"ok": "🟢", "warn": "🟡", "alert": "🔴", "info": "⚪"}


def _kpi(value: float | None, sev: str, fmt: str = ".1f", suffix: str = "") -> str:
    """Format KPI: '🟢 23.5 C' / '- -' when data is missing."""
    icon = _SEVERITY_ICON.get(sev, "⚪")
    if value is None:
        return f"{icon} -"
    try:
        return f"{icon} {format(float(value), fmt)}{suffix}"
    except (TypeError, ValueError):
        return f"{icon} -"


def on_live_tick(state, snapshot) -> None:
    """1 Hz tick: refresh KPIs and status. Tables and charts are updated elsewhere (_slow_refresh_loop)."""
    lat = snapshot.latest or {}

    # Status pill
    age = snapshot.last_age_s
    if age is None:
        state.data_status, state.data_status_color, state.age_display = "🔴 no data", theme.PALETTE["alert"], "-"
    elif age < 5:
        state.data_status, state.data_status_color, state.age_display = "🟢 online", theme.PALETTE["ok"], f"{age:.1f} s ago"
    elif age < 60:
        state.data_status, state.data_status_color, state.age_display = "🟡 stale", theme.PALETTE["warn"], f"{age:.0f} s ago"
    else:
        state.data_status, state.data_status_color, state.age_display = "🔴 offline", theme.PALETTE["alert"], f"{age/60:.0f} min ago"

    # KPIs: emoji + value (Taipy does not substitute {var} into HTML attributes,
    # so the colour indication goes through emojis - reliable on any frontend).
    t = _safe_float(lat.get("temp_c"))
    state.temp_value = _kpi(t, _temp_severity(t), ".1f", " C")

    h = _safe_float(lat.get("humidity_pct"))
    state.humidity_value = (
        _kpi(h, _humidity_severity(h), ".0f", " %") if h is not None else "⚪ - (BMP280)"
    )

    state.pressure_value = _kpi(_safe_float(lat.get("pressure_hpa")), "info", ".0f", " hPa")

    lx = _safe_float(lat.get("lux"))
    state.lux_value = _kpi(lx, _lux_severity(lx), ".0f", " lx")

    cm = _safe_float(lat.get("current_ma"))
    state.current_value = _kpi(cm, _current_severity(cm), ".0f", " mA")

    mq = _safe_float(lat.get("mq2_raw"))
    state.mq2_value = _kpi(mq, _mq2_severity(mq), ".3f")

    state.pir_value = "🟡 motion" if lat.get("pir_state") else "🟢 quiet"

    # Prediction
    pred = snapshot.last_prediction
    if pred and pred.get("display_value") is not None:
        val = pred.get("display_value")
        horizon = pred.get("display_horizon_min") or "?"
        state.prediction_value_display = f"🔵 {val:.2f} C"
        gen = pred.get("generated_at")
        gen_str = "-"
        if gen:
            try:
                gen_str = datetime.fromtimestamp(float(gen), tz=timezone.utc).strftime("%H:%M:%S")
            except Exception:
                pass
        state.prediction_meta = f"in {horizon} min | updated {gen_str}"
    else:
        state.prediction_value_display = "-"
        state.prediction_meta = "Forecast not built yet"

    # Hedera info panel (computed from state, not from snapshot.proofs_recent -
    # the latter is refreshed separately).
    now = time.time()
    next_ts = now - (now % _DIGEST_PERIOD_S) + _DIGEST_PERIOD_S
    wait = int(next_ts - now)
    state.hedera_next_publish = f"in ~{wait // 60} min {wait % 60} s"


def on_slow_tick(state, snapshot) -> None:
    """30 s tick: refresh tables (events / proofs / recommendations) and charts.

    Kept separate from on_live_tick - otherwise tables would jitter every
    second and the user could not click or scroll inside them.
    """
    state.events_recent = snapshot.events_recent
    state.proofs_recent = snapshot.proofs_recent
    state.recommendations = snapshot.recommendations
    # Charts are handled by _slow_refresh_loop via broadcast_callback.
    proofs = snapshot.proofs_recent or {}
    if proofs.get("hedera_consensus_ts") and proofs["hedera_consensus_ts"][0] not in ("-", "", None):
        ts_unix = proofs["hedera_consensus_ts"][0]
        try:
            ts_dt = datetime.fromtimestamp(float(ts_unix), tz=timezone.utc)
            state.hedera_last_publish = ts_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            state.hedera_last_status = (
                "ok" if proofs["pinata_cid"][0] not in ("-", "", None)
                else "Hedera only (Pinata unavailable)"
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Charts: refreshed on demand (NOT every tick - would break user interaction)
# ---------------------------------------------------------------------------


def _refresh_all_charts(state) -> None:
    charts = _build_charts_from_state(state)
    state.chart_temp     = charts["temp"]
    state.chart_humidity = charts["humidity"]
    state.chart_pressure = charts["pressure"]
    state.chart_lux      = charts["lux"]
    state.chart_current  = charts["current"]
    state.chart_mq2      = charts["mq2"]


def on_refresh_chart(state) -> None:
    _refresh_all_charts(state)
    state.last_error = "Charts refreshed"


_QUICK_RANGES = {
    5: "last 5 min",
    15: "last 15 min",
    60: "last hour",
    360: "last 6 hours",
    1440: "last 24 hours",
    10080: "last 7 days",
}


def _set_quick_range(state, minutes: int) -> None:
    state.chart_use_custom_range = False
    state.chart_range_minutes = minutes
    state.chart_range_label = _QUICK_RANGES.get(minutes, f"last {minutes} min")
    # Auto-sync the date pickers to the chosen quick range so the user sees
    # consistent values if they switch to custom mode.
    now = datetime.now()
    state.chart_date_to = now
    state.chart_date_from = now - timedelta(minutes=minutes)
    _refresh_all_charts(state)


def on_range_5m(state) -> None:    _set_quick_range(state, 5)
def on_range_15m(state) -> None:   _set_quick_range(state, 15)
def on_range_1h(state) -> None:    _set_quick_range(state, 60)
def on_range_6h(state) -> None:    _set_quick_range(state, 360)
def on_range_24h(state) -> None:   _set_quick_range(state, 1440)
def on_range_7d(state) -> None:    _set_quick_range(state, 10080)


def on_apply_date_range(state) -> None:
    """Switch to custom-range mode and rebuild charts using chart_date_from/to."""
    f_unix = _to_unix(state.chart_date_from)
    t_unix = _to_unix(state.chart_date_to)
    if f_unix is None or t_unix is None:
        state.last_error = "Pick both dates - start and end of the range"
        return
    if t_unix <= f_unix:
        state.last_error = "End of range must be later than start"
        return
    span_min = (t_unix - f_unix) / 60.0
    if span_min > 60 * 24 * 30:
        state.last_error = "Maximum range is 30 days"
        return
    state.chart_use_custom_range = True
    f_str = datetime.fromtimestamp(f_unix).strftime("%Y-%m-%d %H:%M")
    t_str = datetime.fromtimestamp(t_unix).strftime("%Y-%m-%d %H:%M")
    state.chart_range_label = f"{f_str} -> {t_str}"
    _refresh_all_charts(state)
    state.last_error = f"Range applied ({span_min:.0f} min)"


def on_range_change(state) -> None:
    """Legacy callback for old dropdown selector - keep for back-compat."""
    _refresh_all_charts(state)


def on_refresh_forecast(state) -> None:
    """Request a fresh forecast and rebuild the chart."""
    if not _check_rate_limit(state):
        return
    try:
        from src.tasks.celery_app import celery_app
        celery_app.send_task("src.tasks.ai.predict_task", queue="ai")
        state.last_error = "Forecast refreshing (1-3 seconds)"
    except Exception as exc:
        state.last_error = f"Forecast request error: {exc}"
    snap = data_access.build_live_snapshot()
    on_live_tick(state, snap)
    state.forecast_chart = _build_forecast_chart(snap.last_prediction)


# ---------------------------------------------------------------------------
# Control callbacks
# ---------------------------------------------------------------------------


def _check_rate_limit(state) -> bool:
    now = time.time()
    if now - getattr(state, "last_command_at", 0.0) < _RATE_LIMIT_S:
        state.last_error = f"Wait {_RATE_LIMIT_S:.0f} s between actions"
        return False
    state.last_command_at = now
    return True


def _enqueue_relay(state, ch: int, on: bool) -> None:
    if not _check_rate_limit(state):
        return
    try:
        data_access.enqueue_command("relay", "on" if on else "off", channel=ch)
        state.last_error = f"Relay {ch}: {'on' if on else 'off'}"
    except Exception as exc:
        state.last_error = f"Relay {ch}: error ({exc})"


def on_relay_1(state) -> None: _enqueue_relay(state, 1, state.relay_1)
def on_relay_2(state) -> None: _enqueue_relay(state, 2, state.relay_2)
def on_relay_3(state) -> None: _enqueue_relay(state, 3, state.relay_3)
def on_relay_4(state) -> None: _enqueue_relay(state, 4, state.relay_4)


def on_servo_apply(state) -> None:
    if not _check_rate_limit(state):
        return
    try:
        data_access.enqueue_command(
            "servo", "set_angle", channel=int(state.servo_channel), value=float(state.servo_angle)
        )
        state.last_error = f"Servo {state.servo_channel} -> {state.servo_angle:.0f} deg"
    except Exception as exc:
        state.last_error = f"Servo: error ({exc})"


def _led(state, action: str) -> None:
    if not _check_rate_limit(state):
        return
    try:
        data_access.enqueue_command("leds", action)
        state.last_error = f"Traffic light: {action}"
    except Exception as exc:
        state.last_error = f"Traffic light: error ({exc})"


def on_led_red(state) -> None:    _led(state, "red")
def on_led_yellow(state) -> None: _led(state, "yellow")
def on_led_green(state) -> None:  _led(state, "green")
def on_led_off(state) -> None:    _led(state, "off")


def on_kill_switch(state) -> None:
    try:
        data_access.send_kill_switch()
        state.last_error = "KILL SWITCH sent"
    except Exception as exc:
        state.last_error = f"Kill switch failed: {exc}"


# ---------------------------------------------------------------------------
# Settings - actionable: read/edit thresholds, save back to runtime.yml
# ---------------------------------------------------------------------------


def on_save_thresholds(state) -> None:
    """Write thresholds back to configs/runtime.yml - collector hot-reload will pick them up."""
    try:
        text = _runtime_path.read_text(encoding="utf-8") if _runtime_path.exists() else ""
        try:
            data = yaml.safe_load(text) or {}
        except Exception:
            data = {}
        thr = data.setdefault("thresholds", {})
        thr["temp_c_max"] = float(state.threshold_temp_max)
        thr["temp_c_min"] = float(state.threshold_temp_min)
        thr["humidity_pct_max"] = float(state.threshold_humidity_max)
        thr["mq2_raw_warn"] = float(state.threshold_mq2_warn)
        thr["mq2_raw_alert"] = float(state.threshold_mq2_alert)
        thr["current_ma_max"] = float(state.threshold_current_max)
        thr["lux_min"] = float(state.threshold_lux_min)
        _runtime_path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        state.last_error = "Thresholds saved - will apply within a minute"
    except Exception as exc:
        state.last_error = f"Failed to save thresholds: {exc}"


def on_publish_now(state) -> None:
    """Manual trigger - publish a digest to IPFS+Hedera right now."""
    if not _check_rate_limit(state):
        return
    try:
        from src.tasks.celery_app import celery_app
        celery_app.send_task("src.tasks.external.publish_digest_task", queue="external")
        state.last_error = "Publish requested - result in ~10 seconds"
    except Exception as exc:
        state.last_error = f"Failed to request publish: {exc}"


def on_apply_model(state) -> None:
    """Switch active AI model on the ai-worker + immediately request a prediction."""
    if not _check_rate_limit(state):
        return
    new_id = state.selected_model
    if not new_id or new_id == "-":
        state.last_error = "Pick a model from the list"
        return
    try:
        from src.tasks.celery_app import celery_app
        celery_app.send_task("src.tasks.ai.set_model_task", queue="ai", args=[new_id])
        # Give the ai-worker ~1 s to reload, then immediately request a predict.
        import threading as _t
        def _delayed_predict():
            time.sleep(1.2)
            try:
                celery_app.send_task("src.tasks.ai.predict_task", queue="ai")
            except Exception:
                logger.exception("post-switch predict failed")
        _t.Thread(target=_delayed_predict, daemon=True).start()
        state.last_error = f"Model switching to {new_id} - forecast will appear in 2-4 seconds"
    except Exception as exc:
        state.last_error = f"Failed to switch model: {exc}"


def on_menu(state, action, payload) -> None:
    args = payload.get("args") if isinstance(payload, dict) else None
    target = args[0] if args else None
    if target:
        navigate(state, to=str(target))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def _read(name: str) -> str:
    return (PROJECT_ROOT / "app" / "pages" / f"{name}.md").read_text(encoding="utf-8")


ROOT_MD = """
<|menu|label=PLC Twin|lov={[("overview","Overview"),("charts","Charts"),("control","Control"),("forecast","Forecast"),("events","Events"),("settings","Settings")]}|on_action=on_menu|>
"""


import inspect  # noqa: E402
_FRAME = inspect.currentframe()

pages = {
    "/":         Markdown(ROOT_MD,           frame=_FRAME),
    "overview":  Markdown(_read("overview"), frame=_FRAME),
    "charts":    Markdown(_read("charts"),   frame=_FRAME),
    "control":   Markdown(_read("control"),  frame=_FRAME),
    "forecast":  Markdown(_read("forecast"), frame=_FRAME),
    "events":    Markdown(_read("events"),   frame=_FRAME),
    "settings":  Markdown(_read("settings"), frame=_FRAME),
}


# Pre-fill at module load
def _seed_initial(snap) -> None:
    on_live_tick_seed(snap)


def on_live_tick_seed(snap) -> None:
    """Same as on_live_tick but writes to module-level globals (no state object)."""
    global data_status, data_status_color, age_display
    global temp_value, temp_color, humidity_value, humidity_color
    global pressure_value, pressure_color, lux_value, lux_color
    global current_value, current_color, mq2_value, mq2_color, pir_value, pir_color
    global prediction_value_display, prediction_meta
    global events_recent, proofs_recent, recommendations
    global chart_figure, forecast_chart
    global hedera_last_publish, hedera_last_status, hedera_next_publish

    lat = snap.latest or {}
    age = snap.last_age_s
    if age is None:
        data_status, data_status_color, age_display = "no data", theme.PALETTE["alert"], "-"
    elif age < 5:
        data_status, data_status_color, age_display = "online", theme.PALETTE["ok"], f"{age:.1f} s ago"
    elif age < 60:
        data_status, data_status_color, age_display = "stale", theme.PALETTE["warn"], f"{age:.0f} s ago"
    else:
        data_status, data_status_color, age_display = "offline", theme.PALETTE["alert"], f"{age/60:.0f} min ago"

    t = _safe_float(lat.get("temp_c"))
    temp_value = _fmt(t, ".1f", " C")
    temp_color = theme.status_color(_temp_severity(t))
    h = _safe_float(lat.get("humidity_pct"))
    humidity_value = _fmt(h, ".0f", " %") if h is not None else "- BMP280"
    humidity_color = theme.status_color(_humidity_severity(h))
    pressure_value = _fmt(_safe_float(lat.get("pressure_hpa")), ".0f", " hPa")
    pressure_color = theme.PALETTE["info"]
    lx = _safe_float(lat.get("lux"))
    lux_value = _fmt(lx, ".0f", " lx")
    lux_color = theme.status_color(_lux_severity(lx))
    cm = _safe_float(lat.get("current_ma"))
    current_value = _fmt(cm, ".0f", " mA")
    current_color = theme.status_color(_current_severity(cm))
    mq = _safe_float(lat.get("mq2_raw"))
    mq2_value = _fmt(mq, ".3f")
    mq2_color = theme.status_color(_mq2_severity(mq))
    pir_value = "motion" if lat.get("pir_state") else "quiet"
    pir_color = theme.PALETTE["warn"] if lat.get("pir_state") else theme.PALETTE["ok"]

    pred = snap.last_prediction
    if pred and pred.get("display_value") is not None:
        val = pred.get("display_value")
        horizon = pred.get("display_horizon_min") or "?"
        prediction_value_display = f"{val:.2f} C"
        prediction_meta = f"in {horizon} min"

    events_recent = snap.events_recent
    proofs_recent = snap.proofs_recent
    recommendations = snap.recommendations
    forecast_chart = _build_forecast_chart(pred)
    global chart_temp, chart_humidity, chart_pressure, chart_lux, chart_current, chart_mq2
    charts = _build_all_metric_charts(60)
    chart_temp     = charts["temp"]
    chart_humidity = charts["humidity"]
    chart_pressure = charts["pressure"]
    chart_lux      = charts["lux"]
    chart_current  = charts["current"]
    chart_mq2      = charts["mq2"]


_seed_initial(data_access.build_live_snapshot())


# Slow refresh loop: tables, charts, forecast - every 30 s.
# Charts are built PER-STATE in _broadcast_slow so the user's selected range
# (quick or custom) is preserved across auto-refresh ticks.
def _slow_refresh_loop():
    while True:
        time.sleep(30)
        try:
            snap = data_access.build_live_snapshot()
            gui.broadcast_callback(
                _broadcast_slow,
                [snap],
                module_context=__name__,
            )
        except Exception:
            logger.exception("slow refresh loop error")


def _broadcast_slow(state, snap) -> None:
    on_slow_tick(state, snap)
    # Rebuild charts using THIS client's current range (prevents auto-refresh
    # from resetting quick-range / custom-range button selection).
    try:
        charts = _build_charts_from_state(state)
        state.chart_temp     = charts["temp"]
        state.chart_humidity = charts["humidity"]
        state.chart_pressure = charts["pressure"]
        state.chart_lux      = charts["lux"]
        state.chart_current  = charts["current"]
        state.chart_mq2      = charts["mq2"]
    except Exception:
        logger.exception("per-state chart rebuild failed")
    try:
        state.forecast_chart = _build_forecast_chart(snap.last_prediction)
    except Exception:
        logger.exception("forecast chart rebuild failed")


_STYLEKIT = {
    "color_primary": "#58a6ff",
    "color_secondary": "#3fb950",
    "color_background_dark": theme.PALETTE["bg_primary"],
    "color_paper_dark": theme.PALETTE["bg_card"],
    "color_background_light": "#f6f8fa",
    "color_paper_light": "#ffffff",
    "color_success": theme.PALETTE["ok"],
    "color_warning": theme.PALETTE["warn"],
    "color_error": theme.PALETTE["alert"],
    "border_radius": 10,
    "input_button_height": "42px",
    "font_family": "'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif",
}

gui = Gui(pages=pages)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    refresh.start_refresh(gui, period_s=float(_env.taipy.live_refresh_s))
    threading.Thread(target=_slow_refresh_loop, daemon=True, name="slow-refresh").start()
    gui.run(
        host=_env.taipy.host,
        port=_env.taipy.port,
        dark_mode=bool(_env.taipy.dark_mode),
        use_reloader=False,
        run_browser=False,
        title="PLC Digital Twin",
        stylekit=_STYLEKIT,
        margin="0",
    )


if __name__ == "__main__":
    main()
