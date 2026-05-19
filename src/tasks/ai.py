"""AI tasks: predict (multi-horizon, dispatched per algo) + dynamic model switching.

Multi-horizon flow:
- Each artifact's ``meta["horizons_min"]`` is the source of truth for which
  horizons the model produces.
- ``predictor.predict_multi(features_df)`` returns ``[{horizon_min, value}, ...]``
  in the order declared by meta.
- Legacy single-output artifacts (no ``horizons_min`` in meta) fall through to
  ``predictor.predict(features_df, horizon_min=15)`` which produces a single point.
- The DB column ``predictions.horizon_min`` (scalar) holds a representative
  "primary" horizon (15 if available, else median); the full multi-point timeline
  lives in ``predictions.forecast_json``. UI already consumes the JSON list.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_env_config
from src.env import load_project_env
from src.storage.factory import create_telemetry_store
from src.tasks.celery_app import celery_app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
load_project_env(ENV_PATH)

logger = logging.getLogger(__name__)

_state: dict[str, Any] = {
    "predictor": None,
    "store": None,
    "model_id": None,
}


# ---------------------------------------------------------------------------
# Lifecycle: init / reload predictor
# ---------------------------------------------------------------------------


def _load_predictor(model_id: str | None) -> Any:
    from src.ai.predictor import Predictor
    env = load_env_config(ENV_PATH)
    return Predictor(models_dir=env.ai.models_dir, model_id=model_id)


def init_ai_resources() -> None:
    if _state["predictor"] is not None:
        return
    env = load_env_config(ENV_PATH)
    if not env.ai.enabled:
        logger.info("AI disabled (AI_ENABLED=false)")
        return
    try:
        predictor = _load_predictor(env.ai.model_id)
        _state["predictor"] = predictor
        _state["model_id"] = (
            getattr(predictor, "model_id", None)
            or predictor.meta.get("model_id")
            or env.ai.model_id
        )
        logger.info("ai resources ready: model_id=%s", _state["model_id"])
    except Exception:
        logger.exception("predictor init failed (will retry lazily)")
    if _state["store"] is None:
        try:
            store = create_telemetry_store()
            store.init_schema()
            _state["store"] = store
        except Exception:
            logger.exception("ai-worker storage init failed")


def _reload_predictor(model_id: str) -> bool:
    """Force-reload Predictor with new model_id; returns True on success."""
    try:
        predictor = _load_predictor(model_id)
        _state["predictor"] = predictor
        _state["model_id"] = (
            getattr(predictor, "model_id", None)
            or predictor.meta.get("model_id")
            or model_id
        )
        logger.info("predictor reloaded: model_id=%s", _state["model_id"])
        return True
    except Exception:
        logger.exception("reload predictor failed for %s", model_id)
        return False


def _persist_model_id(model_id: str) -> None:
    """Update MODEL_ID= line in .env so the choice survives stack restart."""
    if not ENV_PATH.exists():
        return
    text = ENV_PATH.read_text()
    pattern = re.compile(r"^MODEL_ID=.*$", re.MULTILINE)
    new_line = f"MODEL_ID={model_id}"
    text = pattern.sub(new_line, text) if pattern.search(text) else (text.rstrip() + f"\n{new_line}\n")
    ENV_PATH.write_text(text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _readings_to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ts_unix" in df.columns:
        df = df.sort_values("ts_unix").reset_index(drop=True)
    return df


def _algo_of(meta: dict) -> str:
    return str(meta.get("algo", "")).lower()


def _pick_primary_horizon(horizons_min: list[int]) -> int:
    """Pick the representative scalar horizon for ``predictions.horizon_min``.

    Prefers 15 (matches existing UI default + historical data) when present;
    otherwise the middle entry of the list.
    """
    if not horizons_min:
        return 15
    if 15 in horizons_min:
        return 15
    return int(horizons_min[len(horizons_min) // 2])


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@celery_app.task(name="src.tasks.ai.predict_task", acks_late=True, max_retries=2)
def predict_task() -> dict:
    """Pull recent readings, run multi-horizon (or legacy single-point) prediction."""
    if _state["predictor"] is None:
        init_ai_resources()
    predictor = _state["predictor"]
    store = _state["store"]
    if predictor is None or store is None:
        return {"status": "disabled"}

    try:
        rows = store.fetch_recent_readings(minutes=180)
    except Exception:
        logger.exception("fetch_recent_readings failed")
        return {"status": "error", "stage": "fetch"}

    df = _readings_to_df(rows)
    if df.empty:
        return {"status": "no_data"}

    try:
        from src.ai.features import build_features, impute_raw_columns
        df_imputed, _imputed_cols = impute_raw_columns(df)
        df_features = build_features(df_imputed)
    except Exception:
        logger.exception("build_features failed")
        return {"status": "error", "stage": "features"}

    algo = _algo_of(predictor.meta)
    horizons_min = predictor.meta.get("horizons_min")
    forecast: list[dict] = []
    primary_horizon: int

    if horizons_min:
        # Multi-horizon path - predictor dispatches by algo internally.
        try:
            forecast = predictor.predict_multi(df_features)
        except Exception:
            logger.exception("predict_multi failed")
            return {"status": "error", "stage": "predict_multi"}
        if not forecast:
            return {"status": "none"}
        primary_horizon = _pick_primary_horizon(list(horizons_min))
    else:
        # Legacy single-output fallback (artifacts without horizons_min in meta).
        primary_horizon = 15
        try:
            result = predictor.predict(df_features, horizon_min=primary_horizon)
        except Exception:
            logger.exception("predictor.predict failed")
            return {"status": "error", "stage": "predict"}
        if result is None:
            return {"status": "none"}
        if isinstance(result, dict):
            value = result.get("forecast")
            primary_horizon = int(result.get("horizon_min", primary_horizon))
        else:
            value = result
        if value is None:
            return {"status": "none"}
        forecast = [{"horizon_min": primary_horizon, "value": float(value)}]

    try:
        store.insert_prediction(
            model_id=_state["model_id"] or "unknown",
            horizon_min=primary_horizon,
            forecast=forecast,
            features=None,
        )
    except Exception:
        logger.exception("insert_prediction failed")
        return {"status": "error", "stage": "store"}

    # Look up the value at the primary horizon for the return summary;
    # fallback to forecast[0] if (somehow) the primary isn't in the list.
    primary_value = next(
        (float(f["value"]) for f in forecast if int(f["horizon_min"]) == primary_horizon),
        float(forecast[0]["value"]),
    )
    return {
        "status": "ok",
        "model_id": _state["model_id"],
        "algo": algo,
        "points": len(forecast),
        "primary_value": primary_value,
        "primary_horizon": primary_horizon,
    }


@celery_app.task(name="src.tasks.ai.set_model_task", acks_late=True)
def set_model_task(model_id: str) -> dict:
    """Switch active model on the ai-worker without restart, persist to .env."""
    if not model_id or not isinstance(model_id, str):
        return {"status": "error", "message": "invalid model_id"}
    ok = _reload_predictor(model_id)
    if not ok:
        return {"status": "error", "model_id": model_id, "message": "reload failed"}
    try:
        _persist_model_id(model_id)
    except Exception:
        logger.exception("persist MODEL_ID to .env failed")
    return {"status": "ok", "model_id": _state["model_id"]}
