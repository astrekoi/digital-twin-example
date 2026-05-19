"""Predictor: load the model once, then call the right backend API.

Backends:
- ``persistence``       - joblib payload bypassed; returns last observed
                          target value × N horizons (multi-horizon only).
- ``linear`` / ``xgboost`` / ``lightgbm`` (multi-horizon) - joblib payload is
                          ``dict[int, regressor]``; iterate per-horizon.
- ``prophet``           - model.make_future_dataframe(...) + per-horizon yhat.
- ``lstm`` (PyTorch)    - joblib payload bundles state_dict + architecture +
                          scaler_X/y; LSTMForecaster rebuilt on first call.
- ``nbeats`` (Darts)    - joblib payload bundles darts_state_dict + arch;
                          NBEATSModel rebuilt + dummy-fit + state_dict load.
- ``lstm`` (Keras, legacy single-output) - falls through legacy predict() path.
- xgboost/sklearn single-output (legacy)  - legacy predict() path.

The predictor never raises on inference errors at the public API level - it
returns None / empty list and logs. Missing features (e.g. humidity_pct on a
BMP280-only stand) are imputed via ``src.ai.features.latest_feature_row_imputed``
and reported back through ``imputed``.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import fill_window_nans, latest_feature_row_imputed
from .loader import load_model

logger = logging.getLogger(__name__)

_DEFAULT_LSTM_SEQ_LEN = 60
_PROPHET_FREQ = "min"

# Algorithms that produce a multi-horizon dict[int, regressor] joblib payload
# and consume a flat tabular feature row at predict time.
_TREE_ALGOS = {"linear", "xgboost", "lightgbm"}


def _algo_of(meta: dict, model_path_suffix: str | None = None) -> str:
    """Detect algo from meta or fall back to file extension hints."""
    algo = (meta.get("algo") or "").strip().lower()
    if algo:
        return algo
    if model_path_suffix == ".keras":
        return "lstm"
    return "sklearn"


class Predictor:
    """ML-model inference on Pi 5.

    Loads the model once at startup. If no model is available, operates in
    no-op mode: ``predict()`` / ``predict_multi()`` return ``None`` / ``[]``
    and log a warning. Any error in a public-API call returns the empty value
    - the stand never crashes.

    For sequence models (PyTorch LSTM, Darts N-BEATS) the rebuilt module is
    cached on the instance so subsequent calls reuse it.
    """

    def __init__(
        self,
        models_dir: Path | str = "models",
        model_id: str | None = None,
    ) -> None:
        self._models_dir = Path(models_dir)
        self._model_id = model_id
        self.model: Any = None
        self.meta: dict = {}
        # Cached rebuilds for sequence models - invalidated on reload().
        self._payload: Any = None
        self._torch_module: Any = None
        self._darts_module: Any = None
        self.reload()

    def reload(self) -> bool:
        """Reload the model from disk. Returns True on success."""
        try:
            model, meta = load_model(
                model_id=self._model_id,
                models_dir=self._models_dir,
            )
            self.model = model
            self.meta = meta or {}
            self._payload = model  # for sequence models this is the dict-of-everything
            self._torch_module = None
            self._darts_module = None
            algo = (self.meta.get("algo") or "").lower()
            if model is None and algo != "persistence":
                # persistence intentionally returns model=None - that's not a
                # "no model available" warning case.
                logger.warning("Predictor: no models available - running in no-op mode")
                return False
            return True
        except Exception:
            logger.exception("Predictor.reload: error loading model")
            self.model = None
            self.meta = {}
            self._payload = None
            self._torch_module = None
            self._darts_module = None
            return False

    # ------------------------------------------------------------------
    # Multi-horizon backends - all return (forecast: list[dict], imputed: list[str])
    # forecast item shape: {"horizon_min": int, "value": float}
    # ------------------------------------------------------------------

    def _predict_persistence_multi(
        self, features_df: pd.DataFrame
    ) -> tuple[list[dict], list[str]]:
        """Persistence: echo last observed target value for every horizon."""
        target = self.meta.get("target")
        horizons = self.meta.get("horizons_min") or []
        latest: float = 0.0
        if target and target in features_df.columns:
            series = features_df[target].dropna()
            if len(series) > 0:
                latest = float(series.iloc[-1])
            else:
                logger.warning(
                    "persistence: target %r is fully NaN in features_df -> using 0.0",
                    target,
                )
        else:
            logger.warning(
                "persistence: target %r missing from features_df columns -> using 0.0",
                target,
            )
        return [{"horizon_min": int(h), "value": latest} for h in horizons], []

    def _predict_tree_multi(
        self, features_df: pd.DataFrame
    ) -> tuple[list[dict], list[str]]:
        """linear / xgboost / lightgbm: ``self.model[h].predict(X)`` per horizon."""
        feat_cols = self.meta.get("feature_columns") or self.meta.get("features")
        if not feat_cols:
            raise ValueError("tree multi: meta.feature_columns / meta.features is empty")
        row, imputed = latest_feature_row_imputed(features_df, feat_cols)
        if row is None:
            logger.warning("tree multi: latest_feature_row_imputed returned None")
            return [], imputed
        X = pd.DataFrame([row])[feat_cols]  # explicit column ordering
        horizons = self.meta.get("horizons_min") or []
        out: list[dict] = []
        warned_singleton = False
        for h in horizons:
            h_int = int(h)
            if isinstance(self.model, dict):
                if h_int not in self.model:
                    logger.warning(
                        "tree multi: horizon %d missing from model dict (keys=%s) - skipping",
                        h_int, sorted(self.model.keys()),
                    )
                    continue
                model_h = self.model[h_int]
            else:
                if not warned_singleton:
                    logger.warning(
                        "Artifact has horizons_min in meta but singleton payload - "
                        "degraded multi-horizon mode (single model echoed for all horizons)"
                    )
                    warned_singleton = True
                model_h = self.model
            value = float(model_h.predict(X)[0])
            out.append({"horizon_min": h_int, "value": value})
        return out, imputed

    def _predict_prophet_multi(
        self, features_df: pd.DataFrame
    ) -> tuple[list[dict], list[str]]:
        """Prophet: make_future_dataframe(periods=max(h)) -> take yhat[h-1]."""
        horizons = self.meta.get("horizons_min") or []
        if not horizons:
            return [], []
        max_h = int(max(horizons))
        future = self.model.make_future_dataframe(
            periods=max_h,
            freq=_PROPHET_FREQ,
            include_history=False,
        )
        fc = self.model.predict(future)
        if fc is None or fc.empty:
            return [], []
        out: list[dict] = []
        for h in horizons:
            try:
                yhat = float(fc.iloc[int(h) - 1]["yhat"])
                out.append({"horizon_min": int(h), "value": yhat})
            except (IndexError, KeyError, TypeError, ValueError):
                logger.warning("prophet: failed to read yhat for horizon %s", h)
                continue
        return out, []

    def _predict_lstm_pytorch_multi(
        self, features_df: pd.DataFrame
    ) -> tuple[list[dict], list[str]]:
        """PyTorch LSTM: rebuild LSTMForecaster + scaler_X/y round-trip + forward."""
        # Lazy imports - keep the predictor module importable on a Pi without torch.
        import torch  # noqa: F401  (used below)

        from .torch_models import LSTMForecaster

        payload = self._payload
        if not isinstance(payload, dict):
            raise ValueError(
                f"lstm pytorch: expected dict payload, got {type(payload).__name__}"
            )
        # STRICT feature column check - payload is the authoritative source.
        feat_cols = payload.get("feature_columns")
        if not feat_cols:
            raise ValueError("lstm pytorch: payload.feature_columns missing")
        missing = [c for c in feat_cols if c not in features_df.columns]
        if missing:
            raise ValueError(f"LSTM features missing on Pi: {missing}")

        arch = payload["architecture_config"]
        seq_len = int(arch.get("sequence_len", _DEFAULT_LSTM_SEQ_LEN))

        if self._torch_module is None:
            module = LSTMForecaster(**arch)
            module.load_state_dict(payload["state_dict"])
            module.eval()
            self._torch_module = module

        if len(features_df) < seq_len:
            logger.warning(
                "lstm pytorch: not enough rows (have=%d, need=%d)",
                len(features_df), seq_len,
            )
            return [], []
        window = features_df.tail(seq_len)[feat_cols]  # explicit column ordering
        window, imputed = fill_window_nans(window, fallback=0.0)

        scaler_X = payload["scaler_X"]
        scaler_y = payload["scaler_y"]
        Xs = scaler_X.transform(window.values).astype(np.float32)
        Xs = Xs.reshape(1, seq_len, len(feat_cols))

        with torch.no_grad():
            out = self._torch_module(torch.from_numpy(Xs)).cpu().numpy()
        vals = scaler_y.inverse_transform(out)[0]

        horizons = payload.get("horizons_min") or self.meta.get("horizons_min") or []
        forecast = [
            {"horizon_min": int(h), "value": float(v)}
            for h, v in zip(horizons, vals, strict=False)
        ]
        return forecast, imputed

    def _predict_nbeats_multi(
        self, features_df: pd.DataFrame
    ) -> tuple[list[dict], list[str]]:
        """Darts N-BEATS: rebuild NBEATSModel, dummy-fit to materialize Lightning
        module, then load trained state_dict and predict."""
        # Lazy imports - keep the predictor module importable on a Pi without darts.
        from darts import TimeSeries
        from darts.models import NBEATSModel

        payload = self._payload
        if not isinstance(payload, dict):
            raise ValueError(
                f"nbeats: expected dict payload, got {type(payload).__name__}"
            )
        feat_cols = payload.get("feature_columns")
        if not feat_cols:
            raise ValueError("nbeats: payload.feature_columns missing")
        missing = [c for c in feat_cols if c not in features_df.columns]
        if missing:
            raise ValueError(f"NBEATS features missing on Pi: {missing}")
        target_col = feat_cols[0]

        arch = payload["architecture_config"]
        input_chunk = int(arch["input_chunk_length"])
        output_chunk = int(arch["output_chunk_length"])

        if self._darts_module is None:
            # Materialize self._darts_module.model (Lightning) via a tiny dummy
            # fit, then overwrite weights from the trained state_dict. The
            # `model.model is None` until first .fit()` contract is undocumented
            # darts internals - pin darts==0.44.x to keep this stable.
            dummy_n = input_chunk + output_chunk + 4
            dummy_values = np.zeros(dummy_n, dtype=np.float64)
            dummy_ts = TimeSeries.from_values(dummy_values)
            module = NBEATSModel(
                **arch,
                n_epochs=1,
                batch_size=2,
                random_state=42,
            )
            module.fit(dummy_ts)
            try:
                module.model.load_state_dict(payload["darts_state_dict"])
            except Exception as exc:
                raise RuntimeError(
                    "nbeats load_state_dict failed (likely darts version mismatch - "
                    f"trainer used darts={payload.get('darts_version', '?')}): {exc}"
                ) from exc
            self._darts_module = module

        if len(features_df) < input_chunk:
            logger.warning(
                "nbeats: not enough rows (have=%d, need=%d)",
                len(features_df), input_chunk,
            )
            return [], []
        ts_values = np.asarray(
            features_df.tail(input_chunk)[target_col].values, dtype=np.float64
        )
        ts = TimeSeries.from_values(ts_values)

        horizons = self.meta.get("horizons_min") or []
        if not horizons:
            return [], []
        max_h = int(max(horizons))
        forecast = self._darts_module.predict(n=max_h, series=ts)
        arr = forecast.values().flatten()
        out: list[dict] = []
        for h in horizons:
            idx = int(h) - 1
            if 0 <= idx < len(arr):
                out.append({"horizon_min": int(h), "value": float(arr[idx])})
        return out, []

    # ------------------------------------------------------------------
    # Legacy single-output backends - used by predict() shim for legacy
    # artifacts (no horizons_min in meta).
    # ------------------------------------------------------------------

    def _predict_sklearn_like(
        self,
        features_df: pd.DataFrame,
        required_features: list[str],
    ) -> tuple[float | None, list[str]]:
        """Sklearn / xgboost path: predict from the last (NaN-imputed) feature row."""
        row, imputed = latest_feature_row_imputed(features_df, required_features)
        if row is None:
            logger.warning(
                "Predictor.predict[xgboost]: missing required columns or empty df",
            )
            return None, []
        X = pd.DataFrame([row])[required_features]
        # Defensive - old single-output artifacts have a single regressor here;
        # if somehow we got a dict (e.g. caller used legacy predict() on a new
        # multi-horizon artifact), use the closest model.
        model = self.model
        if isinstance(model, dict):
            logger.warning(
                "legacy predict(): model is dict - picking first key %s",
                sorted(model.keys())[0] if model else None,
            )
            model = next(iter(model.values()))
        raw_pred = model.predict(X)
        return float(raw_pred[0]), imputed

    def _predict_prophet(
        self,
        features_df: pd.DataFrame,
        horizon_min: int,
    ) -> tuple[float | None, list[str]]:
        """Prophet path: make_future_dataframe(periods=horizon, freq='min')."""
        if "ts_iso" not in features_df.columns or features_df.empty:
            logger.warning("Predictor.predict[prophet]: ts_iso missing or empty df")
            return None, []
        future = self.model.make_future_dataframe(
            periods=int(horizon_min),
            freq=_PROPHET_FREQ,
            include_history=False,
        )
        forecast = self.model.predict(future)
        if forecast is None or forecast.empty:
            logger.warning("Predictor.predict[prophet]: empty forecast")
            return None, []
        return float(forecast.iloc[-1]["yhat"]), []

    def _predict_lstm(
        self,
        features_df: pd.DataFrame,
        required_features: list[str],
    ) -> tuple[float | None, list[str]]:
        """Keras LSTM path (legacy): build (1, seq_len, n_feat) tensor."""
        seq_len = int(self.meta.get("sequence_len") or _DEFAULT_LSTM_SEQ_LEN)
        missing = [c for c in required_features if c not in features_df.columns]
        if missing:
            logger.warning("Predictor.predict[lstm]: missing columns %s", missing)
            return None, []
        if len(features_df) < seq_len:
            logger.warning(
                "Predictor.predict[lstm]: not enough rows (have=%d, need=%d)",
                len(features_df),
                seq_len,
            )
            return None, []
        window = features_df[required_features].iloc[-seq_len:].copy()
        window, imputed = fill_window_nans(window, fallback=0.0)
        x = window.to_numpy(dtype=np.float32).reshape(1, seq_len, len(required_features))
        try:
            raw = self.model.predict(x, verbose=0)
        except TypeError:
            raw = self.model.predict(x)
        return float(np.asarray(raw).reshape(-1)[0]), imputed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_multi(
        self,
        features_df: pd.DataFrame,
        pre_imputed: list[str] | None = None,
    ) -> list[dict]:
        """Multi-horizon predict.

        Returns ``[{"horizon_min": int, "value": float}, ...]`` ordered by the
        horizons listed in ``meta["horizons_min"]``. Returns ``[]`` on any
        predict-time error after logging.
        """
        meta = self.meta
        algo = (meta.get("algo") or "").strip().lower()
        if not algo:
            logger.warning("predict_multi: meta.algo missing")
            return []

        try:
            if algo == "persistence":
                forecast, backend_imputed = self._predict_persistence_multi(features_df)
            elif algo in _TREE_ALGOS:
                forecast, backend_imputed = self._predict_tree_multi(features_df)
            elif algo == "prophet":
                forecast, backend_imputed = self._predict_prophet_multi(features_df)
            elif algo == "lstm":
                framework = (meta.get("framework") or "").strip().lower()
                artifact_type = (meta.get("artifact_type") or "").strip().lower()
                if framework == "pytorch" or artifact_type == "torch":
                    forecast, backend_imputed = self._predict_lstm_pytorch_multi(
                        features_df
                    )
                else:
                    raise ValueError(
                        "lstm multi-horizon requires framework='pytorch' or "
                        "artifact_type='torch' in meta; legacy Keras LSTM is "
                        "single-output only and should go through legacy predict()"
                    )
            elif algo == "nbeats":
                forecast, backend_imputed = self._predict_nbeats_multi(features_df)
            else:
                raise ValueError(f"predict_multi: unsupported algo {algo!r}")
        except Exception:
            logger.exception(
                "predict_multi: %s backend failed for %s",
                algo, meta.get("model_id", "?"),
            )
            return []

        # Merge imputation lists for the log line.
        merged: list[str] = []
        seen: set[str] = set()
        for name in (*(pre_imputed or []), *backend_imputed):
            if name not in seen:
                seen.add(name)
                merged.append(name)
        if merged:
            logger.info(
                "predict_multi[%s]: imputed %d feature(s): %s",
                algo, len(merged), merged,
            )
        return forecast

    def predict(
        self,
        features_df: pd.DataFrame,
        horizon_min: int = 15,
        pre_imputed: list[str] | None = None,
    ) -> dict | None:
        """Legacy single-point predict - kept for ``scripts/predictor_daemon.py``.

        For artifacts with ``meta["horizons_min"]`` this delegates to
        ``predict_multi()`` and returns the point closest to ``horizon_min``.
        For legacy single-output artifacts (no horizons_min in meta) it walks
        the original single-backend path.
        """
        if self.model is None and (self.meta.get("algo") or "").lower() != "persistence":
            logger.warning("Predictor.predict: model is not loaded")
            return None

        # Multi-horizon artifacts: derive from predict_multi().
        if self.meta.get("horizons_min"):
            multi = self.predict_multi(features_df, pre_imputed=pre_imputed)
            if not multi:
                return None
            chosen = min(multi, key=lambda f: abs(int(f["horizon_min"]) - int(horizon_min)))
            return {
                "model_id": self.meta.get("model_id"),
                "horizon_min": int(chosen["horizon_min"]),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "forecast": float(chosen["value"]),
                "features_used": (
                    self.meta.get("feature_columns") or self.meta.get("features")
                ),
                "imputed": [],  # multi path logs imputation internally
                "status": "ok",
            }

        # Legacy single-output path (artifacts without horizons_min).
        required_features: list[str] | None = self.meta.get("features")
        if not required_features:
            logger.warning(
                "Predictor.predict: meta.features is empty for model %s",
                self.meta.get("model_id", "?"),
            )
            return None

        algo = _algo_of(self.meta)
        try:
            if algo == "prophet":
                forecast_value, backend_imputed = self._predict_prophet(
                    features_df, horizon_min
                )
            elif algo in {"lstm", "gru", "rnn", "tcn"}:
                forecast_value, backend_imputed = self._predict_lstm(
                    features_df, required_features
                )
            else:
                forecast_value, backend_imputed = self._predict_sklearn_like(
                    features_df, required_features
                )
        except Exception:
            logger.exception(
                "Predictor.predict: %s backend failed for %s",
                algo,
                self.meta.get("model_id", "?"),
            )
            return None

        if forecast_value is None:
            return None

        merged: list[str] = []
        seen: set[str] = set()
        for name in (*(pre_imputed or []), *backend_imputed):
            if name not in seen:
                seen.add(name)
                merged.append(name)

        if merged:
            logger.info(
                "Predictor.predict[%s]: imputed %d feature(s): %s",
                algo, len(merged), merged,
            )

        return {
            "model_id": self.meta.get("model_id"),
            "horizon_min": horizon_min,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "forecast": forecast_value,
            "features_used": required_features,
            "imputed": merged,
            "status": "ok",
        }
