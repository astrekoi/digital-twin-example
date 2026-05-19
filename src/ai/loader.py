"""ML artifact loader: joblib / keras + meta.json with version checks."""
from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Any

import joblib

from .registry import get_latest_model, list_available_models

logger = logging.getLogger(__name__)

# Anchor torch's C/CUDA-stub initialisation at module-import time, BEFORE any
# other ML framework (tensorflow, darts, etc.) gets a chance to load. On Pi 5
# aarch64 the cu130-suffixed torch wheel ships CUDA stub libraries that, if
# first-touched lazily later - particularly after tensorflow has been imported
# during _check_version("tensorflow", ...) - can SIGABRT with
# "free(): invalid pointer" on subsequent torch.nn forward passes. Eager init
# here has been verified to keep all 7 algo families running in a single
# process. Wrapped in try/except so a Pi without torch installed (e.g. running
# tree-models only) imports this module cleanly.
try:
    import torch as _torch_anchor  # noqa: F401

    _torch_anchor.zeros(1, 1)
except Exception:
    # torch not present, or CPU-only stub init failed for some other reason -
    # downstream lstm/nbeats branches will fail with a clear ImportError when
    # they try to lazy-import torch themselves. The tree/prophet/persistence
    # paths do not need torch and continue to work.
    pass


class ModelNotFoundError(Exception):
    """An explicit model_id was given but is not present in models_dir."""


class ModelLoadError(Exception):
    """Loader failed (joblib or keras)."""


def _check_version(
    label: str,
    meta_version: str | None,
    installed_version: str | None,
) -> None:
    """Warning on version mismatch; does not raise."""
    if meta_version is None or installed_version is None:
        return
    if meta_version != installed_version:
        logger.warning(
            "Version %s: model trained on %s, installed %s - "
            "predictions may differ",
            label,
            meta_version,
            installed_version,
        )


def _installed_sklearn_version() -> str | None:
    try:
        import sklearn
        return sklearn.__version__
    except ImportError:
        return None


def _installed_xgboost_version() -> str | None:
    try:
        import xgboost
        return xgboost.__version__
    except ImportError:
        return None


def _installed_tensorflow_version() -> str | None:
    try:
        import tensorflow as tf
        return tf.__version__
    except ImportError:
        return None


def _installed_lightgbm_version() -> str | None:
    try:
        import lightgbm
        return lightgbm.__version__
    except ImportError:
        return None


def _installed_torch_version() -> str | None:
    try:
        import torch
        # Eagerly touch a no-op CPU tensor so torch's C/CUDA-stub initialisation
        # runs in a controlled context. On aarch64 Pi 5 the cu130-wheel ships
        # CUDA stub libraries that, if first-touched lazily later from a darts
        # NBEATSModel build path, can SIGABRT with "free(): invalid pointer".
        # Anchoring the init here (before any sequence-model code is reached)
        # has been verified to eliminate the crash.
        torch.zeros(1, 1)
        return torch.__version__
    except ImportError:
        return None


def _installed_prophet_version() -> str | None:
    try:
        import prophet
        return prophet.__version__
    except ImportError:
        return None


def _installed_darts_version() -> str | None:
    try:
        import darts
        return darts.__version__
    except ImportError:
        return None


def _load_keras(model_path: Path) -> Any:
    """Load a Keras 3 / TF artifact. Lazy import so dashboard does not pull TF."""
    try:
        import tensorflow as tf  # noqa: F401  (used to confirm runtime is present)
        from keras.models import load_model as keras_load_model
    except ImportError as exc:
        raise ModelLoadError(
            f"TensorFlow not installed; cannot load {model_path.name}. "
            "Run: pip install tensorflow"
        ) from exc
    try:
        return keras_load_model(str(model_path))
    except Exception as exc:
        raise ModelLoadError(f"keras.load_model failed for {model_path}: {exc}") from exc


def _load_joblib(model_path: Path) -> Any:
    try:
        return joblib.load(model_path)
    except Exception as exc:
        raise ModelLoadError(f"Failed to load {model_path}: {exc}") from exc


def load_model(
    model_id: str | None = None,
    models_dir: Path | str = "models",
) -> tuple[Any, dict] | tuple[None, None]:
    """Load a model and its meta.json.

    Parameters
    ----------
    model_id : str | None
        If None - loads the latest model by trained_at/mtime.
        If given - looks up a model with this id; if not found -> ModelNotFoundError.

    Returns (model, meta) or (None, None) when no models exist.
    Supported formats: .joblib (xgboost / sklearn / prophet), .keras (LSTM via TF),
    .pkl (legacy). Loads only locally trusted artifacts in models_dir; readiness
    validation must NOT call this function without explicit runtime permission.
    """
    models_dir = Path(models_dir)
    entry: dict | None = None

    if model_id is None:
        entry = get_latest_model(models_dir)
        if entry is None:
            logger.warning("load_model: no models found in %s", models_dir)
            return None, None
    else:
        entries = list_available_models(models_dir)
        matches = [e for e in entries if e["model_id"] == model_id]
        if not matches:
            raise ModelNotFoundError(
                f"Model '{model_id}' not found in {models_dir}. "
                f"Available: {[e['model_id'] for e in entries]}"
            )
        entry = matches[0]

    meta = entry["meta"]
    _check_version("sklearn", meta.get("sklearn_version"), _installed_sklearn_version())
    _check_version("xgboost", meta.get("xgboost_version"), _installed_xgboost_version())
    _check_version("tensorflow", meta.get("tensorflow_version"), _installed_tensorflow_version())
    _check_version("lightgbm", meta.get("lightgbm_version"), _installed_lightgbm_version())
    _check_version("torch", meta.get("torch_version"), _installed_torch_version())
    _check_version("prophet", meta.get("prophet_version"), _installed_prophet_version())
    _check_version("darts", meta.get("darts_version"), _installed_darts_version())
    _check_version(
        "python",
        meta.get("python_version"),
        platform.python_version(),
    )

    # Persistence bypass: the trainer's joblib payload is a `dict[int,
    # LastValuePredictor]` where `LastValuePredictor` is defined locally inside
    # `training/train_persistence.py`. Pickling captures the class qualname as
    # `__main__.LastValuePredictor` (or `train_persistence.LastValuePredictor`
    # depending on how the trainer was invoked), neither of which is importable
    # on the Pi. The predictor's persistence branch reproduces the
    # `predict = last_observed_target` semantics directly from meta + the
    # latest reading, so the joblib payload is never needed at inference time.
    if (meta.get("algo") or "").lower() == "persistence":
        logger.info(
            "load_model: skipping joblib payload for persistence artifact %s "
            "(inference bypasses payload, uses last observed target)",
            entry["model_id"],
        )
        return None, meta

    model_path: Path = entry["model_path"]
    suffix = model_path.suffix
    if suffix == ".keras":
        model = _load_keras(model_path)
    elif suffix == ".pkl":
        logger.warning("Loading legacy .pkl artifact %s; prefer .joblib", model_path.name)
        model = _load_joblib(model_path)
    elif suffix == ".joblib":
        model = _load_joblib(model_path)
    else:
        raise ModelLoadError(f"Unsupported artifact extension: {suffix} ({model_path.name})")

    logger.info(
        "Model loaded: %s (algo=%s, target=%s, train_rows=%s)",
        entry["model_id"],
        meta.get("algo", "?"),
        meta.get("target", "?"),
        meta.get("train_rows", "?"),
    )
    return model, meta
