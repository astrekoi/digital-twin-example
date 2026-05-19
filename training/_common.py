"""Shared helpers for the multi-horizon training scripts.

All training scripts (persistence, linear, xgboost, lightgbm, prophet, lstm, nbeats)
go through this module for dataset loading, time-based split, multi-horizon target
construction, validator invocation, meta.json writing, and device picking.

Heavy ML frameworks (torch, xgboost, lightgbm) are lazy-imported inside
device-picker functions only, so simply importing this module from a tree-based
script does not pay the torch import cost.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS_MIN: list[int] = [1, 5, 10, 15, 30, 60]
DEFAULT_FREQ_HZ: int = 1
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR: Path = PROJECT_ROOT / "models"
DEFAULT_LOGS_DIR: Path = PROJECT_ROOT / "data" / "training_logs"

# Raw numeric columns used as features for sequence/raw models (LSTM/N-BEATS).
RAW_FEATURE_COLS: list[str] = [
    "temp_c",
    "humidity_pct",
    "pressure_hpa",
    "lux",
    "current_ma",
    "mq2_raw",
]


# --------------------------- dataset loading ---------------------------

def load_dataset(
    csv_path: Path,
    *,
    target: str,
    dropna_target: bool = True,
    dropna_threshold: int = 30_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load CSV, sort by ts_unix, drop or impute rows missing target.

    Returns ``(clean_df, provenance)`` where ``provenance`` records the row
    counts before and after the NaN policy decision plus the strategy applied.
    The provenance dict is meant to be embedded into ``meta.notes`` so that a
    dissertation reviewer can defend exactly how much real data was used.
    """
    df = pd.read_csv(csv_path)
    rows_total = len(df)
    if "ts_unix" in df.columns:
        df = df.sort_values("ts_unix").reset_index(drop=True)
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in {csv_path}")
    target_nan = int(df[target].isna().sum())

    strategy = "none"
    if dropna_target and target_nan > 0:
        kept = rows_total - target_nan
        if kept >= dropna_threshold:
            df = df[df[target].notna()].reset_index(drop=True)
            strategy = "drop"
        else:
            median_val = float(df[target].median(skipna=True))
            df[target] = df[target].fillna(median_val)
            strategy = f"median_impute(value={median_val:.4f}, threshold={dropna_threshold})"
            logger.warning(
                "target=%s post-drop count %d below threshold %d -> median impute (%.4f)",
                target, kept, dropna_threshold, median_val,
            )

    rows_after = len(df)
    provenance = {
        "csv_path": str(csv_path),
        "rows_total": int(rows_total),
        "rows_after_dropna": int(rows_after),
        "rows_dropped": int(rows_total - rows_after),
        "target_nan_count": target_nan,
        "dropna_strategy": strategy,
        "dropna_threshold": int(dropna_threshold),
    }
    logger.info("dataset provenance: %s", provenance)
    return df, provenance


# --------------------------- time-based split ---------------------------

def time_split(df: pd.DataFrame, *, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0,1), got {test_size!r}")
    split_idx = int(len(df) * (1.0 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def three_way_time_split(
    df: pd.DataFrame, *, val_size: float, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-based 3-way split: first (1 - val - test) train, then val, then test.

    Used by trainers that hyperparameter-tune on val while holding test out
    for final reporting (avoids leakage through tuning).
    """
    n = len(df)
    if not 0.0 < val_size < 1.0 or not 0.0 < test_size < 1.0 or val_size + test_size >= 1.0:
        raise ValueError(f"invalid val_size={val_size} test_size={test_size}")
    test_idx = int(n * (1 - test_size))
    val_idx = int(n * (1 - test_size - val_size))
    return (
        df.iloc[:val_idx].copy(),
        df.iloc[val_idx:test_idx].copy(),
        df.iloc[test_idx:].copy(),
    )


# --------------------------- multi-horizon targets ---------------------------

def build_target_lags(
    df: pd.DataFrame,
    *,
    target: str,
    horizons_min: list[int],
    freq_hz: int = DEFAULT_FREQ_HZ,
) -> tuple[pd.DataFrame, list[str]]:
    """Append future-target columns and drop trailing rows that lack any of them.

    Returns ``(df_with_targets, list_of_target_column_names)``.
    Column naming: ``f"{target}_h{h}min"``.
    """
    out = df.copy()
    target_cols: list[str] = []
    for h in horizons_min:
        shift_rows = h * 60 * freq_hz
        col = f"{target}_h{h}min"
        out[col] = out[target].shift(-shift_rows)
        target_cols.append(col)
    out = out.dropna(subset=target_cols).reset_index(drop=True)
    return out, target_cols


# --------------------------- device picking ---------------------------

def device_pick_torch() -> str:
    try:
        import torch  # type: ignore[import-not-found]
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def device_pick_xgb() -> str:
    """Return 'cuda' iff a tiny XGBRegressor fits on cuda, else 'cpu'."""
    try:
        import xgboost as xgb  # type: ignore[import-not-found]
        X = np.random.rand(50, 3).astype(np.float32)
        y = np.random.rand(50).astype(np.float32)
        xgb.XGBRegressor(
            n_estimators=2, device="cuda", tree_method="hist", verbosity=0
        ).fit(X, y)
        return "cuda"
    except Exception as e:
        logger.info("xgb cuda smoke failed (%s) -> cpu", e)
        return "cpu"


def device_pick_lgb() -> str:
    try:
        import lightgbm as lgb  # type: ignore[import-not-found]
        X = np.random.rand(50, 3).astype(np.float32)
        y = np.random.rand(50).astype(np.float32)
        lgb.LGBMRegressor(n_estimators=2, device_type="gpu", verbose=-1).fit(X, y)
        return "gpu"
    except Exception as e:
        logger.info("lightgbm gpu smoke failed (%s) -> cpu", e)
        return "cpu"


# --------------------------- metrics ---------------------------

def per_horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons_min: list[int],
) -> dict[str, dict[str, float]]:
    """Compute RMSE/MAE/R² per horizon.

    ``y_true`` and ``y_pred`` must be shape ``(n_samples, len(horizons_min))``.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: true {y_true.shape} vs pred {y_pred.shape}")
    if y_true.shape[1] != len(horizons_min):
        raise ValueError(
            f"horizon dim mismatch: array has {y_true.shape[1]}, horizons_min has {len(horizons_min)}"
        )

    rmse: dict[str, float] = {}
    mae: dict[str, float] = {}
    r2: dict[str, float] = {}
    for i, h in enumerate(horizons_min):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mse = float(mean_squared_error(yt, yp))
        rmse[str(h)] = float(np.sqrt(mse))
        mae[str(h)] = float(mean_absolute_error(yt, yp))
        r2[str(h)] = float(r2_score(yt, yp))
    return {"rmse_per_horizon": rmse, "mae_per_horizon": mae, "r2_per_horizon": r2}


# --------------------------- meta.json writer ---------------------------

def write_meta_json(
    path: Path,
    *,
    model_id: str,
    algo: str,
    target: str,
    horizons_min: list[int],
    feature_columns: list[str],
    metrics_per_horizon: dict[str, dict[str, float]],
    train_rows: int,
    test_rows: int,
    data_source: str,
    device_used: str,
    framework_version: str,
    training_duration_seconds: float,
    dataset_rows: int,
    provenance: dict[str, Any] | None = None,
    test_size: float = 0.2,
    notes: str | None = None,
    extra: dict[str, Any] | None = None,
    training_extra: dict[str, Any] | None = None,
) -> None:
    """Persist the artifact's meta.json (compatible with src/ai/{loader,registry,artifacts})."""
    versions = _collect_versions()
    notes_combined = (notes or "").strip()
    if provenance:
        prov_str = (
            f"dataset_provenance: rows_total={provenance['rows_total']}, "
            f"rows_after_dropna={provenance['rows_after_dropna']}, "
            f"rows_dropped={provenance['rows_dropped']}, "
            f"target_nan_count={provenance['target_nan_count']}, "
            f"strategy={provenance['dropna_strategy']}, "
            f"threshold={provenance['dropna_threshold']}"
        )
        notes_combined = (notes_combined + "\n" + prov_str).strip() if notes_combined else prov_str

    payload: dict[str, Any] = {
        "model_id": model_id,
        "algo": algo,
        "target": target,
        "horizons_min": list(horizons_min),
        "features": list(feature_columns),
        "feature_columns": list(feature_columns),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "metrics": metrics_per_horizon,
        "training": {
            "device_used": device_used,
            "framework_version": framework_version,
            "training_duration_seconds": float(training_duration_seconds),
            "data_source": data_source,
            "dataset_rows": int(dataset_rows),
            "test_size": float(test_size),
            "split_type": "time-based",
            "provenance": provenance or {},
            **(training_extra or {}),
        },
        "notes": notes_combined,
        **versions,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def _collect_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python_version": platform.python_version()}
    for mod_name, key in [
        ("pandas", "pandas_version"),
        ("numpy", "numpy_version"),
        ("sklearn", "sklearn_version"),
        ("xgboost", "xgboost_version"),
        ("lightgbm", "lightgbm_version"),
        ("torch", "torch_version"),
        ("prophet", "prophet_version"),
        ("darts", "darts_version"),
    ]:
        try:
            mod = __import__(mod_name)
            out[key] = getattr(mod, "__version__", None)
        except Exception:
            out[key] = None
    return out


# --------------------------- validator ---------------------------

def validate_artifact_inline(
    model_path: Path, meta_path: Path
) -> tuple[bool, list[str], list[str]]:
    """Run scripts/validate_model_artifact.py via subprocess; return (ok, errors, warnings)."""
    script = PROJECT_ROOT / "scripts" / "validate_model_artifact.py"
    cmd = [
        sys.executable,
        str(script),
        "--model",
        str(model_path),
        "--meta",
        str(meta_path),
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    try:
        results = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return False, [f"validator stdout not JSON: {proc.stdout!r} stderr={proc.stderr!r}"], []
    result = results[0] if isinstance(results, list) and results else (
        results if isinstance(results, dict) else {}
    )
    errors = list(result.get("errors", []))
    warnings = list(result.get("warnings", []))
    status_ok = result.get("status") == "ok" and not errors
    return status_ok, errors, warnings


# --------------------------- helpers ---------------------------

def artifact_paths(models_dir: Path, model_id: str) -> tuple[Path, Path]:
    return models_dir / f"{model_id}.joblib", models_dir / f"{model_id}.meta.json"


def parse_horizons(spec: str) -> list[int]:
    """Parse "1,5,10,15,30,60" into [1,5,10,15,30,60]; sorts and de-duplicates."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return sorted({int(p) for p in parts})


def ensure_dirs() -> None:
    DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def run_optuna_search(
    model_factory,
    search_space: dict[str, list],
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    seed: int = 42,
    trials_csv_path: Path | None = None,
) -> tuple[dict[str, Any], float, "pd.DataFrame", dict[str, list]]:
    """TPE-Optuna search minimising RMSE on val. Returns
    (best_params, best_rmse, trials_df, search_space_actual).

    `search_space`: ``{param_name: [candidate_values]}``; every entry is
    suggested as ``trial.suggest_categorical``. ``model_factory(params)``
    returns a fresh sklearn-compatible regressor for each trial.

    The search NEVER touches the test partition - both X_val/y_val and
    X_train/y_train must be from val/train respectively. The caller is
    responsible for the (train, val, test) split.
    """
    import optuna
    from sklearn.metrics import mean_squared_error

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):  # type: ignore[no-untyped-def]
        params = {
            name: trial.suggest_categorical(name, candidates)
            for name, candidates in search_space.items()
        }
        model = model_factory(params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, pred)))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_df = study.trials_dataframe()
    if trials_csv_path is not None:
        trials_csv_path.parent.mkdir(parents=True, exist_ok=True)
        trials_df.to_csv(trials_csv_path, index=False)
    return dict(study.best_params), float(study.best_value), trials_df, dict(search_space)


def ffill_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Forward+back fill NaN in numeric columns. Used by tree-based trainers
    BEFORE build_features so that humidity_pct NaN does not cascade through
    lag/rolling/delta features and decimate the training set.

    Operates on a copy; does not mutate the caller's DataFrame.
    """
    out = df.copy()
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    if numeric_cols:
        out[numeric_cols] = out[numeric_cols].ffill().bfill()
    return out
