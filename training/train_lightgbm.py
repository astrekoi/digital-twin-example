"""LightGBM regressor for forecasting telemetry time series - multi-horizon.

EXTERNAL MACHINE ONLY. Same contract as ``train_xgboost.py``.

Stage 2 (2026-05-01) added Optuna hyperparameter search via ``--tune``:
- 70/15/15 train/val/test split (test partition strictly held out).
- Optuna trials minimise RMSE@15 on (train_fit, val_eval).
- Refit on (train+val) with best_params, eval on held-out test.
- ``meta.training.optuna`` records search_space, best_params, best_val_rmse_at_15.

GPU: tries ``device_type="gpu"`` smoke; pip wheels on Windows usually fall back
to CPU.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.ai.features import build_features  # noqa: E402
from training._common import (  # noqa: E402
    DEFAULT_HORIZONS_MIN,
    DEFAULT_LOGS_DIR,
    artifact_paths,
    build_target_lags,
    device_pick_lgb,
    ensure_dirs,
    ffill_numeric,
    load_dataset,
    parse_horizons,
    per_horizon_metrics,
    run_optuna_search,
    three_way_time_split,
    time_split,
    validate_artifact_inline,
    write_meta_json,
)

logger = logging.getLogger(__name__)


LGB_SEARCH_SPACE: dict[str, list] = {
    "n_estimators": [100, 200, 400, 800],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 5.0],
}

LGB_DEFAULTS: dict = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


def _select_feature_cols(features_df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [
        c for c in features_df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(features_df[c])
    ]


def _build_lgb(params: dict, *, device_type: str):
    import lightgbm as lgb

    # subsample_freq=1 is required for `subsample` to actually take effect in
    # LightGBM (bagging happens once per `subsample_freq` iterations).
    return lgb.LGBMRegressor(
        **params,
        random_state=42,
        device_type=device_type,
        subsample_freq=1,
        verbose=-1,
    )


def train(
    input_path: Path,
    target: str,
    horizons_min: list[int],
    models_dir: Path,
    model_id: str,
    val_size: float,
    test_size: float,
    data_source: str,
    tune: bool,
    n_trials: int,
    parent_v1: str | None,
) -> None:
    import lightgbm as lgb

    t0 = time.perf_counter()
    df, provenance = load_dataset(input_path, target=target)
    df = ffill_numeric(df)

    features_df = build_features(df)
    features_df, target_cols = build_target_lags(
        features_df, target=target, horizons_min=horizons_min
    )
    features_df = features_df.dropna().reset_index(drop=True)

    exclude = {"ts_iso", "ts_unix", "id", target, *target_cols}
    feature_cols = _select_feature_cols(features_df, exclude)

    if tune:
        train_df, val_df, test_df = three_way_time_split(
            features_df, val_size=val_size, test_size=test_size
        )
    else:
        train_df, test_df = time_split(features_df, test_size=test_size)
        val_df = train_df.iloc[0:0]

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32) if len(val_df) else None
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)

    target_h15_col = next(c for c, h in zip(target_cols, horizons_min, strict=True) if h == 15)

    device_type = device_pick_lgb()
    logger.info(
        "lightgbm: device_type=%s, %d features, %d horizons, train=%d val=%d test=%d, tune=%s",
        device_type, len(feature_cols), len(horizons_min),
        len(train_df), len(val_df), len(test_df), tune,
    )

    optuna_meta: dict | None = None
    best_params = dict(LGB_DEFAULTS)

    if tune:
        if X_val is None or len(X_val) == 0:
            raise RuntimeError("--tune requires non-empty val partition")
        y_train_h15 = train_df[target_h15_col].to_numpy(dtype=np.float32)
        y_val_h15 = val_df[target_h15_col].to_numpy(dtype=np.float32)

        trials_csv = DEFAULT_LOGS_DIR / f"{model_id}_optuna.csv"
        logger.info("starting Optuna: n_trials=%d, target=RMSE@15 on val", n_trials)
        t_search = time.perf_counter()
        best_params_search, best_val_rmse, trials_df, search_space_actual = run_optuna_search(
            lambda params: _build_lgb(params, device_type=device_type),
            LGB_SEARCH_SPACE,
            X_train=X_train,
            y_train=y_train_h15,
            X_val=X_val,
            y_val=y_val_h15,
            n_trials=n_trials,
            seed=42,
            trials_csv_path=trials_csv,
        )
        search_duration = time.perf_counter() - t_search
        logger.info(
            "Optuna done in %.1fs: best_val_rmse@15=%.4f, best_params=%s",
            search_duration, best_val_rmse, best_params_search,
        )
        best_params = best_params_search
        optuna_meta = {
            "n_trials": n_trials,
            "search_space": search_space_actual,
            "best_params": best_params_search,
            "best_val_rmse_at_15": best_val_rmse,
            "trials_log_path": str(trials_csv.relative_to(_PROJECT_ROOT))
            if trials_csv.is_relative_to(_PROJECT_ROOT) else str(trials_csv),
            "search_duration_seconds": search_duration,
        }

    if tune and len(val_df):
        fit_df = pd.concat([train_df, val_df], ignore_index=True)
    else:
        fit_df = train_df
    X_fit = fit_df[feature_cols].to_numpy(dtype=np.float32)

    models: dict[int, lgb.LGBMRegressor] = {}
    y_pred_cols: list[np.ndarray] = []
    y_true_cols: list[np.ndarray] = []
    for h, tcol in zip(horizons_min, target_cols, strict=True):
        y_fit = fit_df[tcol].to_numpy(dtype=np.float32)
        y_test = test_df[tcol].to_numpy(dtype=np.float32)
        m = _build_lgb(best_params, device_type=device_type)
        m.fit(X_fit, y_fit)
        models[h] = m
        y_pred_cols.append(m.predict(X_test))
        y_true_cols.append(y_test)

    y_pred = np.column_stack(y_pred_cols)
    y_true = np.column_stack(y_true_cols)
    metrics = per_horizon_metrics(y_true, y_pred, horizons_min)
    duration = time.perf_counter() - t0

    logger.info(
        "TEST: lightgbm trained in %.2fs; rmse@15=%.4f rmse@60=%.4f device_type=%s tuned=%s",
        duration,
        metrics["rmse_per_horizon"].get("15", float("nan")),
        metrics["rmse_per_horizon"].get("60", float("nan")),
        device_type, tune,
    )

    ensure_dirs()
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib_path, meta_path = artifact_paths(models_dir, model_id)
    joblib.dump(models, joblib_path, compress=3)

    training_extra: dict = {"tuned": tune}
    if optuna_meta is not None:
        training_extra["optuna"] = optuna_meta
        training_extra["best_hyperparams"] = best_params
        training_extra["val_size"] = val_size
    if parent_v1:
        training_extra["parent_v1"] = parent_v1

    notes_extra = ""
    if tune:
        notes_extra = (
            f" Tuned via Optuna n_trials={n_trials}, search minimised RMSE@15 on val; "
            f"refit on (train+val), eval on held-out test. parent={parent_v1 or 'n/a'}."
        )

    write_meta_json(
        meta_path,
        model_id=model_id,
        algo="lightgbm",
        target=target,
        horizons_min=horizons_min,
        feature_columns=feature_cols,
        metrics_per_horizon=metrics,
        train_rows=len(fit_df),
        test_rows=len(test_df),
        data_source=data_source,
        device_used=device_type,
        framework_version=f"lightgbm=={lgb.__version__}",
        training_duration_seconds=duration,
        dataset_rows=provenance["rows_after_dropna"],
        provenance=provenance,
        test_size=test_size,
        notes=(
            f"LightGBM multi-horizon (one regressor per horizon). "
            f"params={best_params}, device_type={device_type}, subsample_freq=1."
            + notes_extra
        ),
        training_extra=training_extra,
    )

    ok, errors, warnings = validate_artifact_inline(joblib_path, meta_path)
    if not ok:
        logger.error("validator errors: %s", errors)
        sys.exit(2)
    if warnings:
        logger.warning("validator warnings: %s", warnings)
    print(
        f"OK lightgbm {model_id}: train(+val)={len(fit_df)} test={len(test_df)} "
        f"duration={duration:.2f}s device_type={device_type} tuned={tune} "
        f"test_rmse@15={metrics['rmse_per_horizon'].get('15', float('nan')):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM trainer (multi-horizon, Optuna-tunable)")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target", default="temp_c")
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS_MIN))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--data-source", choices=["real", "synthetic", "hybrid"], default="real")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--parent-v1", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    test_size = args.test_size if args.tune else (args.test_size if args.test_size != 0.15 else 0.2)

    train(
        input_path=args.input,
        target=args.target,
        horizons_min=parse_horizons(args.horizons),
        models_dir=args.models_dir,
        model_id=args.model_id,
        val_size=args.val_size,
        test_size=test_size,
        data_source=args.data_source,
        tune=args.tune,
        n_trials=args.n_trials,
        parent_v1=args.parent_v1,
    )


if __name__ == "__main__":
    main()
