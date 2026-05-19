"""Naive persistence baseline - predict-equals-last-value.

EXTERNAL MACHINE ONLY.

For each horizon h, the prediction is simply the most recent observed value of
the target. This is the canonical thesis baseline that any "real" model must
outperform for that horizon to be considered useful.

Artifact format: ``joblib.dump({h: LastValuePredictor()}, "<id>.joblib")``.
The ``LastValuePredictor`` exposes ``.predict(X) -> ndarray`` so a future Pi-side
predictor extension can iterate the dict and call each entry uniformly.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Project bootstrap - works from any cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np

from training._common import (  # noqa: E402
    DEFAULT_HORIZONS_MIN,
    artifact_paths,
    build_target_lags,
    ensure_dirs,
    load_dataset,
    parse_horizons,
    per_horizon_metrics,
    time_split,
    validate_artifact_inline,
    write_meta_json,
)

logger = logging.getLogger(__name__)


class LastValuePredictor:
    """Returns the value at column ``target_idx`` of the input as the forecast.

    Operational persistence: ŷ(t+h) = y(t). ``target_idx`` selects which column
    of the feature matrix holds the current target observation. By convention
    feature_columns=[target] so target_idx=0.

    Compatible with the ``model.predict(X) -> ndarray`` sklearn interface so a
    future Pi predictor can dispatch dict-of-models uniformly.
    """

    def __init__(self, target_idx: int = 0) -> None:
        self.target_idx = int(target_idx)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LastValuePredictor":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            return np.array([X_arr[self.target_idx]], dtype=np.float64)
        return X_arr[:, self.target_idx].astype(np.float64, copy=False)


def train(
    input_path: Path,
    target: str,
    horizons_min: list[int],
    models_dir: Path,
    model_id: str,
    test_size: float,
    data_source: str,
) -> None:
    t0 = time.perf_counter()
    df, provenance = load_dataset(input_path, target=target)

    # Multi-horizon target columns (drop trailing NaN rows from the shifts).
    df, target_cols = build_target_lags(df, target=target, horizons_min=horizons_min)
    train_df, test_df = time_split(df, test_size=test_size)

    # Feature columns for persistence are irrelevant (model ignores X), but we
    # report the target column as the single "feature" used so meta.features is
    # never empty (validator requires non-empty list).
    feature_columns = [target]

    # Fit one LastValuePredictor per horizon. Persistence is the same predictor
    # for all horizons (always echo current target observation), but we keep
    # dict-shape parity with the other multi-horizon scripts so the orchestrator
    # / loader can treat all artifacts uniformly.
    models: dict[int, LastValuePredictor] = {}
    X_test = test_df[feature_columns].to_numpy()
    y_pred_cols: list[np.ndarray] = []
    y_true_cols: list[np.ndarray] = []

    for h, tcol in zip(horizons_min, target_cols, strict=True):
        m = LastValuePredictor(target_idx=0)
        m.fit(train_df[feature_columns].to_numpy(), train_df[tcol].to_numpy())
        models[h] = m
        y_pred_cols.append(m.predict(X_test))
        y_true_cols.append(test_df[tcol].to_numpy())

    y_pred = np.column_stack(y_pred_cols)
    y_true = np.column_stack(y_true_cols)
    metrics = per_horizon_metrics(y_true, y_pred, horizons_min)

    duration = time.perf_counter() - t0
    logger.info(
        "persistence trained in %.2fs; rmse@15=%.4f mae@15=%.4f",
        duration,
        metrics["rmse_per_horizon"].get("15", float("nan")),
        metrics["mae_per_horizon"].get("15", float("nan")),
    )

    ensure_dirs()
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib_path, meta_path = artifact_paths(models_dir, model_id)
    joblib.dump(models, joblib_path, compress=3)

    write_meta_json(
        meta_path,
        model_id=model_id,
        algo="persistence",
        target=target,
        horizons_min=horizons_min,
        feature_columns=feature_columns,
        metrics_per_horizon=metrics,
        train_rows=len(train_df),
        test_rows=len(test_df),
        data_source=data_source,
        device_used="cpu",
        framework_version="numpy",
        training_duration_seconds=duration,
        dataset_rows=provenance["rows_after_dropna"],
        provenance=provenance,
        test_size=test_size,
        notes="naive baseline: predict = last observed value",
    )

    ok, errors, warnings = validate_artifact_inline(joblib_path, meta_path)
    if not ok:
        logger.error("validator errors: %s", errors)
        sys.exit(2)
    if warnings:
        logger.warning("validator warnings: %s", warnings)
    print(f"OK persistence {model_id}: train={len(train_df)} test={len(test_df)} duration={duration:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistence baseline trainer (multi-horizon)")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target", default="temp_c")
    parser.add_argument(
        "--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS_MIN),
        help="Comma-separated horizons in minutes",
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--data-source", choices=["real", "synthetic", "hybrid"], default="real")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    train(
        input_path=args.input,
        target=args.target,
        horizons_min=parse_horizons(args.horizons),
        models_dir=args.models_dir,
        model_id=args.model_id,
        test_size=args.test_size,
        data_source=args.data_source,
    )


if __name__ == "__main__":
    main()
