"""Linear regression baseline (sklearn Ridge wrapped in MultiOutputRegressor).

EXTERNAL MACHINE ONLY.

Features come from ``src.ai.features.build_features`` so the linear baseline
shares the same input space as XGBoost / LightGBM - fair comparison.
Multi-output via ``sklearn.multioutput.MultiOutputRegressor(Ridge(alpha=1.0))``
which fits one Ridge per horizon under the hood.

Artifact format: ``joblib.dump({h: Ridge_h}, "<id>.joblib")`` - dict keyed by
horizon-minute. Note: stored as a dict (not the MultiOutputRegressor wrapper
itself) for storage-format parity with the other multi-horizon trainers.
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
    artifact_paths,
    build_target_lags,
    ensure_dirs,
    ffill_numeric,
    load_dataset,
    parse_horizons,
    per_horizon_metrics,
    time_split,
    validate_artifact_inline,
    write_meta_json,
)

logger = logging.getLogger(__name__)


def _select_feature_cols(features_df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [
        c for c in features_df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(features_df[c])
    ]


def train(
    input_path: Path,
    target: str,
    horizons_min: list[int],
    models_dir: Path,
    model_id: str,
    test_size: float,
    data_source: str,
    alpha: float,
) -> None:
    from sklearn.linear_model import Ridge

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
    if not feature_cols:
        raise RuntimeError("no numeric feature columns left after exclusion")

    train_df, test_df = time_split(features_df, test_size=test_size)
    X_train = train_df[feature_cols].to_numpy()
    X_test = test_df[feature_cols].to_numpy()

    models: dict[int, Ridge] = {}
    y_pred_cols: list[np.ndarray] = []
    y_true_cols: list[np.ndarray] = []

    for h, tcol in zip(horizons_min, target_cols, strict=True):
        y_train = train_df[tcol].to_numpy()
        y_test = test_df[tcol].to_numpy()
        m = Ridge(alpha=alpha, random_state=42)
        m.fit(X_train, y_train)
        models[h] = m
        y_pred_cols.append(m.predict(X_test))
        y_true_cols.append(y_test)

    y_pred = np.column_stack(y_pred_cols)
    y_true = np.column_stack(y_true_cols)
    metrics = per_horizon_metrics(y_true, y_pred, horizons_min)
    duration = time.perf_counter() - t0

    logger.info(
        "linear trained in %.2fs; rmse@15=%.4f rmse@60=%.4f",
        duration,
        metrics["rmse_per_horizon"].get("15", float("nan")),
        metrics["rmse_per_horizon"].get("60", float("nan")),
    )

    ensure_dirs()
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib_path, meta_path = artifact_paths(models_dir, model_id)
    joblib.dump(models, joblib_path, compress=3)

    import sklearn

    write_meta_json(
        meta_path,
        model_id=model_id,
        algo="linear",
        target=target,
        horizons_min=horizons_min,
        feature_columns=feature_cols,
        metrics_per_horizon=metrics,
        train_rows=len(train_df),
        test_rows=len(test_df),
        data_source=data_source,
        device_used="cpu",
        framework_version=f"sklearn=={sklearn.__version__} (Ridge alpha={alpha})",
        training_duration_seconds=duration,
        dataset_rows=provenance["rows_after_dropna"],
        provenance=provenance,
        test_size=test_size,
        notes=f"Ridge regression baseline (alpha={alpha})",
    )

    ok, errors, warnings = validate_artifact_inline(joblib_path, meta_path)
    if not ok:
        logger.error("validator errors: %s", errors)
        sys.exit(2)
    if warnings:
        logger.warning("validator warnings: %s", warnings)
    print(f"OK linear {model_id}: train={len(train_df)} test={len(test_df)} duration={duration:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear (Ridge) trainer (multi-horizon)")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target", default="temp_c")
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS_MIN))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--data-source", choices=["real", "synthetic", "hybrid"], default="real")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularisation strength")
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
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
