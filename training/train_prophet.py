"""Prophet for seasonal forecasting of telemetry time series - multi-horizon.

EXTERNAL MACHINE ONLY. Prophet pulls cmdstanpy and is heavy. Do not install on Pi.

Multi-horizon strategy: a single Prophet model is trained on the train partition.
For each test row at time ``ds_t`` we ask Prophet to predict at ``ds_t + h*60s``
for each horizon h, and compare against the actual value at that future time.

Stage 1 (2026-05-01) introduced 70/15/15 train/val/test split (was 80/20) and
exposed the seasonality / changepoint hyperparameters via CLI:
    --changepoint-prior-scale (default 0.05)
    --seasonality-mode (additive | multiplicative, default additive)
    --daily-seasonality, --weekly-seasonality, --yearly-seasonality (true|false|auto)

Test partition is held out for the final reported metrics; val is used only as
a sanity-check log line (Prophet does not formally hyperparameter-search here).
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

from training._common import (  # noqa: E402
    DEFAULT_HORIZONS_MIN,
    artifact_paths,
    build_target_lags,
    ensure_dirs,
    load_dataset,
    parse_horizons,
    per_horizon_metrics,
    three_way_time_split,
    validate_artifact_inline,
    write_meta_json,
)

logger = logging.getLogger(__name__)


def _build_ds_column(df: pd.DataFrame) -> pd.DataFrame:
    if "ts_iso" in df.columns:
        ds = pd.to_datetime(df["ts_iso"], utc=True, errors="coerce").dt.tz_convert(None)
    elif "ts_unix" in df.columns:
        ds = pd.to_datetime(df["ts_unix"], unit="s", utc=True).dt.tz_convert(None)
    else:
        raise ValueError("CSV must have ts_iso or ts_unix")
    return df.assign(ds=ds)


def _parse_seasonality_flag(value: str) -> bool | str:
    """Prophet accepts True/False/'auto' for seasonality flags. Parse CLI string."""
    v = (value or "").strip().lower()
    if v in {"true", "1", "yes", "on"}:
        return True
    if v in {"false", "0", "no", "off"}:
        return False
    if v in {"auto", ""}:
        return "auto"
    raise ValueError(f"unrecognised seasonality flag {value!r}")


def _predict_for_test_set(model, test_ds: np.ndarray, horizons_min: list[int]) -> np.ndarray:
    """For each ds in test_ds and each horizon h, predict at ds + h minutes.
    Returns shape (n_test, n_horizons)."""
    future_dfs: list[pd.DataFrame] = []
    for h in horizons_min:
        offset = pd.Timedelta(minutes=h)
        future_dfs.append(pd.DataFrame({"ds": test_ds + offset.to_numpy()}))
    future_all = pd.concat(future_dfs, ignore_index=True)
    forecast = model.predict(future_all)
    yhat = forecast["yhat"].to_numpy()
    n = len(test_ds)
    out = np.zeros((n, len(horizons_min)), dtype=np.float64)
    for j in range(len(horizons_min)):
        out[:, j] = yhat[j * n : (j + 1) * n]
    return out


def train(
    input_path: Path,
    target: str,
    horizons_min: list[int],
    models_dir: Path,
    model_id: str,
    val_size: float,
    test_size: float,
    data_source: str,
    changepoint_prior_scale: float,
    seasonality_mode: str,
    daily_seasonality: bool | str,
    weekly_seasonality: bool | str,
    yearly_seasonality: bool | str,
    parent_v1: str | None,
) -> None:
    from prophet import Prophet
    import prophet as prophet_pkg

    t0 = time.perf_counter()
    df, provenance = load_dataset(input_path, target=target)
    df = _build_ds_column(df)
    df = df.dropna(subset=["ds", target]).sort_values("ds").reset_index(drop=True)

    df, target_cols = build_target_lags(df, target=target, horizons_min=horizons_min)
    train_df, val_df, test_df = three_way_time_split(
        df, val_size=val_size, test_size=test_size
    )

    prophet_train = pd.DataFrame({"ds": train_df["ds"], "y": train_df[target]})
    logger.info(
        "Prophet: training on %d rows (val=%d, test=%d), range %s -> %s; %d horizons; "
        "cps=%s seasonality=%s daily=%s weekly=%s yearly=%s",
        len(prophet_train), len(val_df), len(test_df),
        prophet_train["ds"].min(), prophet_train["ds"].max(), len(horizons_min),
        changepoint_prior_scale, seasonality_mode,
        daily_seasonality, weekly_seasonality, yearly_seasonality,
    )

    # uncertainty_samples=0 cuts ~6 GB peak RAM on hybrid dataset; we only use yhat.
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        uncertainty_samples=0,
    )
    model.fit(prophet_train)

    # Sanity check on val partition.
    val_pred = _predict_for_test_set(model, val_df["ds"].to_numpy(), horizons_min)
    val_true = np.column_stack([val_df[col].to_numpy() for col in target_cols])
    val_metrics = per_horizon_metrics(val_true, val_pred, horizons_min)
    logger.info(
        "VAL sanity: rmse@15=%.4f rmse@60=%.4f mae@15=%.4f",
        val_metrics["rmse_per_horizon"].get("15", float("nan")),
        val_metrics["rmse_per_horizon"].get("60", float("nan")),
        val_metrics["mae_per_horizon"].get("15", float("nan")),
    )

    # Final metrics on held-out test partition.
    y_pred = _predict_for_test_set(model, test_df["ds"].to_numpy(), horizons_min)
    y_true = np.column_stack([test_df[col].to_numpy() for col in target_cols])
    metrics = per_horizon_metrics(y_true, y_pred, horizons_min)
    duration = time.perf_counter() - t0

    logger.info(
        "TEST: prophet trained in %.2fs; rmse@15=%.4f rmse@60=%.4f mae@15=%.4f r2@15=%.4f",
        duration,
        metrics["rmse_per_horizon"].get("15", float("nan")),
        metrics["rmse_per_horizon"].get("60", float("nan")),
        metrics["mae_per_horizon"].get("15", float("nan")),
        metrics["r2_per_horizon"].get("15", float("nan")),
    )

    ensure_dirs()
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib_path, meta_path = artifact_paths(models_dir, model_id)
    joblib.dump(model, joblib_path, compress=3)

    best_hyperparams = {
        "changepoint_prior_scale": changepoint_prior_scale,
        "seasonality_mode": seasonality_mode,
        "daily_seasonality": daily_seasonality,
        "weekly_seasonality": weekly_seasonality,
        "yearly_seasonality": yearly_seasonality,
        "uncertainty_samples": 0,
    }
    training_extra = {
        "tuned": parent_v1 is not None,
        "best_hyperparams": best_hyperparams,
        "val_metrics": {
            "rmse_per_horizon": val_metrics["rmse_per_horizon"],
            "mae_per_horizon": val_metrics["mae_per_horizon"],
            "r2_per_horizon": val_metrics["r2_per_horizon"],
        },
        "val_size": val_size,
    }
    if parent_v1:
        training_extra["parent_v1"] = parent_v1

    write_meta_json(
        meta_path,
        model_id=model_id,
        algo="prophet",
        target=target,
        horizons_min=horizons_min,
        feature_columns=["ds"],
        metrics_per_horizon=metrics,
        train_rows=len(train_df),
        test_rows=len(test_df),
        data_source=data_source,
        device_used="cpu",
        framework_version=f"prophet=={prophet_pkg.__version__}",
        training_duration_seconds=duration,
        dataset_rows=provenance["rows_after_dropna"],
        provenance=provenance,
        test_size=test_size,
        notes=(
            "Prophet uses ds (timestamp) only; features=['ds']. "
            f"Hyperparams: cps={changepoint_prior_scale}, seasonality={seasonality_mode}, "
            f"daily={daily_seasonality}, weekly={weekly_seasonality}, yearly={yearly_seasonality}. "
            f"Split: train/val/test = {1-val_size-test_size:.2f}/{val_size:.2f}/{test_size:.2f} time-based; "
            "metrics reported on held-out test partition (val for sanity only). "
            f"{'Tuned attempt; parent=' + parent_v1 + '.' if parent_v1 else ''}"
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
        f"OK prophet {model_id}: train={len(train_df)} val={len(val_df)} "
        f"test={len(test_df)} duration={duration:.2f}s "
        f"test_rmse@15={metrics['rmse_per_horizon'].get('15', float('nan')):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prophet trainer (multi-horizon)")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target", default="temp_c")
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS_MIN))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--data-source", choices=["real", "synthetic", "hybrid"], default="real")
    parser.add_argument(
        "--changepoint-prior-scale",
        type=float,
        default=0.05,
        help="Prophet trend changepoint flexibility (Prophet default 0.05).",
    )
    parser.add_argument(
        "--seasonality-mode",
        choices=["additive", "multiplicative"],
        default="additive",
    )
    parser.add_argument(
        "--daily-seasonality",
        type=_parse_seasonality_flag,
        default="auto",
        help="true|false|auto (default auto)",
    )
    parser.add_argument(
        "--weekly-seasonality",
        type=_parse_seasonality_flag,
        default="auto",
    )
    parser.add_argument(
        "--yearly-seasonality",
        type=_parse_seasonality_flag,
        default="auto",
    )
    parser.add_argument(
        "--parent-v1",
        default=None,
        help="If this is a tuned re-run, set the parent v1 model_id for provenance.",
    )
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
        val_size=args.val_size,
        test_size=args.test_size,
        data_source=args.data_source,
        changepoint_prior_scale=args.changepoint_prior_scale,
        seasonality_mode=args.seasonality_mode,
        daily_seasonality=args.daily_seasonality,
        weekly_seasonality=args.weekly_seasonality,
        yearly_seasonality=args.yearly_seasonality,
        parent_v1=args.parent_v1,
    )


if __name__ == "__main__":
    main()
