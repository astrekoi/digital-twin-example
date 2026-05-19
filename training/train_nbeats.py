"""N-BEATS (via Darts) for multi-horizon forecasting.

EXTERNAL MACHINE ONLY.

Strategy:
- Resample the 1 Hz telemetry to 1-minute frequency (mean per minute) so
  output_chunk_length=60 covers the maximum horizon (60 min) without exploding
  GPU memory.
- Train ``darts.models.NBEATSModel`` on the train partition (first 80%).
- Evaluate via ``historical_forecasts`` with stride=15 over the test partition,
  pulling per-horizon predictions at minute offsets [1, 5, 10, 15, 30, 60].

Artifact format (validator-compatible .joblib): a dict containing
``darts_state_dict``, ``architecture_config``, ``horizons_min``,
``feature_columns`` (target only), and ``framework="darts"``. The Pi-side
loader does not yet rehydrate Darts models - see Pi TODO.
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
    device_pick_torch,
    ensure_dirs,
    load_dataset,
    parse_horizons,
    per_horizon_metrics,
    validate_artifact_inline,
    write_meta_json,
)

logger = logging.getLogger(__name__)


def _build_minute_series(df: pd.DataFrame, target: str) -> pd.Series:
    if "ts_iso" in df.columns:
        ts = pd.to_datetime(df["ts_iso"], utc=True, errors="coerce").dt.tz_convert(None)
    elif "ts_unix" in df.columns:
        ts = pd.to_datetime(df["ts_unix"], unit="s", utc=True).dt.tz_convert(None)
    else:
        raise ValueError("CSV must have ts_iso or ts_unix")
    s = pd.Series(df[target].to_numpy(dtype=np.float64), index=ts).dropna()
    s = s.sort_index()
    s = s.resample("1min").mean().dropna()
    s = s.asfreq("1min").ffill().bfill()
    return s


def train(
    input_path: Path,
    target: str,
    horizons_min: list[int],
    models_dir: Path,
    model_id: str,
    test_size: float,
    data_source: str,
    input_chunk_length: int,
    output_chunk_length: int,
    n_epochs: int,
    batch_size: int,
    stride: int,
    max_train_time_hours: float,
    subsample_rows: int,
) -> None:
    import torch
    from darts import TimeSeries
    from darts.models import NBEATSModel
    import darts

    if max(horizons_min) > output_chunk_length:
        raise ValueError(
            f"output_chunk_length={output_chunk_length} must cover max horizon {max(horizons_min)}"
        )

    t0 = time.perf_counter()
    df, provenance = load_dataset(input_path, target=target)

    rows_pre_subsample = len(df)
    if subsample_rows > 0 and len(df) > subsample_rows:
        df = df.iloc[-subsample_rows:].reset_index(drop=True)
        logger.warning(
            "subsample applied (raw 1Hz): kept last %d of %d rows (time-based, preserves recency)",
            subsample_rows, rows_pre_subsample,
        )

    s = _build_minute_series(df, target)
    n_minutes = len(s)
    logger.info("nbeats: 1-min series length=%d (%.1f hours)", n_minutes, n_minutes / 60)

    if n_minutes < input_chunk_length + output_chunk_length + 50:
        raise RuntimeError(
            f"insufficient minute-series rows ({n_minutes}); need at least "
            f"{input_chunk_length + output_chunk_length + 50}"
        )

    split_idx = int(n_minutes * (1.0 - test_size))
    s_train = s.iloc[:split_idx]
    s_full = s

    ts_train = TimeSeries.from_series(s_train)
    ts_full = TimeSeries.from_series(s_full)

    device = device_pick_torch()
    accelerator = "gpu" if device == "cuda" else "cpu"
    # Convert max_train_time_hours into Lightning's max_time format. Lightning
    # checks max_time at the end of each batch and stops gracefully - perfect
    # for an unattended run-against-budget on CPU.
    max_time_hours_int = max(0, int(max_train_time_hours))
    max_time_minutes_int = int((max_train_time_hours - max_time_hours_int) * 60)
    pl_trainer_kwargs: dict = {
        "accelerator": accelerator,
        "devices": 1,
        "enable_progress_bar": False,
        "max_time": {"hours": max_time_hours_int, "minutes": max_time_minutes_int},
    }
    if accelerator == "gpu":
        pl_trainer_kwargs["precision"] = "32-true"

    logger.info(
        "nbeats: device=%s accelerator=%s, input_chunk=%d, output_chunk=%d, epochs=%d",
        device, accelerator, input_chunk_length, output_chunk_length, n_epochs,
    )

    torch.manual_seed(42)
    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        num_blocks=3,
        num_layers=4,
        layer_widths=128,
        n_epochs=n_epochs,
        batch_size=batch_size,
        random_state=42,
        pl_trainer_kwargs=pl_trainer_kwargs,
    )
    model.fit(ts_train)

    # Rolling forecasts over the test partition.
    forecasts = model.historical_forecasts(
        series=ts_full,
        start=split_idx,
        forecast_horizon=max(horizons_min),
        stride=stride,
        last_points_only=False,
        retrain=False,
        verbose=False,
        show_warnings=False,
    )
    n_forecasts = len(forecasts)
    if n_forecasts == 0:
        raise RuntimeError("historical_forecasts returned empty list - test partition too small")

    horizons_idx = [h - 1 for h in horizons_min]
    # darts >=0.30 renamed pd_series() -> to_series().
    full_series = ts_full.to_series() if hasattr(ts_full, "to_series") else ts_full.pd_series()

    y_pred_arr = np.zeros((n_forecasts, len(horizons_min)), dtype=np.float64)
    y_true_arr = np.zeros((n_forecasts, len(horizons_min)), dtype=np.float64)
    for i, fc in enumerate(forecasts):
        fc_values = fc.values().flatten()
        gt_values = full_series.loc[fc.time_index].to_numpy()
        for j, idx in enumerate(horizons_idx):
            y_pred_arr[i, j] = fc_values[idx]
            y_true_arr[i, j] = gt_values[idx]

    metrics = per_horizon_metrics(y_true_arr, y_pred_arr, horizons_min)
    duration = time.perf_counter() - t0

    logger.info(
        "nbeats trained in %.2fs; n_forecasts=%d  rmse@15=%.4f rmse@60=%.4f device=%s",
        duration, n_forecasts,
        metrics["rmse_per_horizon"].get("15", float("nan")),
        metrics["rmse_per_horizon"].get("60", float("nan")),
        device,
    )

    ensure_dirs()
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib_path, meta_path = artifact_paths(models_dir, model_id)
    state_payload = {
        "darts_state_dict": {k: v.cpu() for k, v in model.model.state_dict().items()},
        "architecture_config": {
            "input_chunk_length": input_chunk_length,
            "output_chunk_length": output_chunk_length,
            "num_blocks": 3,
            "num_layers": 4,
            "layer_widths": 128,
        },
        "horizons_min": list(horizons_min),
        "feature_columns": [target],
        "framework": "darts",
        "darts_version": darts.__version__,
        "torch_version": torch.__version__,
    }
    joblib.dump(state_payload, joblib_path, compress=3)

    write_meta_json(
        meta_path,
        model_id=model_id,
        algo="nbeats",
        target=target,
        horizons_min=horizons_min,
        feature_columns=[target],
        metrics_per_horizon=metrics,
        train_rows=split_idx,
        test_rows=n_minutes - split_idx,
        data_source=data_source,
        device_used=device,
        framework_version=f"darts=={darts.__version__}, torch=={torch.__version__}",
        training_duration_seconds=duration,
        dataset_rows=n_minutes,
        provenance=provenance,
        test_size=test_size,
        notes=(
            f"Darts N-BEATS (input_chunk={input_chunk_length}, output_chunk={output_chunk_length}, "
            f"target_epochs={n_epochs}, batch={batch_size}, "
            f"max_train_time={max_train_time_hours}h via Lightning max_time). "
            f"Resampled to 1-minute. "
            f"Subsample: {'yes (' + str(subsample_rows) + ' of ' + str(rows_pre_subsample) + ' raw 1Hz rows kept)' if subsample_rows > 0 and rows_pre_subsample > subsample_rows else 'no'}. "
            f"Evaluation: historical_forecasts stride={stride} over test partition. "
            "Pi-side loader does not yet rehydrate Darts models - see Pi TODO."
        ),
        extra={
            "artifact_type": "darts_torch",
            "stride": stride,
            "resample": "1min",
            "max_train_time_hours": max_train_time_hours,
            "target_epochs": n_epochs,
            "subsample_rows": subsample_rows,
            "rows_pre_subsample": rows_pre_subsample,
        },
    )

    ok, errors, warnings = validate_artifact_inline(joblib_path, meta_path)
    if not ok:
        logger.error("validator errors: %s", errors)
        sys.exit(2)
    if warnings:
        logger.warning("validator warnings: %s", warnings)
    print(f"OK nbeats {model_id}: train_min={split_idx} test_min={n_minutes - split_idx} "
          f"forecasts={n_forecasts} duration={duration:.2f}s device={device}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Darts N-BEATS trainer (multi-horizon)")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target", default="temp_c")
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS_MIN))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--data-source", choices=["real", "synthetic", "hybrid"], default="real")
    parser.add_argument("--input-chunk-length", type=int, default=60)
    parser.add_argument("--output-chunk-length", type=int, default=60)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument(
        "--max-train-time-hours",
        type=float,
        default=4.0,
        help="Stop training after this many hours (Lightning max_time). Default 4.",
    )
    parser.add_argument(
        "--subsample-rows",
        type=int,
        default=0,
        help="If >0, keep only the last N rows of the raw 1Hz CSV (time-based, before resampling).",
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
        test_size=args.test_size,
        data_source=args.data_source,
        input_chunk_length=args.input_chunk_length,
        output_chunk_length=args.output_chunk_length,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        stride=args.stride,
        max_train_time_hours=args.max_train_time_hours,
        subsample_rows=args.subsample_rows,
    )


if __name__ == "__main__":
    main()
