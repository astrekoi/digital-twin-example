"""LSTM (PyTorch) for multi-horizon forecasting.

EXTERNAL MACHINE ONLY. Replaces the previous Keras-based LSTM trainer.

Architecture (configurable via CLI flags):
    Input:  (batch, sequence_len, n_features)  - raw timesteps, no engineered features
    LSTM:   hidden_size=64, num_layers=2, batch_first=True, dropout=0.2
    Head:   Linear(hidden_size, n_horizons)  - single forward pass yields all horizons

Features: raw 6 numeric sensor columns (temp_c, humidity_pct, pressure_hpa, lux,
current_ma, mq2_raw). NaN in non-target features is forward-filled. Both inputs
and targets are standardized using a ``StandardScaler`` fit ONLY on the train
partition (no leakage). Early stopping tracks val loss (in scaled space) with
patience; on stop we restore the best weights and report metrics in the
original (unscaled) target space.

Split: 70 / 15 / 15 train / val / test, time-based (no shuffle). Val drives
early stopping; test is held out for final metrics.

Artifact format (validator-compatible .joblib wrapping torch state):
    joblib.dump({
        "state_dict": best_model.state_dict(),
        "architecture_config": {...},
        "horizons_min": [...],
        "feature_columns": [...],
        "scaler_X": StandardScaler,
        "scaler_y": StandardScaler,
        "feature_means": [...],   # legacy alias for scaler_X.mean_
        "feature_stds": [...],    # legacy alias for scaler_X.scale_
    }, "<id>.joblib")

The Pi-side loader (when implemented) must apply scaler_X to the input window
and inverse-transform predictions through scaler_y before consuming them.
"""
from __future__ import annotations

import argparse
import copy
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
    RAW_FEATURE_COLS,
    artifact_paths,
    build_target_lags,
    device_pick_torch,
    ensure_dirs,
    load_dataset,
    parse_horizons,
    per_horizon_metrics,
    validate_artifact_inline,
    write_meta_json,
)

logger = logging.getLogger(__name__)


def _make_sequences(
    X: np.ndarray, Y: np.ndarray, *, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a fixed window over (X, Y) producing (n_seq, seq_len, n_features) and
    (n_seq, n_horizons). Requires len(X) == len(Y) and >= seq_len.
    The label at index i is the target row at i + seq_len - 1 (last row of window)."""
    n, n_features = X.shape
    if n < seq_len:
        raise ValueError(f"need at least {seq_len} rows, got {n}")
    n_seq = n - seq_len + 1
    n_horizons = Y.shape[1]
    Xs = np.zeros((n_seq, seq_len, n_features), dtype=np.float32)
    Ys = np.zeros((n_seq, n_horizons), dtype=np.float32)
    for i in range(n_seq):
        Xs[i] = X[i : i + seq_len]
        Ys[i] = Y[i + seq_len - 1]
    return Xs, Ys


def _three_way_time_split(
    df: pd.DataFrame, *, val_size: float, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-based 3-way split: first (1-val-test) train, then val, then test (all chronological)."""
    n = len(df)
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError(f"invalid val_size={val_size} test_size={test_size}")
    test_idx = int(n * (1 - test_size))
    val_idx = int(n * (1 - test_size - val_size))
    return (
        df.iloc[:val_idx].copy(),
        df.iloc[val_idx:test_idx].copy(),
        df.iloc[test_idx:].copy(),
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
    sequence_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_train_time_hours: float,
    subsample_rows: int,
    patience: int,
    grad_clip: float,
) -> None:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    t0 = time.perf_counter()
    df, provenance = load_dataset(input_path, target=target)

    feature_columns = [c for c in RAW_FEATURE_COLS if c in df.columns]
    if not feature_columns:
        raise RuntimeError(f"none of {RAW_FEATURE_COLS} present in {input_path}")
    df[feature_columns] = df[feature_columns].ffill().bfill()

    df, target_cols = build_target_lags(df, target=target, horizons_min=horizons_min)
    df = df.dropna(subset=feature_columns + target_cols).reset_index(drop=True)

    rows_pre_subsample = len(df)
    if subsample_rows > 0 and len(df) > subsample_rows:
        df = df.iloc[-subsample_rows:].reset_index(drop=True)
        logger.warning(
            "subsample applied: kept last %d of %d rows (time-based, preserves recency)",
            subsample_rows, rows_pre_subsample,
        )

    train_df, val_df, test_df = _three_way_time_split(
        df, val_size=val_size, test_size=test_size
    )

    X_train_raw = train_df[feature_columns].to_numpy(dtype=np.float32)
    X_val_raw = val_df[feature_columns].to_numpy(dtype=np.float32)
    X_test_raw = test_df[feature_columns].to_numpy(dtype=np.float32)
    Y_train_raw = train_df[target_cols].to_numpy(dtype=np.float32)
    Y_val_raw = val_df[target_cols].to_numpy(dtype=np.float32)
    Y_test_raw = test_df[target_cols].to_numpy(dtype=np.float32)

    # Fit scalers on TRAIN ONLY - no data leakage from val/test.
    scaler_X = StandardScaler().fit(X_train_raw)
    scaler_y = StandardScaler().fit(Y_train_raw)
    X_train = scaler_X.transform(X_train_raw).astype(np.float32)
    X_val = scaler_X.transform(X_val_raw).astype(np.float32)
    X_test = scaler_X.transform(X_test_raw).astype(np.float32)
    Y_train_scaled = scaler_y.transform(Y_train_raw).astype(np.float32)
    Y_val_scaled = scaler_y.transform(Y_val_raw).astype(np.float32)

    Xs_train, Ys_train = _make_sequences(X_train, Y_train_scaled, seq_len=sequence_len)
    Xs_val, Ys_val = _make_sequences(X_val, Y_val_scaled, seq_len=sequence_len)
    # Test: keep raw Y for final metrics in original scale.
    Xs_test, Ys_test_raw = _make_sequences(X_test, Y_test_raw, seq_len=sequence_len)

    device = device_pick_torch()
    logger.info(
        "lstm: device=%s, seq_len=%d, train_seq=%d, val_seq=%d, test_seq=%d, "
        "n_features=%d, n_horizons=%d, patience=%d, grad_clip=%.2f",
        device, sequence_len, len(Xs_train), len(Xs_val), len(Xs_test),
        len(feature_columns), len(horizons_min), patience, grad_clip,
    )

    class LSTMForecaster(nn.Module):
        def __init__(self, n_features: int, hidden_size: int, num_layers: int,
                     dropout: float, n_horizons: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size, n_horizons)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(self.dropout(last))

    torch.manual_seed(42)
    model = LSTMForecaster(
        n_features=len(feature_columns),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        n_horizons=len(horizons_min),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # ReduceLROnPlateau halves the LR when val loss plateaus for 3 consecutive
    # epochs - independent of and finer-grained than the early-stop patience.
    # min_lr clamp prevents the LR from collapsing to numerical zero.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(Xs_train).to(device)
    Yt = torch.from_numpy(Ys_train).to(device)
    Xv = torch.from_numpy(Xs_val).to(device)
    Yv = torch.from_numpy(Ys_val).to(device)
    Xtest = torch.from_numpy(Xs_test).to(device)

    n_train = len(Xt)
    budget_seconds = max_train_time_hours * 3600.0
    train_start = time.perf_counter()
    actual_epochs = 0
    early_stopped_time = False
    early_stopped_patience = False
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    patience_counter = 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            x_batch = Xt[idx]
            y_batch = Yt[idx]
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            train_loss += float(loss.item()) * len(idx)
        train_loss /= n_train

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss = float(loss_fn(val_pred, Yv).item())
        actual_epochs = epoch
        elapsed = time.perf_counter() - train_start

        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "epoch %02d/%02d  train_mse=%.4f  val_mse=%.4f  best=%.4f@e%d  "
            "patience=%d/%d  lr=%.2e  elapsed=%.1fs",
            epoch, epochs, train_loss, val_loss, best_val_loss, best_epoch,
            patience_counter, patience, current_lr, elapsed,
        )

        if patience_counter >= patience:
            logger.warning(
                "early stop: val loss did not improve for %d epochs after best at e%d (best=%.4f)",
                patience, best_epoch, best_val_loss,
            )
            early_stopped_patience = True
            break
        if elapsed > budget_seconds and epoch < epochs:
            logger.warning(
                "time budget %s h exceeded after epoch %d/%d; stopping early",
                max_train_time_hours, epoch, epochs,
            )
            early_stopped_time = True
            break

    # Restore best weights for inference / artifact save.
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("restored best weights from epoch %d (val_mse=%.4f)", best_epoch, best_val_loss)

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(Xtest).cpu().numpy()
    # Inverse-transform predictions to original target scale.
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = Ys_test_raw
    metrics = per_horizon_metrics(y_true, y_pred, horizons_min)
    duration = time.perf_counter() - t0

    logger.info(
        "lstm trained in %.2fs; rmse@15=%.4f rmse@60=%.4f best_epoch=%d best_val_mse=%.4f device=%s",
        duration,
        metrics["rmse_per_horizon"].get("15", float("nan")),
        metrics["rmse_per_horizon"].get("60", float("nan")),
        best_epoch, best_val_loss, device,
    )

    ensure_dirs()
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib_path, meta_path = artifact_paths(models_dir, model_id)
    feature_means = scaler_X.mean_.astype(np.float64).tolist()
    feature_stds = scaler_X.scale_.astype(np.float64).tolist()
    state_payload = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "architecture_config": {
            "n_features": len(feature_columns),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "sequence_len": sequence_len,
            "n_horizons": len(horizons_min),
        },
        "horizons_min": list(horizons_min),
        "feature_columns": list(feature_columns),
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "framework": "pytorch",
    }
    joblib.dump(state_payload, joblib_path, compress=3)

    early_stop_label = ""
    if early_stopped_patience:
        early_stop_label = " [EARLY STOP - val patience]"
    elif early_stopped_time:
        early_stop_label = " [EARLY STOP - time budget]"

    write_meta_json(
        meta_path,
        model_id=model_id,
        algo="lstm",
        target=target,
        horizons_min=horizons_min,
        feature_columns=feature_columns,
        metrics_per_horizon=metrics,
        train_rows=len(Xs_train),
        test_rows=len(Xs_test),
        data_source=data_source,
        device_used=device,
        framework_version=f"torch=={torch.__version__}",
        training_duration_seconds=duration,
        dataset_rows=provenance["rows_after_dropna"],
        provenance=provenance,
        test_size=test_size,
        notes=(
            f"PyTorch LSTM (hidden={hidden_size}, layers={num_layers}, dropout={dropout}, "
            f"seq_len={sequence_len}, target_epochs={epochs}, actual_epochs={actual_epochs}, "
            f"best_epoch={best_epoch}, best_val_mse={best_val_loss:.4f} (scaled space)"
            + early_stop_label
            + f", batch={batch_size}, lr_init={learning_rate}, patience={patience}, "
            f"grad_clip={grad_clip}, lr_scheduler=ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)). "
            f"Split: train/val/test = {1-val_size-test_size:.2f}/{val_size:.2f}/{test_size:.2f} time-based. "
            f"Subsample: {'yes (' + str(subsample_rows) + ' of ' + str(rows_pre_subsample) + ' rows kept)' if subsample_rows > 0 and rows_pre_subsample > subsample_rows else 'no'}. "
            "Artifact = joblib of {state_dict, architecture_config, scaler_X, scaler_y, ...}. "
            "Pi-side loader does not yet support PyTorch state-dict reload - see Pi-side TODO."
        ),
        extra={
            "sequence_len": sequence_len,
            "artifact_type": "torch",
            "actual_epochs": actual_epochs,
            "target_epochs": epochs,
            "early_stopped_time_budget": early_stopped_time,
            "early_stopped_patience": early_stopped_patience,
            "early_stop_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "patience": patience,
            "grad_clip": grad_clip,
            "val_size": val_size,
            "max_train_time_hours": max_train_time_hours,
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
    print(
        f"OK lstm {model_id}: train_seq={len(Xs_train)} val_seq={len(Xs_val)} "
        f"test_seq={len(Xs_test)} duration={duration:.2f}s device={device} "
        f"best_epoch={best_epoch} best_val_mse={best_val_loss:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM (PyTorch) trainer (multi-horizon)")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target", default="temp_c")
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS_MIN))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--val-size", type=float, default=0.15, help="Val partition fraction (default 0.15)")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test partition fraction (default 0.15)")
    parser.add_argument("--data-source", choices=["real", "synthetic", "hybrid"], default="real")
    parser.add_argument("--sequence-len", type=int, default=60)
    parser.add_argument("--hidden-size", type=int, default=32, help="LSTM hidden size (default 32; was 64 pre-Stage 3)")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3, help="LSTM dropout (default 0.3; was 0.2 pre-Stage 3)")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs (default 50; early stop typical)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7, help="Early-stop patience on val loss (default 7; was 5 pre-Stage 3)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip max-norm (default 1.0; 0 disables)")
    parser.add_argument(
        "--max-train-time-hours",
        type=float,
        default=3.0,
        help="Stop training after this many hours (early stop). Default 3.",
    )
    parser.add_argument(
        "--subsample-rows",
        type=int,
        default=0,
        help="If >0, keep only the last N rows of the input (time-based, preserves recency). Default 0 (no subsample).",
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
        sequence_len=args.sequence_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_train_time_hours=args.max_train_time_hours,
        subsample_rows=args.subsample_rows,
        patience=args.patience,
        grad_clip=args.grad_clip,
    )


if __name__ == "__main__":
    main()
