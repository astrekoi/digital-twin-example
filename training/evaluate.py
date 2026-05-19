"""Compare ML models on a test set -> markdown table.

EXTERNAL MACHINE ONLY - do NOT run on Raspberry Pi.
Loads only joblib-compatible models; keras artifacts are skipped.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Bootstrap: works from any CWD without requiring PYTHONPATH.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.ai.features import build_features
from src.ai.registry import list_available_models

logger = logging.getLogger(__name__)

_ROWS_PER_MINUTE = 60


def _evaluate_model(
    entry: dict,
    features_df: pd.DataFrame,
    target: str,
    horizon_min: int,
) -> dict | None:
    """Evaluate one joblib model on features_df. Returns metrics dict or None."""
    meta = entry["meta"]

    # Skip keras and other non-joblib artifacts.
    if meta.get("artifact_type") and meta.get("artifact_type") != "joblib":
        logger.warning(
            "Skipping %s: artifact_type=%s (not joblib)",
            entry["model_id"], meta.get("artifact_type"),
        )
        return None

    # Load.
    try:
        import joblib
        model = joblib.load(entry["joblib_path"])
    except Exception as exc:
        logger.warning("Failed to load %s: %s", entry["model_id"], exc)
        return None

    feature_cols: list[str] | None = meta.get("features")
    if not feature_cols:
        logger.warning("%s: meta.features is empty - skipping", entry["model_id"])
        return None

    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        logger.warning("%s: missing features %s - skipping", entry["model_id"], missing)
        return None

    # Build the target column.
    horizon_rows = horizon_min * _ROWS_PER_MINUTE
    target_ahead = f"{target}_{horizon_min}min_ahead"
    eval_df = features_df.copy()
    eval_df[target_ahead] = eval_df[target].shift(-horizon_rows)
    eval_df = eval_df.dropna(subset=feature_cols + [target_ahead])

    if eval_df.empty:
        logger.warning("%s: no rows to evaluate after dropna", entry["model_id"])
        return None

    # Time-based 80/20 split.
    split_idx = int(len(eval_df) * 0.8)
    test_df = eval_df.iloc[split_idx:]
    if test_df.empty:
        logger.warning("%s: empty test set", entry["model_id"])
        return None

    try:
        X_test = test_df[feature_cols]
        y_test = test_df[target_ahead]
        y_pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
    except Exception as exc:
        logger.warning("predict error for %s: %s", entry["model_id"], exc)
        return None

    return {
        "model_id": entry["model_id"],
        "algo": meta.get("algo", "?"),
        "target": meta.get("target", target),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "train_rows": meta.get("train_rows", "?"),
        "test_rows": len(test_df),
        "trained_at": meta.get("trained_at", "?"),
    }


def _render_markdown(results: list[dict], input_path: Path, models_dir: Path) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# Model Comparison",
        "",
        f"Generated: {now}  ",
        f"Input: `{input_path}`  ",
        f"Models dir: `{models_dir}`",
        "",
        "| Model ID | Algo | Target | MAE | RMSE | R² | Train rows | Test rows | Trained at |",
        "|----------|------|--------|-----|------|----|-----------|-----------|-----------|",
    ]
    for r in results:
        lines.append(
            f"| {r['model_id']} | {r['algo']} | {r['target']} "
            f"| {r['mae']:.4f} | {r['rmse']:.4f} | {r['r2']:.4f} "
            f"| {r['train_rows']} | {r['test_rows']} | {r['trained_at']} |"
        )
    if not results:
        lines.append("| - | - | - | - | - | - | - | - | - |")
        lines.append("")
        lines.append("> No joblib models available for comparison.")
    lines.append("")
    return "\n".join(lines)


def evaluate(
    input_path: Path,
    models_dir: Path,
    output_path: Path,
    target: str,
    horizon_min: int,
    dry_run: bool,
) -> None:
    entries = list_available_models(models_dir)

    if not entries:
        logger.warning("models_dir empty or not found: %s", models_dir)
        print("[evaluate] No models to compare.")
        return

    logger.info("Models found: %d", len(entries))

    logger.info("Reading %s ...", input_path)
    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows", len(df))

    if target not in df.columns:
        logger.error("Column '%s' not found. Available: %s", target, list(df.columns))
        sys.exit(1)

    if "ts_unix" in df.columns:
        df = df.sort_values("ts_unix").reset_index(drop=True)

    logger.info("Building features ...")
    features_df = build_features(df)

    if dry_run:
        print(f"\n[dry-run] Input rows: {len(df)}")
        print(f"[dry-run] Features: {len(features_df.columns)}")
        print(f"[dry-run] Models found: {len(entries)}")
        for e in entries:
            atype = e["meta"].get("artifact_type", "joblib")
            print(f"  - {e['model_id']}  algo={e['meta'].get('algo','?')}  artifact={atype}")
        print("[dry-run] Evaluation and writing skipped.")
        return

    results = []
    for entry in entries:
        logger.info("Evaluating: %s", entry["model_id"])
        result = _evaluate_model(entry, features_df, target, horizon_min)
        if result is not None:
            results.append(result)
            logger.info(
                "  MAE=%.4f  RMSE=%.4f  R²=%.4f",
                result["mae"], result["rmse"], result["r2"],
            )

    md = _render_markdown(results, input_path, models_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    logger.info("Comparison written: %s", output_path)
    print(f"\nComparison: {output_path}")
    print(f"Models evaluated: {len(results)}/{len(entries)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model evaluation - EXTERNAL MACHINE ONLY"
    )
    parser.add_argument("--input", required=True, help="Path to telemetry CSV")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output", default="docs/model_comparison.md")
    parser.add_argument("--target", default="temp_c")
    parser.add_argument("--horizon-min", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    evaluate(
        input_path=Path(args.input),
        models_dir=Path(args.models_dir),
        output_path=Path(args.output),
        target=args.target,
        horizon_min=args.horizon_min,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
