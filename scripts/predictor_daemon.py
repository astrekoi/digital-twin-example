"""AI-prediction daemon: readings -> features -> predict -> predictions.

A separate process. Does NOT own GPIO and does NOT touch hardware.
Reads from SQLite, writes predictions to the predictions table.

Modes:
  --once       - single cycle and exit
  <no flags>   - loop every --interval seconds
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_collector_config, setup_logging
from src.db import fetch_readings_last_n_minutes, init_db, insert_prediction

logger = logging.getLogger(__name__)

_LOOKBACK_MINUTES = 4500  # 75 hours - covers full available history


def run_prediction_cycle(
    config: Any,
    predictor: Any,
    horizon_min: int,
) -> bool:
    """One cycle: read readings -> build features -> predict -> write to DB.

    Returns True on a successful predict.
    """
    import pandas as pd

    from src.ai.features import build_features, impute_raw_columns

    conn = init_db(config.db_path)
    try:
        rows = fetch_readings_last_n_minutes(conn, _LOOKBACK_MINUTES)
        if not rows:
            logger.warning(
                "predictor_daemon: no data in readings for the last %d min",
                _LOOKBACK_MINUTES,
            )
            return False

        df = pd.DataFrame(rows)
        df, raw_imputed = impute_raw_columns(df)
        features_df = build_features(df)

        result = predictor.predict(
            features_df,
            horizon_min=horizon_min,
            pre_imputed=raw_imputed,
        )
        if result is None:
            logger.warning("predictor_daemon: predict returned None - no prediction stored")
            return False

        insert_prediction(
            conn,
            model_id=result["model_id"],
            horizon_min=result["horizon_min"],
            forecast=[{"ts_offset_min": result["horizon_min"], "value": result["forecast"]}],
            features={
                "features_used": result["features_used"],
                "generated_at": result["generated_at"],
            },
        )
        imputed = result.get("imputed") or []
        logger.info(
            "Prediction stored: model=%s horizon=%dm forecast=%.4g imputed=%s",
            result["model_id"],
            result["horizon_min"],
            result["forecast"],
            imputed if imputed else "none",
        )
        return True
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="PLC Twin - predictor daemon")
    parser.add_argument("--config", default="configs/collector.yml")
    parser.add_argument("--once", action="store_true", help="One cycle and exit")
    parser.add_argument("--interval", type=int, default=60, help="Prediction interval (s)")
    parser.add_argument("--horizon", type=int, default=15, help="Forecast horizon (min)")
    parser.add_argument("--models-dir", default="models", help="Directory with model artifacts")
    parser.add_argument(
        "--model-id", default=None, help="Specific model_id to load (default: latest by trained_at)"
    )
    args = parser.parse_args()

    setup_logging()
    config = load_collector_config(Path(args.config))
    models_dir = Path(args.models_dir)

    logger.info(
        "predictor_daemon: once=%s interval=%ds horizon=%dmin models_dir=%s model_id=%s",
        args.once,
        args.interval,
        args.horizon,
        models_dir,
        args.model_id,
    )

    from src.ai.predictor import Predictor

    predictor = Predictor(models_dir=models_dir, model_id=args.model_id)
    if predictor.model is None:
        logger.warning("predictor_daemon: model unavailable (no files in %s)", models_dir)
        if args.once:
            return

    if args.once:
        run_prediction_cycle(config, predictor, args.horizon)
        return

    logger.info("predictor_daemon running. Ctrl+C to stop.")
    while True:
        try:
            run_prediction_cycle(config, predictor, args.horizon)
        except Exception:
            logger.exception("predictor_daemon: cycle error")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
