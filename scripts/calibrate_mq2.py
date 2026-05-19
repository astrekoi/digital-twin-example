"""Record raw MQ-2 data for calibration. No ppm conversion.

CSV: data/mq2_calibration_YYYY-MM-DD.csv
Fields: ts_iso, ts_unix, label, raw, voltage_v, expected_vcc, notes

Default --expected-vcc 5V - current stand uses the 5V->V jumper.
3V3 is supported only as an explicit alternate hardware mode.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

CSV_FIELDNAMES = ["ts_iso", "ts_unix", "label", "raw", "voltage_v", "expected_vcc", "notes"]
_DEFAULT_VCC = "5V"


def main() -> None:
    parser = argparse.ArgumentParser(description="MQ-2 calibration data recorder")
    parser.add_argument(
        "--label", default="unlabeled", help="Environment label (clean_air, smoke, gas, ...)"
    )
    parser.add_argument("--duration-sec", type=int, default=60, help="Recording duration (s)")
    parser.add_argument(
        "--period-sec", type=float, default=1.0, help="Interval between samples (s)"
    )
    parser.add_argument(
        "--expected-vcc",
        default=_DEFAULT_VCC,
        choices=["3V3", "5V"],
        help=(
            "Analog IO jumper position. "
            f"Current stand: {_DEFAULT_VCC}->V. "
            "3V3 is an explicit alternate mode only."
        ),
    )
    parser.add_argument("--notes", default="", help="Free-form session notes")
    parser.add_argument(
        "--dry-run", action="store_true", help="Stub data, no real MQ-2 / troykahat"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    vref = 3.3 if args.expected_vcc == "3V3" else 5.0
    n_samples = max(1, int(args.duration_sec / args.period_sec))

    today = date.today().isoformat()
    csv_dir = Path("data")
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"mq2_calibration_{today}.csv"
    new_file = not csv_path.exists()

    logger.info(
        "MQ-2 calibration: label=%r vcc=%s vref=%.1fV samples=%d period=%.1fs -> %s",
        args.label,
        args.expected_vcc,
        vref,
        n_samples,
        args.period_sec,
        csv_path,
    )

    sensor: Any = None
    if not args.dry_run:
        try:
            from src.sensors.mq2 import MQ2Sensor

            sensor = MQ2Sensor(channel=0, expected_vcc=args.expected_vcc)
        except Exception as exc:
            logger.error("Failed to initialise MQ-2: %s", exc)
            logger.error("Run with --dry-run to test without hardware.")
            return
    else:
        logger.info("dry-run: stub data (raw=0.05 + linear trend)")

    written = 0
    try:
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if new_file:
                writer.writeheader()

            for i in range(n_samples):
                ts = datetime.now(timezone.utc)

                if args.dry_run:
                    raw: float | None = round(0.05 + 0.001 * i, 4)
                    voltage_v: float | None = round(raw * vref, 4)
                else:
                    try:
                        data = sensor.read()
                        raw = data["mq2_raw"]
                        voltage_v = data["mq2_voltage"]
                    except Exception as exc:
                        logger.warning("[%d/%d] MQ-2 read error: %s", i + 1, n_samples, exc)
                        raw = None
                        voltage_v = None

                writer.writerow(
                    {
                        "ts_iso": ts.isoformat(),
                        "ts_unix": ts.timestamp(),
                        "label": args.label,
                        "raw": raw,
                        "voltage_v": voltage_v,
                        "expected_vcc": args.expected_vcc,
                        "notes": args.notes,
                    }
                )
                f.flush()
                written += 1

                if raw is not None:
                    logger.info(
                        "[%d/%d] raw=%.4f voltage=%.3fV",
                        i + 1,
                        n_samples,
                        raw,
                        voltage_v,
                    )
                else:
                    logger.info("[%d/%d] raw=None (error)", i + 1, n_samples)

                if i < n_samples - 1:
                    time.sleep(args.period_sec)

    except KeyboardInterrupt:
        logger.info("Interrupted by user (recorded %d/%d)", written, n_samples)
    finally:
        if sensor is not None:
            sensor.close()

    logger.info("Recorded %d samples to %s", written, csv_path)


if __name__ == "__main__":
    main()
