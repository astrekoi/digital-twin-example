"""Synthetic telemetry dataset generator - for AI smoke pipeline only.

Produces a CSV in the same schema as `readings` (src/db.py) from seeded
deterministic physical models. NOT a substitute for real data: never train
thesis-final models on this output. Artifacts derived from it must have
data_source="synthetic" in their .meta.json.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

COLUMNS = [
    "ts_iso",
    "ts_unix",
    "temp_c",
    "humidity_pct",
    "pressure_hpa",
    "lux",
    "current_ma",
    "bus_voltage",
    "shunt_mv",
    "power_mw",
    "ds18b20_c",
    "mq2_raw",
    "mq2_voltage",
    "pir_state",
]


def _solar_lux(t_seconds_local: float, day_seconds: float = 86400.0) -> float:
    """Daily lux profile: 0 at night, smooth ramp through dawn/dusk, plateau midday.

    Uses cos² envelope between sunrise and sunset; 0 outside.
    """
    sunrise = 6.0 * 3600.0
    sunset = 20.0 * 3600.0
    if t_seconds_local < sunrise or t_seconds_local > sunset:
        return 0.0
    span = sunset - sunrise
    phase = (t_seconds_local - sunrise) / span  # 0..1
    envelope = math.sin(phase * math.pi) ** 2
    return 50.0 + envelope * 950.0  # 50..1000


def _temp_profile(t_seconds_local: float, week_phase: float) -> float:
    """Indoor temp diurnal cycle 18..26°C plus slow weekly drift +-1°C."""
    diurnal = 22.0 + 4.0 * math.sin(2 * math.pi * (t_seconds_local - 6 * 3600) / 86400.0)
    weekly = 1.0 * math.sin(week_phase)
    return diurnal + weekly


def _humidity_from_temp(temp_c: float, rng: random.Random) -> float:
    """Anti-correlation with temp; clamp to physical range."""
    base = 90.0 - (temp_c - 18.0) * 4.0  # 18°C -> 90%, 26°C -> 58%
    return max(20.0, min(95.0, base + rng.gauss(0.0, 3.0)))


def _pressure_walk(prev: float, rng: random.Random) -> float:
    """Ornstein-Uhlenbeck-ish walk around 1013 hPa, bounded 990..1025."""
    drift = (1013.0 - prev) * 0.001
    next_val = prev + drift + rng.gauss(0.0, 0.05)
    return max(990.0, min(1025.0, next_val))


def _is_in_fan_event(t_unix: float, rng: random.Random) -> bool:
    """Fan turns on for 5 min every ~30 min (deterministic via slot hash)."""
    slot = int(t_unix // 60)  # minute slot
    cycle = slot % 30
    return cycle < 5


def _current_for_event(in_event: bool, since_event_start_s: int, rng: random.Random) -> float:
    """Inrush ~180-220 mA for first ~15 s, then steady 80-120 mA, else ~0."""
    if not in_event:
        return rng.gauss(0.0, 0.5)
    if since_event_start_s < 15:
        return rng.uniform(180.0, 220.0)
    return rng.uniform(80.0, 120.0)


def _mq2_baseline(rng: random.Random, in_event: bool) -> float:
    """Baseline 0.10..0.30; during a 'gas event' lifts to 0.55..0.80."""
    if in_event:
        return rng.uniform(0.55, 0.80)
    return rng.uniform(0.10, 0.30)


def _pir_for_hour(hour_of_day: int, rng: random.Random) -> int:
    """PIR more likely during day (6..23); rare at night."""
    if 6 <= hour_of_day <= 23:
        p = 0.05  # ~5% sample fires during day
    else:
        p = 0.005
    return 1 if rng.random() < p else 0


def generate(
    output_path: Path,
    duration_days: float,
    sample_period_s: int,
    seed: int,
    start_iso: str | None,
) -> dict:
    """Generate synthetic telemetry CSV. Returns summary dict."""
    rng = random.Random(seed)
    if start_iso:
        start_dt = datetime.fromisoformat(start_iso)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
    else:
        total_s = int(duration_days * 86400.0)
        start_dt = datetime.now(timezone.utc) - timedelta(seconds=total_s)

    duration_s = int(duration_days * 86400.0)
    n_rows = duration_s // sample_period_s
    if n_rows < 1:
        raise ValueError(
            f"duration_days={duration_days} too small for sample_period_s={sample_period_s}"
        )

    pressure = 1013.0
    fan_event_start_unix: float | None = None
    mq2_event_remaining = 0  # samples
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        for i in range(n_rows):
            ts_dt = start_dt + timedelta(seconds=i * sample_period_s)
            t_unix = ts_dt.timestamp()
            t_local_seconds = ts_dt.hour * 3600 + ts_dt.minute * 60 + ts_dt.second
            week_phase = (t_unix / 86400.0) * 2 * math.pi / 7.0

            temp = _temp_profile(t_local_seconds, week_phase) + rng.gauss(0.0, 0.3)
            humidity = _humidity_from_temp(temp, rng)
            pressure = _pressure_walk(pressure, rng)
            lux = _solar_lux(t_local_seconds) + rng.gauss(0.0, 5.0)
            lux = max(0.0, lux)

            in_fan = _is_in_fan_event(t_unix, rng)
            if in_fan and fan_event_start_unix is None:
                fan_event_start_unix = t_unix
            elif not in_fan and fan_event_start_unix is not None:
                fan_event_start_unix = None
            since_start_s = int(t_unix - fan_event_start_unix) if fan_event_start_unix else 0
            current_ma = _current_for_event(in_fan, since_start_s, rng)

            bus_voltage = 5.0 + rng.gauss(0.0, 0.02)
            shunt_mv = current_ma * 0.1  # 0.1 Ω shunt
            power_mw = max(0.0, bus_voltage * current_ma)

            # MQ-2: occasionally trigger a gas event lasting a few minutes
            if mq2_event_remaining == 0 and rng.random() < (
                1.0 / (12 * 60 * 60 // sample_period_s)
            ):
                # ~2 events/day on average
                mq2_event_remaining = rng.randint(120 // sample_period_s, 1800 // sample_period_s)
            in_mq2_event = mq2_event_remaining > 0
            if in_mq2_event:
                mq2_event_remaining -= 1
            mq2_raw = _mq2_baseline(rng, in_mq2_event)
            mq2_voltage = mq2_raw * 5.0  # 5V->V jumper per CLAUDE.md

            # DS18B20: tracks temp_c with delay and lower noise
            ds18b20_c = temp + rng.gauss(0.0, 0.1) - 0.2  # slight offset bias

            pir_state = _pir_for_hour(ts_dt.hour, rng)

            writer.writerow(
                {
                    "ts_iso": ts_dt.isoformat(),
                    "ts_unix": round(t_unix, 3),
                    "temp_c": round(temp, 3),
                    "humidity_pct": round(humidity, 2),
                    "pressure_hpa": round(pressure, 2),
                    "lux": round(lux, 1),
                    "current_ma": round(current_ma, 2),
                    "bus_voltage": round(bus_voltage, 3),
                    "shunt_mv": round(shunt_mv, 3),
                    "power_mw": round(power_mw, 2),
                    "ds18b20_c": round(ds18b20_c, 3),
                    "mq2_raw": round(mq2_raw, 4),
                    "mq2_voltage": round(mq2_voltage, 4),
                    "pir_state": pir_state,
                }
            )

    size_bytes = output_path.stat().st_size
    end_dt = start_dt + timedelta(seconds=(n_rows - 1) * sample_period_s)
    return {
        "output": str(output_path),
        "rows": n_rows,
        "size_bytes": size_bytes,
        "start_iso": start_dt.isoformat(),
        "end_iso": end_dt.isoformat(),
        "sample_period_s": sample_period_s,
        "duration_days": duration_days,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic telemetry dataset generator (AI smoke pipeline)"
    )
    parser.add_argument("--output", type=Path, required=True, help="CSV output path")
    parser.add_argument(
        "--duration-days", type=float, default=1.0, help="Days of data (default 1.0)"
    )
    parser.add_argument(
        "--sample-period-s", type=int, default=5, help="Sample period seconds (default 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for reproducibility (default 42)"
    )
    parser.add_argument(
        "--start-iso", default=None, help="Start ISO timestamp (default: now - duration)"
    )
    args = parser.parse_args()

    summary = generate(
        output_path=args.output,
        duration_days=args.duration_days,
        sample_period_s=args.sample_period_s,
        seed=args.seed,
        start_iso=args.start_iso,
    )

    print(f"output:          {summary['output']}")
    print(f"rows:            {summary['rows']}")
    print(f"size_bytes:      {summary['size_bytes']}")
    print(f"start_iso:       {summary['start_iso']}")
    print(f"end_iso:         {summary['end_iso']}")
    print(f"sample_period_s: {summary['sample_period_s']}")
    print(f"duration_days:   {summary['duration_days']}")
    print(f"seed:            {summary['seed']}")


if __name__ == "__main__":
    main()
