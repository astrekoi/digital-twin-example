"""Merge a real telemetry CSV and a synthetic CSV into a hybrid dataset.

The synthetic rows are placed first (older timestamps), the real rows last.
If the timestamp ranges overlap the synthetic dataset is shifted into the past
so its last row immediately precedes the first real row with no gap.

Output includes a sidecar JSON (<output>.meta.json) recording:
  rows_real, rows_synthetic, ts_range_real, ts_range_synthetic, data_source.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or header-less CSV: {path}")
        headers = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    return headers, rows


def _check_schema(real_headers: list[str], synth_headers: list[str]) -> None:
    real_set = set(real_headers)
    synth_set = set(synth_headers)
    if real_set != synth_set:
        only_real = real_set - synth_set
        only_synth = synth_set - real_set
        parts = []
        if only_real:
            parts.append(f"only in real: {sorted(only_real)}")
        if only_synth:
            parts.append(f"only in synthetic: {sorted(only_synth)}")
        raise ValueError("Column schema mismatch - " + "; ".join(parts))


def _ts_unix(row: dict) -> float:
    try:
        return float(row["ts_unix"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Row missing valid ts_unix: {row!r}") from exc


def merge_datasets(
    real_path: Path,
    synthetic_path: Path,
    output_path: Path,
    *,
    prepend_synthetic: bool = True,
    seed: int = 42,
) -> dict:
    """Merge synthetic + real CSVs into a hybrid training dataset.

    Synthetic rows are shifted into the past so the last synthetic row's
    ts_unix immediately precedes the first real row. Writes <output>.meta.json
    alongside the merged CSV. Returns merge statistics.

    prepend_synthetic=False is not yet implemented; seed is reserved.
    """
    if not prepend_synthetic:
        raise NotImplementedError("prepend_synthetic=False is not yet implemented")

    real_headers, real_rows = _read_csv(real_path)
    synth_headers, synth_rows = _read_csv(synthetic_path)

    _check_schema(real_headers, synth_headers)

    if not real_rows:
        raise ValueError(f"Real dataset is empty: {real_path}")
    if not synth_rows:
        raise ValueError(f"Synthetic dataset is empty: {synthetic_path}")

    real_timestamps = [_ts_unix(r) for r in real_rows]
    synth_timestamps = [_ts_unix(r) for r in synth_rows]

    real_min = min(real_timestamps)
    real_max = max(real_timestamps)
    synth_min = min(synth_timestamps)
    synth_max = max(synth_timestamps)

    # Shift synthetic into the past so its last row's ts_unix < real first row's ts_unix.
    # We place synthetic to end exactly 1 sample-period before real starts.
    # Estimate sample period from real dataset (or synthetic if real has 1 row).
    if len(real_timestamps) >= 2:
        real_period = (real_max - real_min) / (len(real_timestamps) - 1)
    elif len(synth_timestamps) >= 2:
        real_period = (synth_max - synth_min) / (len(synth_timestamps) - 1)
    else:
        real_period = 1.0

    # Target: synth_max_shifted = real_min - real_period
    required_synth_max = real_min - real_period
    shift = required_synth_max - synth_max  # may be negative (shift back) or positive (already ok)

    shifted_synth_rows = []
    for row in synth_rows:
        new_row = dict(row)
        new_ts = _ts_unix(row) + shift
        new_row["ts_unix"] = f"{new_ts:.6f}"
        shifted_synth_rows.append(new_row)

    merged = shifted_synth_rows + real_rows

    all_ts = [_ts_unix(r) for r in merged]
    if len(set(f"{t:.3f}" for t in all_ts)) < len(all_ts):
        # Duplicates at ms precision are unlikely but warn rather than fail.
        import warnings

        warnings.warn(
            "Merged dataset has duplicate ts_unix values at millisecond precision", stacklevel=2
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = real_headers
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    shifted_ts = [_ts_unix(r) for r in shifted_synth_rows]
    meta = {
        "rows_real": len(real_rows),
        "rows_synthetic": len(synth_rows),
        "rows_total": len(merged),
        "ts_range_real": {"min": real_min, "max": real_max},
        "ts_range_synthetic": {"min": min(shifted_ts), "max": max(shifted_ts)},
        "synthetic_shift_s": shift,
        "data_source": "hybrid",
        "real_path": str(real_path),
        "synthetic_path": str(synthetic_path),
        "output_path": str(output_path),
    }

    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge a real CSV and a synthetic CSV into a hybrid training dataset"
    )
    p.add_argument("--real", required=True, metavar="PATH", help="Path to real telemetry CSV")
    p.add_argument("--synthetic", required=True, metavar="PATH", help="Path to synthetic CSV")
    p.add_argument("--output", required=True, metavar="PATH", help="Output merged CSV path")
    p.add_argument(
        "--prepend-synthetic",
        action="store_true",
        default=True,
        help="Place synthetic rows first with older timestamps (default: true)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed (reserved for future jitter)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    real_path = Path(args.real)
    synth_path = Path(args.synthetic)
    output_path = Path(args.output)

    for p, label in [(real_path, "--real"), (synth_path, "--synthetic")]:
        if not p.exists():
            print(f"ERROR: {label} path not found: {p}", file=sys.stderr)
            return 2

    try:
        meta = merge_datasets(
            real_path,
            synth_path,
            output_path,
            prepend_synthetic=args.prepend_synthetic,
            seed=args.seed,
        )
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    print(
        f"Merged {meta['rows_synthetic']} synthetic + {meta['rows_real']} real"
        f" = {meta['rows_total']} rows -> {output_path}"
    )
    print(f"Sidecar meta: {meta_path}")
    print(f"data_source: {meta['data_source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
