"""Export readings from SQLite to CSV for external-machine training.

Safe by default:
- opens SQLite with mode=ro, so a missing DB is not created;
- exports only the readings table;
- never touches hardware, collector, predictor, or commands;
- does not overwrite output unless --overwrite is passed.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

READINGS_COLUMNS = [
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


def connect_readonly(db_path: Path | str) -> sqlite3.Connection:
    """Open SQLite in read-only URI mode. Missing file raises OperationalError."""
    path = Path(db_path)
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _where_clause(since_ts: float | None, include_null: bool) -> tuple[str, list[float]]:
    clauses: list[str] = []
    params: list[float] = []
    if since_ts is not None:
        clauses.append("ts_unix >= ?")
        params.append(since_ts)
    if not include_null:
        nullable_columns = [col for col in READINGS_COLUMNS if col not in ("ts_iso", "ts_unix")]
        clauses.extend(f"{col} IS NOT NULL" for col in nullable_columns)
    if not clauses:
        return "", params
    return "WHERE " + " AND ".join(clauses), params


def count_rows(
    conn: sqlite3.Connection,
    *,
    since_ts: float | None = None,
    include_null: bool = True,
) -> int:
    where_sql, params = _where_clause(since_ts, include_null)
    row = conn.execute(f"SELECT COUNT(*) FROM readings {where_sql}", params).fetchone()
    return int(row[0])


def iter_readings(
    conn: sqlite3.Connection,
    *,
    since_ts: float | None = None,
    limit: int | None = None,
    include_null: bool = True,
) -> Iterable[sqlite3.Row]:
    where_sql, params = _where_clause(since_ts, include_null)
    limit_sql = ""
    if limit is not None:
        limit_sql = " LIMIT ?"
        params.append(limit)
    columns_sql = ", ".join(READINGS_COLUMNS)
    return conn.execute(
        f"SELECT {columns_sql} FROM readings {where_sql} ORDER BY ts_unix ASC{limit_sql}",
        params,
    )


def default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("data") / f"training_export_{stamp}.csv"


def export_readings_to_csv(
    db_path: Path | str,
    output_path: Path | str,
    *,
    since_ts: float | None = None,
    limit: int | None = None,
    include_null: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be > 0")

    output = Path(output_path)
    conn = connect_readonly(db_path)
    try:
        row_count = count_rows(conn, since_ts=since_ts, include_null=include_null)
        if limit is not None:
            row_count = min(row_count, limit)

        result = {
            "rows": row_count,
            "columns": list(READINGS_COLUMNS),
            "output": str(output),
            "dry_run": dry_run,
            "include_null": include_null,
        }

        if dry_run or row_count == 0:
            return result

        if output.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=READINGS_COLUMNS)
            writer.writeheader()
            for row in iter_readings(
                conn,
                since_ts=since_ts,
                limit=limit,
                include_null=include_null,
            ):
                writer.writerow({col: row[col] for col in READINGS_COLUMNS})
        return result
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export SQLite readings to CSV for external training"
    )
    parser.add_argument("--db", default="data/telemetry.db", help="SQLite DB path")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument(
        "--since-ts", type=float, default=None, help="Only export rows with ts_unix >= value"
    )
    parser.add_argument(
        "--last-hour",
        action="store_true",
        help="Shortcut: export only the last 3600 seconds (overrides --since-ts)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to export")
    parser.add_argument("--format", choices=["csv"], default="csv")
    parser.add_argument("--dry-run", action="store_true", help="Show row count and columns only")
    parser.add_argument(
        "--include-null",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include rows with NULL sensor values (default: true). Use --no-include-null to drop them.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacing an existing output file"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import time as _time

    parser = build_parser()
    args = parser.parse_args(argv)

    since_ts = args.since_ts
    if args.last_hour:
        since_ts = _time.time() - 3600.0

    output = Path(args.output) if args.output else default_output_path()
    try:
        result = export_readings_to_csv(
            args.db,
            output,
            since_ts=since_ts,
            limit=args.limit,
            include_null=args.include_null,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except (FileExistsError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    action = "Would export" if args.dry_run else "Exported"
    if result["rows"] == 0 and not args.dry_run:
        action = "No rows to export"
    print(f"{action} {result['rows']} rows to {result['output']}")
    print("Columns:", ", ".join(result["columns"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
