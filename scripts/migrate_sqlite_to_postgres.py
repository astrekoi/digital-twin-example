"""Migrate readings/commands/predictions/events/proofs from SQLite to Postgres.

Applies migrations/postgres/001_init.sql (idempotent), then copies rows in
batches of 1000. proofs uses ON CONFLICT DO NOTHING to preserve existing
rows by idempotency_key.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import db as sqlite_db
from src.env import load_project_env

BATCH_SIZE = 1000


def _ts_unix_to_pg_ts(ts: float | None):
    if ts is None:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(float(ts), tz=timezone.utc)


def _migrate_table(pg_conn, sqlite_conn, table: str, columns: tuple[str, ...]) -> int:
    """Copy a table row by row (simple, deterministic, low memory)."""
    cur_sl = sqlite_conn.execute(f"SELECT {', '.join(columns)} FROM {table}")
    placeholders = ", ".join(["%s"] * len(columns))
    cols_sql = ", ".join(columns)
    total = 0
    batch: list = []
    with pg_conn.cursor() as pg_cur:
        for row in cur_sl:
            batch.append(tuple(row[c] for c in columns))
            if len(batch) >= BATCH_SIZE:
                pg_cur.executemany(
                    f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders})",
                    batch,
                )
                total += len(batch)
                batch.clear()
                pg_conn.commit()
                print(f"  {table}: {total} rows…")
        if batch:
            pg_cur.executemany(
                f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders})",
                batch,
            )
            total += len(batch)
            pg_conn.commit()
    return total


def _migrate_proofs(pg_conn, sqlite_conn) -> int:
    """proofs has UNIQUE on idempotency_key - use ON CONFLICT DO NOTHING."""
    cur_sl = sqlite_conn.execute(
        "SELECT idempotency_key, digest_ts_unix, pinata_cid, hedera_topic_id, "
        "hedera_consensus_ts, hedera_sequence_number, payload_blake2b FROM proofs"
    )
    total = 0
    with pg_conn.cursor() as pg_cur:
        for row in cur_sl:
            consensus_ts_dt = None
            if row["hedera_consensus_ts"] is not None:
                try:
                    consensus_ts_dt = _ts_unix_to_pg_ts(float(row["hedera_consensus_ts"]))
                except (TypeError, ValueError):
                    consensus_ts_dt = None
            pg_cur.execute(
                "INSERT INTO proofs (idempotency_key, digest_ts_unix, pinata_cid, "
                "hedera_topic_id, hedera_consensus_ts, hedera_sequence_number, "
                "payload_blake2b) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (idempotency_key) DO NOTHING",
                (
                    row["idempotency_key"],
                    row["digest_ts_unix"],
                    row["pinata_cid"],
                    row["hedera_topic_id"],
                    consensus_ts_dt,
                    row["hedera_sequence_number"],
                    row["payload_blake2b"],
                ),
            )
            total += 1
    pg_conn.commit()
    return total


def main() -> int:
    load_project_env(PROJECT_ROOT / ".env")

    sqlite_path = Path(os.environ.get("SQLITE_DB_PATH", "data/telemetry.db"))
    pg_dsn = os.environ.get("POSTGRES_DSN", "")
    if not pg_dsn:
        sys.exit("FATAL: POSTGRES_DSN is empty")
    if not sqlite_path.exists():
        sys.exit(f"FATAL: SQLite DB not found at {sqlite_path}")

    try:
        import psycopg
    except ImportError:
        sys.exit("FATAL: psycopg not installed - run uv sync --extra postgres")

    print(f"source SQLite: {sqlite_path}")
    print(f"target Postgres: {pg_dsn}")

    sqlite_conn = sqlite_db.open_connection(sqlite_path)

    schema_path = PROJECT_ROOT / "migrations" / "postgres" / "001_init.sql"
    print(f"applying schema: {schema_path}")
    pg_conn = psycopg.connect(pg_dsn, autocommit=False)
    try:
        with pg_conn.cursor() as cur:
            cur.execute(schema_path.read_text())
        pg_conn.commit()

        t0 = time.time()
        readings = _migrate_table(
            pg_conn,
            sqlite_conn,
            "readings",
            (
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
            ),
        )
        commands = _migrate_table(
            pg_conn,
            sqlite_conn,
            "commands",
            (
                "created_at",
                "actuator",
                "channel",
                "action",
                "value",
                "params",
                "status",
                "applied_at",
                "error_msg",
            ),
        )
        predictions = _migrate_table(
            pg_conn,
            sqlite_conn,
            "predictions",
            ("generated_at", "model_id", "horizon_min", "forecast_json", "features_json"),
        )
        events = _migrate_table(
            pg_conn,
            sqlite_conn,
            "events",
            ("ts_unix", "ts_iso", "event_type", "source", "message", "value"),
        )
        proofs = _migrate_proofs(pg_conn, sqlite_conn)

        elapsed = time.time() - t0
        print()
        print(f"DONE in {elapsed:.1f}s:")
        print(f"  readings    {readings}")
        print(f"  commands    {commands}")
        print(f"  predictions {predictions}")
        print(f"  events      {events}")
        print(f"  proofs      {proofs}")
    finally:
        pg_conn.close()
        sqlite_conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
