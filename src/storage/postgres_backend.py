"""Postgres telemetry backend (psycopg 3 + psycopg_pool).

Activated by DB_BACKEND=postgres in .env. Pool size is role-aware
(POOL_MAX_HW / POOL_MAX_AI / POOL_MAX_EXT / POOL_MAX_TAIPY).
Schema is in migrations/postgres/001_init.sql - run it before first use:

    psql "$POSTGRES_DSN" -f migrations/postgres/001_init.sql

or via scripts/migrate_sqlite_to_postgres.py which applies the schema and
then copies SQLite rows.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from src.role import Role, detect_role, pool_max_for_role

logger = logging.getLogger(__name__)


_READING_FIELDS = (
    "ts_iso", "ts_unix", "temp_c", "humidity_pct", "pressure_hpa", "lux",
    "current_ma", "bus_voltage", "shunt_mv", "power_mw", "ds18b20_c",
    "mq2_raw", "mq2_voltage", "pir_state",
)


class PostgresTelemetryStore:
    """TelemetryStore implementation backed by Postgres."""

    def __init__(self, dsn: str, role: Role | None = None) -> None:
        try:
            from psycopg_pool import ConnectionPool  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "psycopg-pool is not installed. Run: uv sync --extra postgres"
            ) from exc
        self._dsn = dsn
        self._role: Role = role or detect_role()
        self._pool_max = pool_max_for_role(self._role)
        self._pool: Any = ConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=self._pool_max,
            timeout=10.0,
            kwargs={"autocommit": False},
        )
        logger.info("Postgres pool opened: role=%s max=%d", self._role, self._pool_max)

    def init_schema(self) -> None:
        """Apply migrations/postgres/001_init.sql idempotently."""
        from pathlib import Path
        sql_path = Path(__file__).resolve().parents[2] / "migrations" / "postgres" / "001_init.sql"
        if not sql_path.exists():
            raise FileNotFoundError(f"migration file not found: {sql_path}")
        ddl = sql_path.read_text()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()

    def close(self) -> None:
        self._pool.close()

    # --- readings ---
    def insert_reading(self, row: dict) -> None:
        cols = ", ".join(_READING_FIELDS)
        placeholders = ", ".join(["%s"] * len(_READING_FIELDS))
        values = tuple(row.get(c) for c in _READING_FIELDS)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO readings ({cols}) VALUES ({placeholders})",
                    values,
                )
            conn.commit()

    def fetch_latest_reading(self) -> dict | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM readings ORDER BY ts_unix DESC LIMIT 1"
                )
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [d.name for d in cur.description]
                return dict(zip(cols, row))

    def fetch_recent_readings(self, minutes: int = 60) -> list[dict]:
        cutoff = time.time() - minutes * 60
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM readings WHERE ts_unix >= %s ORDER BY ts_unix ASC",
                    (cutoff,),
                )
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]

    def fetch_readings_range(
        self, from_unix: float, to_unix: float, step_s: float = 0
    ) -> list[dict]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                if step_s and step_s > 0:
                    cur.execute(
                        """
                        SELECT * FROM readings
                        WHERE ts_unix BETWEEN %s AND %s
                          AND id IN (
                              SELECT MIN(id) FROM readings
                              WHERE ts_unix BETWEEN %s AND %s
                              GROUP BY ((ts_unix - %s) / %s)::int
                          )
                        ORDER BY ts_unix ASC
                        """,
                        (from_unix, to_unix, from_unix, to_unix, from_unix, step_s),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM readings WHERE ts_unix BETWEEN %s AND %s ORDER BY ts_unix ASC",
                        (from_unix, to_unix),
                    )
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]

    # --- predictions ---
    def insert_prediction(
        self,
        model_id: str,
        horizon_min: int,
        forecast: list[dict],
        features: dict | None = None,
    ) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO predictions (generated_at, model_id, horizon_min, "
                    "forecast_json, features_json) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)",
                    (
                        time.time(),
                        model_id,
                        horizon_min,
                        json.dumps(forecast),
                        json.dumps(features) if features is not None else None,
                    ),
                )
            conn.commit()

    def fetch_recent_predictions(self, limit: int = 20) -> list[dict]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM predictions ORDER BY generated_at DESC LIMIT %s",
                    (limit,),
                )
                cols = [d.name for d in cur.description]
                rows: list[dict] = []
                for r in cur.fetchall():
                    d = dict(zip(cols, r))
                    fj = d.get("forecast_json")
                    if isinstance(fj, str):
                        d["forecast"] = json.loads(fj)
                    else:
                        d["forecast"] = fj
                    rows.append(d)
                return rows

    def fetch_latest_prediction(self) -> dict | None:
        rows = self.fetch_recent_predictions(limit=1)
        return rows[0] if rows else None

    # --- commands ---
    def create_command(
        self,
        actuator: str,
        action: str,
        channel: int | None = None,
        value: float | None = None,
    ) -> int:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO commands (created_at, actuator, channel, action, value, status) "
                    "VALUES (%s, %s, %s, %s, %s, 'pending') RETURNING id",
                    (time.time(), actuator, channel, action, value),
                )
                cmd_id = cur.fetchone()[0]
            conn.commit()
            return int(cmd_id)

    def fetch_pending_commands(self) -> list:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM commands WHERE status='pending' ORDER BY created_at ASC"
                )
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]

    def update_command_status(
        self,
        cmd_id: int,
        status: str,
        error_msg: str | None = None,
    ) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE commands SET status=%s, applied_at=%s, error_msg=%s WHERE id=%s",
                    (status, time.time(), error_msg, cmd_id),
                )
            conn.commit()

    # --- events ---
    def insert_event(
        self,
        event_type: str,
        message: str,
        source: str | None = None,
        value: float | None = None,
    ) -> None:
        ts = time.time()
        ts_iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO events (ts_unix, ts_iso, event_type, source, message, value) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (ts, ts_iso, event_type, source, message, value),
                )
            conn.commit()

    def fetch_events(
        self,
        since_ts_unix: float | None = None,
        event_type: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        clauses: list[str] = []
        args: list = []
        if event_type is not None:
            clauses.append("event_type = %s")
            args.append(event_type)
        if since_ts_unix is not None:
            clauses.append("ts_unix >= %s")
            args.append(since_ts_unix)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        args.append(limit)
        sql = f"SELECT * FROM events {where} ORDER BY ts_unix DESC LIMIT %s"
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]

    # --- proofs ---
    def insert_proof(
        self,
        *,
        idempotency_key: str,
        digest_ts_unix: float,
        payload_blake2b: str,
        pinata_cid: str | None = None,
        hedera_topic_id: str | None = None,
        hedera_consensus_ts: str | None = None,
        hedera_sequence_number: int | None = None,
    ) -> int:
        consensus_ts_dt: datetime | None = None
        if hedera_consensus_ts is not None:
            try:
                consensus_ts_dt = datetime.fromtimestamp(float(hedera_consensus_ts), tz=timezone.utc)
            except (TypeError, ValueError):
                consensus_ts_dt = None
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO proofs (idempotency_key, digest_ts_unix, pinata_cid, "
                    "hedera_topic_id, hedera_consensus_ts, hedera_sequence_number, "
                    "payload_blake2b) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                    (
                        idempotency_key,
                        digest_ts_unix,
                        pinata_cid,
                        hedera_topic_id,
                        consensus_ts_dt,
                        hedera_sequence_number,
                        payload_blake2b,
                    ),
                )
                proof_id = cur.fetchone()[0]
            conn.commit()
            return int(proof_id)

    def fetch_proof_by_idempotency_key(self, idempotency_key: str) -> dict | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM proofs WHERE idempotency_key = %s",
                    (idempotency_key,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [d.name for d in cur.description]
                return dict(zip(cols, row))

    def fetch_proofs(
        self,
        *,
        since_ts_unix: float | None = None,
        limit: int = 200,
    ) -> list[dict]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                if since_ts_unix is None:
                    cur.execute(
                        "SELECT * FROM proofs ORDER BY digest_ts_unix DESC LIMIT %s",
                        (limit,),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM proofs WHERE digest_ts_unix >= %s "
                        "ORDER BY digest_ts_unix DESC LIMIT %s",
                        (since_ts_unix, limit),
                    )
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]

    # --- maintenance ---
    def cleanup_old_readings(self, retention_days: int = 7) -> int:
        cutoff = time.time() - retention_days * 86400
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM readings WHERE ts_unix < %s", (cutoff,))
                deleted = cur.rowcount
            conn.commit()
            return int(deleted)
