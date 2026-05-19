"""SQLite storage backend - thin delegate over src.db helpers."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from src import db


class SQLiteTelemetryStore:
    """TelemetryStore implementation backed by SQLite WAL.

    init_schema() creates ALL tables (readings, commands, predictions, events,
    proofs) via src.db._SCHEMA_DDL - required for the publish layer to work
    on default SQLite installations.
    """

    def __init__(self, db_path: Path | str = "data/telemetry.db") -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = db.init_db(self.db_path)
        return self._conn

    def init_schema(self) -> None:
        if self._conn is None:
            self._conn = db.init_db(self.db_path)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # --- readings ---
    def insert_reading(self, row: dict) -> None:
        db.insert_reading(self.conn, row)

    def fetch_latest_reading(self) -> dict | None:
        return db.fetch_latest_reading(self.conn)

    def fetch_recent_readings(self, minutes: int = 60) -> list[dict]:
        return db.fetch_readings_last_n_minutes(self.conn, minutes)

    def fetch_readings_range(
        self, from_unix: float, to_unix: float, step_s: float = 0
    ) -> list[dict]:
        return db.fetch_readings_range(self.conn, from_unix, to_unix, step_s)

    # --- predictions ---
    def insert_prediction(
        self,
        model_id: str,
        horizon_min: int,
        forecast: list[dict],
        features: dict | None = None,
    ) -> None:
        db.insert_prediction(self.conn, model_id, horizon_min, forecast, features)

    def fetch_recent_predictions(self, limit: int = 20) -> list[dict]:
        return db.fetch_recent_predictions(self.conn, limit=limit)

    def fetch_latest_prediction(self) -> dict | None:
        return db.fetch_latest_prediction(self.conn)

    # --- commands ---
    def create_command(
        self,
        actuator: str,
        action: str,
        channel: int | None = None,
        value: float | None = None,
    ) -> int:
        return db.insert_command(self.conn, actuator, action, channel=channel, value=value)

    def fetch_pending_commands(self) -> list:
        return db.fetch_pending_commands(self.conn)

    def update_command_status(
        self,
        cmd_id: int,
        status: str,
        error_msg: str | None = None,
    ) -> None:
        db.update_command_status(self.conn, cmd_id, status, error_msg=error_msg)

    # --- events ---
    def insert_event(
        self,
        event_type: str,
        message: str,
        source: str | None = None,
        value: float | None = None,
    ) -> None:
        db.insert_event(self.conn, event_type, message, source=source, value=value)

    def fetch_events(
        self,
        since_ts_unix: float | None = None,
        event_type: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        return db.fetch_events(
            self.conn,
            limit=limit,
            event_type=event_type,
            from_unix=since_ts_unix,
        )

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
        return db.insert_proof(
            self.conn,
            idempotency_key=idempotency_key,
            digest_ts_unix=digest_ts_unix,
            payload_blake2b=payload_blake2b,
            pinata_cid=pinata_cid,
            hedera_topic_id=hedera_topic_id,
            hedera_consensus_ts=hedera_consensus_ts,
            hedera_sequence_number=hedera_sequence_number,
        )

    def fetch_proof_by_idempotency_key(self, idempotency_key: str) -> dict | None:
        return db.fetch_proof_by_idempotency_key(self.conn, idempotency_key)

    def fetch_proofs(
        self,
        *,
        since_ts_unix: float | None = None,
        limit: int = 200,
    ) -> list[dict]:
        return db.fetch_proofs(self.conn, since_ts_unix=since_ts_unix, limit=limit)

    # --- maintenance ---
    def cleanup_old_readings(self, retention_days: int = 7) -> int:
        return db.cleanup_old_readings(self.conn, retention_days=retention_days)
