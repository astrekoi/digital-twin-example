"""Storage backend contracts: telemetry, predictions, commands, events, proofs.

Two implementations live alongside this contract:
- SQLiteTelemetryStore (default; src.db delegate)
- PostgresTelemetryStore (opt-in via DB_BACKEND=postgres + POSTGRES_DSN)
"""
from __future__ import annotations

from typing import Protocol


class TelemetryStore(Protocol):
    """Contract shared by SQLite and Postgres adapters."""

    # --- lifecycle ---
    def init_schema(self) -> None: ...
    def close(self) -> None: ...

    # --- readings ---
    def insert_reading(self, row: dict) -> None: ...
    def fetch_latest_reading(self) -> dict | None: ...
    def fetch_recent_readings(self, minutes: int = 60) -> list[dict]: ...
    def fetch_readings_range(
        self, from_unix: float, to_unix: float, step_s: float = 0
    ) -> list[dict]: ...

    # --- predictions ---
    def insert_prediction(
        self,
        model_id: str,
        horizon_min: int,
        forecast: list[dict],
        features: dict | None = None,
    ) -> None: ...
    def fetch_recent_predictions(self, limit: int = 20) -> list[dict]: ...
    def fetch_latest_prediction(self) -> dict | None: ...

    # --- commands ---
    def create_command(
        self,
        actuator: str,
        action: str,
        channel: int | None = None,
        value: float | None = None,
    ) -> int: ...
    def fetch_pending_commands(self) -> list: ...
    def update_command_status(
        self,
        cmd_id: int,
        status: str,
        error_msg: str | None = None,
    ) -> None: ...

    # --- events ---
    def insert_event(
        self,
        event_type: str,
        message: str,
        source: str | None = None,
        value: float | None = None,
    ) -> None: ...
    def fetch_events(
        self,
        since_ts_unix: float | None = None,
        event_type: str | None = None,
        limit: int = 200,
    ) -> list[dict]: ...

    # --- proofs (idempotent publish layer for Pinata + Hedera) ---
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
    ) -> int: ...
    def fetch_proof_by_idempotency_key(self, idempotency_key: str) -> dict | None: ...
    def fetch_proofs(
        self,
        *,
        since_ts_unix: float | None = None,
        limit: int = 200,
    ) -> list[dict]: ...

    # --- maintenance ---
    def cleanup_old_readings(self, retention_days: int = 7) -> int: ...
