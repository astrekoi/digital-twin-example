"""SQLite layer: schema and helpers for collector, predictor, dashboard.

Three processes share one DB file in WAL mode:
  - collector: writes readings + events, reads/updates commands
  - predictor: reads readings, writes predictions
  - dashboard: reads everything, writes commands
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS readings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_iso        TEXT    NOT NULL,
    ts_unix       REAL    NOT NULL,
    temp_c        REAL,
    humidity_pct  REAL,
    pressure_hpa  REAL,
    lux           REAL,
    current_ma    REAL,
    bus_voltage   REAL,
    shunt_mv      REAL,
    power_mw      REAL,
    ds18b20_c     REAL,
    mq2_raw       REAL,
    mq2_voltage   REAL,
    pir_state     INTEGER
);

CREATE INDEX IF NOT EXISTS idx_readings_ts_unix
    ON readings(ts_unix);

CREATE TABLE IF NOT EXISTS commands (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL    NOT NULL,
    actuator    TEXT    NOT NULL,
    channel     INTEGER,
    action      TEXT    NOT NULL,   -- on|off|set_angle|set_level|update_runtime_config
    value       REAL,
    params      TEXT,               -- JSON payload for structured commands
    status      TEXT    NOT NULL DEFAULT 'pending',  -- pending|applied|failed
    applied_at  REAL,
    error_msg   TEXT
);

-- Composite index: collector queries pending rows ordered by created_at.
CREATE INDEX IF NOT EXISTS idx_commands_status_created
    ON commands(status, created_at);

CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at   REAL    NOT NULL,
    model_id       TEXT    NOT NULL,
    horizon_min    INTEGER NOT NULL,
    forecast_json  TEXT    NOT NULL,  -- JSON: [{ts_unix, value}, ...]
    features_json  TEXT               -- JSON snapshot of input features (for debugging)
);

CREATE INDEX IF NOT EXISTS idx_predictions_generated_at
    ON predictions(generated_at);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_unix     REAL    NOT NULL,
    ts_iso      TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,  -- pir|threshold|sensor_error|command
    source      TEXT,
    message     TEXT    NOT NULL,
    value       REAL
);

CREATE INDEX IF NOT EXISTS idx_events_ts_unix
    ON events(ts_unix);

CREATE TABLE IF NOT EXISTS proofs (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key        TEXT    UNIQUE NOT NULL,
    digest_ts_unix         REAL    NOT NULL,
    pinata_cid             TEXT,
    hedera_topic_id        TEXT,
    hedera_consensus_ts    TEXT,
    hedera_sequence_number INTEGER,
    payload_blake2b        TEXT    NOT NULL,
    created_at             REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_proofs_digest_ts
    ON proofs(digest_ts_unix);
"""

# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------


def open_connection(db_path: Path) -> sqlite3.Connection:
    """Open a WAL-mode connection. Each process opens its own."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL allows one writer concurrent with many readers.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA cache_size=-8000")  # 8 MB page cache
    return conn


def _migrate_commands_params(conn: sqlite3.Connection) -> None:
    """Add params TEXT column to commands table if it doesn't exist yet."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(commands)")}
    if "params" not in existing:
        conn.execute("ALTER TABLE commands ADD COLUMN params TEXT")
        conn.commit()


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create the DB (if absent), apply the schema, return a connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = open_connection(db_path)
    conn.executescript(_SCHEMA_DDL)
    _migrate_commands_params(conn)
    logger.info("DB initialised: %s", db_path)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Connection, None, None]:
    """Transaction context manager that rolls back on exception."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ---------------------------------------------------------------------------
# Collector: write readings
# ---------------------------------------------------------------------------


def insert_reading(conn: sqlite3.Connection, row: dict) -> None:
    """Insert a single reading row. None in row -> NULL in DB."""
    conn.execute(
        """
        INSERT INTO readings (
            ts_iso, ts_unix,
            temp_c, humidity_pct, pressure_hpa,
            lux,
            current_ma, bus_voltage, shunt_mv, power_mw,
            ds18b20_c,
            mq2_raw, mq2_voltage,
            pir_state
        ) VALUES (
            :ts_iso, :ts_unix,
            :temp_c, :humidity_pct, :pressure_hpa,
            :lux,
            :current_ma, :bus_voltage, :shunt_mv, :power_mw,
            :ds18b20_c,
            :mq2_raw, :mq2_voltage,
            :pir_state
        )
        """,
        row,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Collector: read and execute commands
# ---------------------------------------------------------------------------


def fetch_pending_commands(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return pending commands ordered by created_at (uses the index)."""
    cur = conn.execute(
        "SELECT * FROM commands WHERE status = 'pending' ORDER BY created_at ASC"
    )
    return cur.fetchall()


def update_command_status(
    conn: sqlite3.Connection,
    cmd_id: int,
    status: str,
    error_msg: str | None = None,
) -> None:
    """Mark a command as applied or failed."""
    conn.execute(
        """
        UPDATE commands
        SET status = ?, applied_at = ?, error_msg = ?
        WHERE id = ?
        """,
        (status, time.time(), error_msg, cmd_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Dashboard: write commands
# ---------------------------------------------------------------------------


def insert_command(
    conn: sqlite3.Connection,
    actuator: str,
    action: str,
    channel: int | None = None,
    value: float | None = None,
) -> int:
    """Write a pending command, return its id."""
    cur = conn.execute(
        """
        INSERT INTO commands (created_at, actuator, channel, action, value)
        VALUES (?, ?, ?, ?, ?)
        """,
        (time.time(), actuator, channel, action, value),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def insert_runtime_config_command(
    conn: sqlite3.Connection,
    params: dict,
) -> int:
    """Write an update_runtime_config command with a JSON params payload.

    Dashboard calls this; collector processes it and writes to runtime.yml.
    params: dict of config changes, e.g. {"thresholds": {"temp_c_max": 45.0}}.
    """
    cur = conn.execute(
        """
        INSERT INTO commands (created_at, actuator, channel, action, value, params)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (time.time(), "runtime_config", None, "update_runtime_config", None,
         json.dumps(params)),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def fetch_recent_commands(
    conn: sqlite3.Connection, limit: int = 50
) -> list[sqlite3.Row]:
    cur = conn.execute(
        "SELECT * FROM commands ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    return cur.fetchall()


def delete_applied_commands(conn: sqlite3.Connection) -> int:
    """Delete commands with status 'applied' or 'failed'. Pending commands are kept.

    Returns the number of rows deleted.
    """
    cur = conn.execute(
        "DELETE FROM commands WHERE status IN ('applied', 'failed')"
    )
    conn.commit()
    return cur.rowcount


# ---------------------------------------------------------------------------
# Collector: events
# ---------------------------------------------------------------------------


def insert_event(
    conn: sqlite3.Connection,
    event_type: str,
    message: str,
    source: str | None = None,
    value: float | None = None,
) -> None:
    from datetime import datetime, timezone

    now = time.time()
    ts_iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO events (ts_unix, ts_iso, event_type, source, message, value)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (now, ts_iso, event_type, source, message, value),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Dashboard: read readings
# ---------------------------------------------------------------------------


def fetch_latest_reading(conn: sqlite3.Connection) -> dict | None:
    cur = conn.execute(
        "SELECT * FROM readings ORDER BY ts_unix DESC LIMIT 1"
    )
    row = cur.fetchone()
    return dict(row) if row else None


def fetch_readings_range(
    conn: sqlite3.Connection,
    from_unix: float,
    to_unix: float,
    step_s: float = 0,
) -> list[dict]:
    """Return rows in the range [from_unix, to_unix].

    step_s > 0: aggregate by windows (MIN id per window) for downsampling.
    Uses the index idx_readings_ts_unix.
    """
    if step_s <= 0:
        cur = conn.execute(
            "SELECT * FROM readings WHERE ts_unix BETWEEN ? AND ? ORDER BY ts_unix ASC",
            (from_unix, to_unix),
        )
        return [dict(r) for r in cur.fetchall()]

    # Downsampling: take the first row in each step_s window.
    cur = conn.execute(
        """
        SELECT * FROM readings
        WHERE ts_unix BETWEEN ? AND ?
          AND id IN (
              SELECT MIN(id) FROM readings
              WHERE ts_unix BETWEEN ? AND ?
              GROUP BY CAST((ts_unix - ?) / ? AS INTEGER)
          )
        ORDER BY ts_unix ASC
        """,
        (from_unix, to_unix, from_unix, to_unix, from_unix, step_s),
    )
    return [dict(r) for r in cur.fetchall()]


def fetch_readings_last_n_minutes(
    conn: sqlite3.Connection, minutes: int
) -> list[dict]:
    from_unix = time.time() - minutes * 60
    cur = conn.execute(
        "SELECT * FROM readings WHERE ts_unix >= ? ORDER BY ts_unix ASC",
        (from_unix,),
    )
    return [dict(r) for r in cur.fetchall()]


def fetch_recent_readings(
    conn: sqlite3.Connection,
    limit: int = 100,
) -> list[dict]:
    """Return the latest N readings in chronological order.

    The query fetches newest rows first for index-friendly limiting, then
    reverses them so chart/table consumers receive oldest -> newest rows.
    """
    safe_limit = max(1, min(int(limit), 5000))
    cur = conn.execute(
        "SELECT * FROM readings ORDER BY ts_unix DESC LIMIT ?",
        (safe_limit,),
    )
    return [dict(r) for r in reversed(cur.fetchall())]


def fetch_sensor_stats(
    conn: sqlite3.Connection,
    column: str,
    from_unix: float,
) -> dict:
    """min/max/avg/count over a period. `column` must be a readings column."""
    # The column name is constructed internally, not from user input - no
    # SQL injection vector.
    allowed = {
        "temp_c", "humidity_pct", "pressure_hpa", "lux",
        "current_ma", "bus_voltage", "shunt_mv", "power_mw",
        "ds18b20_c", "mq2_raw", "mq2_voltage",
    }
    if column not in allowed:
        raise ValueError(f"Unknown column: {column}")
    cur = conn.execute(
        f"""
        SELECT
            MIN({column})  AS min_val,
            MAX({column})  AS max_val,
            AVG({column})  AS avg_val,
            COUNT({column}) AS count_val
        FROM readings
        WHERE ts_unix >= ? AND {column} IS NOT NULL
        """,
        (from_unix,),
    )
    row = cur.fetchone()
    return dict(row) if row else {}


# ---------------------------------------------------------------------------
# Dashboard: events
# ---------------------------------------------------------------------------


def fetch_events(
    conn: sqlite3.Connection,
    limit: int = 200,
    event_type: str | None = None,
    from_unix: float | None = None,
) -> list[dict]:
    if event_type and from_unix:
        cur = conn.execute(
            "SELECT * FROM events WHERE event_type = ? AND ts_unix >= ? ORDER BY ts_unix DESC LIMIT ?",
            (event_type, from_unix, limit),
        )
    elif event_type:
        cur = conn.execute(
            "SELECT * FROM events WHERE event_type = ? ORDER BY ts_unix DESC LIMIT ?",
            (event_type, limit),
        )
    elif from_unix:
        cur = conn.execute(
            "SELECT * FROM events WHERE ts_unix >= ? ORDER BY ts_unix DESC LIMIT ?",
            (from_unix, limit),
        )
    else:
        cur = conn.execute(
            "SELECT * FROM events ORDER BY ts_unix DESC LIMIT ?", (limit,)
        )
    return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Predictor: read and write predictions
# ---------------------------------------------------------------------------


def insert_prediction(
    conn: sqlite3.Connection,
    model_id: str,
    horizon_min: int,
    forecast: list[dict],
    features: dict | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO predictions (generated_at, model_id, horizon_min, forecast_json, features_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            time.time(),
            model_id,
            horizon_min,
            json.dumps(forecast),
            json.dumps(features) if features else None,
        ),
    )
    conn.commit()


def fetch_latest_prediction(
    conn: sqlite3.Connection,
    model_id: str | None = None,
    horizon_min: int | None = None,
) -> dict | None:
    """Return the latest prediction (optional filter by model_id/horizon_min)."""
    if model_id and horizon_min:
        cur = conn.execute(
            """SELECT * FROM predictions
               WHERE model_id = ? AND horizon_min = ?
               ORDER BY generated_at DESC LIMIT 1""",
            (model_id, horizon_min),
        )
    elif model_id:
        cur = conn.execute(
            "SELECT * FROM predictions WHERE model_id = ? ORDER BY generated_at DESC LIMIT 1",
            (model_id,),
        )
    else:
        cur = conn.execute(
            "SELECT * FROM predictions ORDER BY generated_at DESC LIMIT 1"
        )
    row = cur.fetchone()
    if not row:
        return None
    result = dict(row)
    result["forecast"] = json.loads(result["forecast_json"])
    return result


def fetch_recent_predictions(
    conn: sqlite3.Connection,
    limit: int = 20,
) -> list[dict]:
    """Return the most recent predictions (used by the storage abstraction and dashboard)."""
    cur = conn.execute(
        "SELECT * FROM predictions ORDER BY generated_at DESC LIMIT ?",
        (limit,),
    )
    rows = []
    for row in cur.fetchall():
        result = dict(row)
        result["forecast"] = json.loads(result["forecast_json"])
        rows.append(result)
    return rows


# ---------------------------------------------------------------------------
# Proofs (Pinata CID + Hedera consensus receipt) - idempotent publish layer
# ---------------------------------------------------------------------------


def insert_proof(
    conn: sqlite3.Connection,
    *,
    idempotency_key: str,
    digest_ts_unix: float,
    payload_blake2b: str,
    pinata_cid: str | None = None,
    hedera_topic_id: str | None = None,
    hedera_consensus_ts: str | None = None,
    hedera_sequence_number: int | None = None,
) -> int:
    """Insert a proof row. UNIQUE on idempotency_key - IntegrityError on duplicate."""
    cur = conn.execute(
        """
        INSERT INTO proofs (
            idempotency_key, digest_ts_unix, pinata_cid,
            hedera_topic_id, hedera_consensus_ts, hedera_sequence_number,
            payload_blake2b, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            idempotency_key,
            digest_ts_unix,
            pinata_cid,
            hedera_topic_id,
            hedera_consensus_ts,
            hedera_sequence_number,
            payload_blake2b,
            time.time(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid or 0)


def fetch_proof_by_idempotency_key(
    conn: sqlite3.Connection,
    idempotency_key: str,
) -> dict | None:
    cur = conn.execute(
        "SELECT * FROM proofs WHERE idempotency_key = ?",
        (idempotency_key,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def fetch_proofs(
    conn: sqlite3.Connection,
    *,
    since_ts_unix: float | None = None,
    limit: int = 200,
) -> list[dict]:
    if since_ts_unix is None:
        cur = conn.execute(
            "SELECT * FROM proofs ORDER BY digest_ts_unix DESC LIMIT ?",
            (limit,),
        )
    else:
        cur = conn.execute(
            "SELECT * FROM proofs WHERE digest_ts_unix >= ? "
            "ORDER BY digest_ts_unix DESC LIMIT ?",
            (since_ts_unix, limit),
        )
    return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Maintenance: 7-day ring buffer + VACUUM
# ---------------------------------------------------------------------------


def cleanup_old_readings(conn: sqlite3.Connection, retention_days: int = 7) -> int:
    """Delete rows older than retention_days. Call hourly from collector."""
    cutoff = time.time() - retention_days * 86400
    cur = conn.execute("DELETE FROM readings WHERE ts_unix < ?", (cutoff,))
    conn.commit()
    deleted = cur.rowcount
    if deleted > 0:
        logger.info("Readings cleanup: deleted %d rows (cutoff=%s)", deleted, cutoff)
    return deleted


def vacuum(conn: sqlite3.Connection) -> None:
    """VACUUM - defragmentation after bulk DELETE. Call weekly.

    Requires no active transactions. Safe in WAL mode.
    For scheduled execution (e.g. 03:00) use a systemd timer:
    python -c "from src.db import *; ..."
    """
    logger.info("Starting VACUUM...")
    conn.execute("VACUUM")
    logger.info("VACUUM done")
