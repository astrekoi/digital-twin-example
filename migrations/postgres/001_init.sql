-- migrations/postgres/001_init.sql
-- Idempotent. Run via psql or scripts/migrate_sqlite_to_postgres.py.
-- Mirrors src/db.py SQLite schema with PG-native types.

CREATE TABLE IF NOT EXISTS readings (
    id            BIGSERIAL PRIMARY KEY,
    ts_iso        TEXT             NOT NULL,
    ts_unix       DOUBLE PRECISION NOT NULL,
    temp_c        DOUBLE PRECISION,
    humidity_pct  DOUBLE PRECISION,
    pressure_hpa  DOUBLE PRECISION,
    lux           DOUBLE PRECISION,
    current_ma    DOUBLE PRECISION,
    bus_voltage   DOUBLE PRECISION,
    shunt_mv      DOUBLE PRECISION,
    power_mw      DOUBLE PRECISION,
    ds18b20_c     DOUBLE PRECISION,
    mq2_raw       DOUBLE PRECISION,
    mq2_voltage   DOUBLE PRECISION,
    pir_state     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_readings_ts_unix ON readings(ts_unix);

CREATE TABLE IF NOT EXISTS commands (
    id          BIGSERIAL PRIMARY KEY,
    created_at  DOUBLE PRECISION NOT NULL,
    actuator    TEXT             NOT NULL,
    channel     INTEGER,
    action      TEXT             NOT NULL,
    value       DOUBLE PRECISION,
    params      JSONB,
    status      TEXT             NOT NULL DEFAULT 'pending',
    applied_at  DOUBLE PRECISION,
    error_msg   TEXT
);
CREATE INDEX IF NOT EXISTS idx_commands_status_created ON commands(status, created_at);

CREATE TABLE IF NOT EXISTS predictions (
    id            BIGSERIAL PRIMARY KEY,
    generated_at  DOUBLE PRECISION NOT NULL,
    model_id      TEXT             NOT NULL,
    horizon_min   INTEGER          NOT NULL,
    forecast_json JSONB            NOT NULL,
    features_json JSONB
);
CREATE INDEX IF NOT EXISTS idx_predictions_generated_at ON predictions(generated_at);

CREATE TABLE IF NOT EXISTS events (
    id          BIGSERIAL PRIMARY KEY,
    ts_unix     DOUBLE PRECISION NOT NULL,
    ts_iso      TEXT             NOT NULL,
    event_type  TEXT             NOT NULL,
    source      TEXT,
    message     TEXT             NOT NULL,
    value       DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_events_ts_unix ON events(ts_unix);

CREATE TABLE IF NOT EXISTS proofs (
    id                     BIGSERIAL PRIMARY KEY,
    idempotency_key        TEXT             UNIQUE NOT NULL,
    digest_ts_unix         DOUBLE PRECISION NOT NULL,
    pinata_cid             TEXT,
    hedera_topic_id        TEXT,
    hedera_consensus_ts    TIMESTAMPTZ,
    hedera_sequence_number BIGINT,
    payload_blake2b        TEXT             NOT NULL,
    created_at             TIMESTAMPTZ      NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_proofs_digest_ts ON proofs(digest_ts_unix);
