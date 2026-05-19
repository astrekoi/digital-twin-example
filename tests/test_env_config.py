"""Env-driven config loader."""
from __future__ import annotations

from src.config import load_env_config


def test_env_defaults(monkeypatch, tmp_path):
    fake_env = tmp_path / ".env"
    fake_env.write_text("APP_ENV=test\n")
    cfg = load_env_config(fake_env)
    assert cfg.app.app_env == "test"
    assert cfg.database.backend == "sqlite"
    assert cfg.celery.predict_period_s == 60
    assert cfg.celery.digest_period_s == 300
    assert cfg.celery.retention_period_s == 3600
    assert cfg.celery.sample_period_s == 1.0
    assert cfg.celery.apply_commands_period_s == 1.0
    assert cfg.taipy.port == 5000
    assert cfg.integrations.allow_external_api_calls is False
    assert cfg.hedera.enabled is False


def test_hedera_picks_up_env(monkeypatch, tmp_path):
    # _isolate_env autouse fixture sets HEDERA_ENABLED=false before this runs;
    # python-dotenv defaults to override=False, so os.environ wins. Override
    # via monkeypatch directly to test the loader logic.
    monkeypatch.setenv("HEDERA_ENABLED", "true")
    monkeypatch.setenv("HEDERA_NETWORK", "mainnet")
    monkeypatch.setenv("HEDERA_OPERATOR_ID", "0.0.1234")
    monkeypatch.setenv("HEDERA_OPERATOR_KEY", "deadbeef")
    monkeypatch.setenv("HEDERA_TOPIC_ID", "0.0.5678")
    fake_env = tmp_path / ".env"
    fake_env.write_text("APP_ENV=test\n")
    cfg = load_env_config(fake_env)
    assert cfg.hedera.enabled is True
    assert cfg.hedera.network == "mainnet"
    assert cfg.hedera.operator_id == "0.0.1234"
    assert cfg.hedera.topic_id == "0.0.5678"


def test_celery_periods_override(monkeypatch, tmp_path):
    fake_env = tmp_path / ".env"
    fake_env.write_text(
        "PREDICT_PERIOD_S=30\n"
        "DIGEST_PERIOD_S=120\n"
        "SAMPLE_PERIOD_S=2.5\n"
    )
    cfg = load_env_config(fake_env)
    assert cfg.celery.predict_period_s == 30
    assert cfg.celery.digest_period_s == 120
    assert cfg.celery.sample_period_s == 2.5
