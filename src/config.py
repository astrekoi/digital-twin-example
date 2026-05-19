"""Pydantic models for every project YAML config and loader utilities."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from src.env import (
    get_env_bool,
    get_env_int,
    get_env_path,
    get_env_str,
    load_project_env,
)

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for all entry points."""
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Collector config
# ---------------------------------------------------------------------------


class SensorBME280Config(BaseModel):
    enabled: bool = True
    i2c_address: int = 0x76
    # "auto": read chip_id (0x60=BME280, 0x58=BMP280) and pick the driver.
    # chip_id=0x58 (BMP280): humidity_pct will be None in readings.
    kind: Literal["auto", "bme280", "bmp280"] = "auto"


class SensorBH1750Config(BaseModel):
    enabled: bool = True
    i2c_address: int = 0x23


class SensorINA219Config(BaseModel):
    enabled: bool = True
    # A0/A1=GND -> 0x40. NOT 0x41.
    i2c_address: int = 0x40


class SensorDS18B20Config(BaseModel):
    enabled: bool = True


class SensorMQ2Config(BaseModel):
    enabled: bool = True
    # Current stand uses Troyka HAT Analog IO jumper 5V->V for MQ-2.
    # 3V3 remains supported only as an explicit alternate hardware mode.
    expected_vcc: Literal["3V3", "5V"] = "5V"


class SensorPIRConfig(BaseModel):
    enabled: bool = True
    # WP 4 = BCM 23 (the Amperka PDF has a typo - BCM 27 is listed twice).
    bcm_pin: int = 23


class SensorsConfig(BaseModel):
    bme280: SensorBME280Config = Field(default_factory=SensorBME280Config)
    bh1750: SensorBH1750Config = Field(default_factory=SensorBH1750Config)
    ina219: SensorINA219Config = Field(default_factory=SensorINA219Config)
    ds18b20: SensorDS18B20Config = Field(default_factory=SensorDS18B20Config)
    mq2: SensorMQ2Config = Field(default_factory=SensorMQ2Config)
    pir: SensorPIRConfig = Field(default_factory=SensorPIRConfig)


class ActuatorRelayConfig(BaseModel):
    enabled: bool = True
    # IN1..IN4 in channel order 0-3. HIGH Trigger (SRD-05VDC-SL-C).
    # BCM 7 and BCM 8 are forbidden (hardware pull-up -> relay latch).
    bcm_pins: list[int] = Field(default=[25, 24, 12, 16])


class ActuatorServoConfig(BaseModel):
    # Disabled by default: requires the Analog IO jumper in 5V->V position.
    # Do not use simultaneously with MQ-2 through the same Analog IO jumper.
    enabled: bool = False
    expected_vcc: Literal["3V3", "5V"] = "5V"
    analog_channels: list[int] = Field(default=[1, 2])  # A1, A2 on HAT


class ActuatorLEDConfig(BaseModel):
    enabled: bool = True
    bcm_red: int = 22    # WP 3
    bcm_yellow: int = 27  # WP 2
    bcm_green: int = 17  # WP 0


class ActuatorsConfig(BaseModel):
    relay: ActuatorRelayConfig = Field(default_factory=ActuatorRelayConfig)
    servo: ActuatorServoConfig = Field(default_factory=ActuatorServoConfig)
    leds: ActuatorLEDConfig = Field(default_factory=ActuatorLEDConfig)


class CollectorConfig(BaseModel):
    sample_period_s: float = 1.0
    db_path: Path = Path("data/telemetry.db")
    csv_dir: Path = Path("data")
    csv_prefix: str = "sensors"
    retention_days: int = 7
    # How often to re-read runtime.yml to apply new thresholds.
    config_reload_interval_s: int = 60
    runtime_config_path: Path = Path("configs/runtime.yml")
    sensors: SensorsConfig = Field(default_factory=SensorsConfig)
    actuators: ActuatorsConfig = Field(default_factory=ActuatorsConfig)


# ---------------------------------------------------------------------------
# Environment-driven config overlay
# ---------------------------------------------------------------------------


class AppEnvConfig(BaseModel):
    app_env: str = "development"
    log_level: str = "INFO"
    collector_config: Path = Path("configs/collector.yml")
    runtime_config: Path = Path("configs/runtime.yml")


class DatabaseBackendConfig(BaseModel):
    backend: Literal["sqlite", "postgres"] = "sqlite"
    sqlite_db_path: Path = Path("data/telemetry.db")
    postgres_dsn: str | None = None


class IntegrationConfig(BaseModel):
    ipfs_enabled: bool = False
    ipfs_provider: Literal["none", "pinata", "web3_storage", "local"] = "none"
    pinata_jwt: str | None = None
    pinata_api_base: str = "https://api.pinata.cloud"
    ipfs_gateway_base: str = "https://gateway.pinata.cloud/ipfs"
    web3_storage_token: str | None = None
    local_ipfs_api_url: str = "http://127.0.0.1:5001"
    allow_external_api_calls: bool = False


class HederaConfig(BaseModel):
    """Hedera Consensus Service config (replacement for the removed IOTA layer)."""
    enabled: bool = False
    network: Literal["testnet", "mainnet", "previewnet"] = "testnet"
    operator_id: str | None = None
    operator_key: str | None = None       # ECDSA secp256k1 hex (no 0x), or DER
    topic_id: str | None = None
    tx_fee_hbar: int = 2
    mirror_node_url: str = "https://testnet.mirrornode.hedera.com"


class CeleryConfig(BaseModel):
    redis_url: str = "redis://127.0.0.1:6379/0"
    timezone: str = "Europe/Moscow"
    sample_period_s: float = 1.0           # APScheduler in hw-worker
    apply_commands_period_s: float = 1.0   # APScheduler in hw-worker
    predict_period_s: int = 60             # Celery beat
    digest_period_s: int = 300             # Celery beat
    retention_period_s: int = 3600         # Celery beat
    retention_days: int = 7


class TaipyConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 5000
    dark_mode: bool = True
    live_refresh_s: float = 1.0
    charts_refresh_s: float = 5.0


class AIEnvConfig(BaseModel):
    enabled: bool = False
    models_dir: Path = Path("models")
    model_id: str | None = None
    predictor_interval_s: int = 60
    recommender_enabled: bool = False
    recommender_dry_run: bool = True


class SafetyEnvConfig(BaseModel):
    allow_hardware_runtime: bool = False


class ProjectEnvConfig(BaseModel):
    app: AppEnvConfig = Field(default_factory=AppEnvConfig)
    database: DatabaseBackendConfig = Field(default_factory=DatabaseBackendConfig)
    integrations: IntegrationConfig = Field(default_factory=IntegrationConfig)
    hedera: HederaConfig = Field(default_factory=HederaConfig)
    celery: CeleryConfig = Field(default_factory=CeleryConfig)
    taipy: TaipyConfig = Field(default_factory=TaipyConfig)
    ai: AIEnvConfig = Field(default_factory=AIEnvConfig)
    safety: SafetyEnvConfig = Field(default_factory=SafetyEnvConfig)


def load_env_config(dotenv_path: Path | str = ".env") -> ProjectEnvConfig:
    """Load optional .env values into typed config models.

    This function is explicit by design: importing src.config does not read .env.
    Missing python-dotenv is acceptable; os.environ still works.
    """
    load_project_env(dotenv_path)
    return ProjectEnvConfig(
        app=AppEnvConfig(
            app_env=get_env_str("APP_ENV", "development") or "development",
            log_level=get_env_str("LOG_LEVEL", "INFO") or "INFO",
            collector_config=get_env_path("COLLECTOR_CONFIG", "configs/collector.yml")
            or Path("configs/collector.yml"),
            runtime_config=get_env_path("RUNTIME_CONFIG", "configs/runtime.yml")
            or Path("configs/runtime.yml"),
        ),
        database=DatabaseBackendConfig(
            backend=get_env_str("DB_BACKEND", "sqlite") or "sqlite",
            sqlite_db_path=get_env_path("SQLITE_DB_PATH", "data/telemetry.db")
            or Path("data/telemetry.db"),
            postgres_dsn=get_env_str("POSTGRES_DSN"),
        ),
        integrations=IntegrationConfig(
            ipfs_enabled=get_env_bool("IPFS_ENABLED", False),
            ipfs_provider=get_env_str("IPFS_PROVIDER", "none") or "none",  # type: ignore[arg-type]
            pinata_jwt=get_env_str("PINATA_JWT"),
            pinata_api_base=get_env_str("PINATA_BASE_URL", "https://api.pinata.cloud")
            or "https://api.pinata.cloud",
            ipfs_gateway_base=get_env_str(
                "IPFS_GATEWAY_BASE",
                "https://gateway.pinata.cloud/ipfs",
            )
            or "https://gateway.pinata.cloud/ipfs",
            web3_storage_token=get_env_str("WEB3_STORAGE_TOKEN"),
            local_ipfs_api_url=get_env_str("LOCAL_IPFS_API_URL", "http://127.0.0.1:5001")
            or "http://127.0.0.1:5001",
            allow_external_api_calls=get_env_bool("ALLOW_EXTERNAL_API_CALLS", False),
        ),
        hedera=HederaConfig(
            enabled=get_env_bool("HEDERA_ENABLED", False),
            network=get_env_str("HEDERA_NETWORK", "testnet") or "testnet",  # type: ignore[arg-type]
            operator_id=get_env_str("HEDERA_OPERATOR_ID"),
            operator_key=get_env_str("HEDERA_OPERATOR_KEY"),
            topic_id=get_env_str("HEDERA_TOPIC_ID"),
            tx_fee_hbar=get_env_int("HEDERA_TX_FEE_HBAR", 2) or 2,
            mirror_node_url=get_env_str("HEDERA_MIRROR_NODE_URL", "https://testnet.mirrornode.hedera.com")
            or "https://testnet.mirrornode.hedera.com",
        ),
        celery=CeleryConfig(
            redis_url=get_env_str("REDIS_URL", "redis://127.0.0.1:6379/0") or "redis://127.0.0.1:6379/0",
            timezone=get_env_str("CELERY_TIMEZONE", "Europe/Moscow") or "Europe/Moscow",
            sample_period_s=float(get_env_str("SAMPLE_PERIOD_S", "1.0") or "1.0"),
            apply_commands_period_s=float(get_env_str("APPLY_COMMANDS_PERIOD_S", "1.0") or "1.0"),
            predict_period_s=get_env_int("PREDICT_PERIOD_S", 60) or 60,
            digest_period_s=get_env_int("DIGEST_PERIOD_S", 300) or 300,
            retention_period_s=get_env_int("RETENTION_PERIOD_S", 3600) or 3600,
            retention_days=get_env_int("RETENTION_DAYS", 7) or 7,
        ),
        taipy=TaipyConfig(
            host=get_env_str("TAIPY_HOST", "0.0.0.0") or "0.0.0.0",
            port=get_env_int("TAIPY_PORT", 5000) or 5000,
            dark_mode=get_env_bool("TAIPY_DARK_MODE", True),
            live_refresh_s=float(get_env_str("TAIPY_LIVE_REFRESH_S", "1.0") or "1.0"),
            charts_refresh_s=float(get_env_str("TAIPY_CHARTS_REFRESH_S", "5.0") or "5.0"),
        ),
        ai=AIEnvConfig(
            enabled=get_env_bool("AI_ENABLED", False),
            models_dir=get_env_path("MODELS_DIR", "models") or Path("models"),
            model_id=get_env_str("MODEL_ID"),
            predictor_interval_s=get_env_int("PREDICT_PERIOD_S", 60) or 60,
            recommender_enabled=get_env_bool("AI_RECOMMENDER_ENABLED", False),
            recommender_dry_run=get_env_bool("AI_RECOMMENDER_DRY_RUN", True),
        ),
        safety=SafetyEnvConfig(
            allow_hardware_runtime=get_env_bool("ALLOW_HARDWARE_RUNTIME", False),
        ),
    )


# ---------------------------------------------------------------------------
# Runtime config (edited from the UI, applied without restart)
# ---------------------------------------------------------------------------


class ThresholdsConfig(BaseModel):
    temp_c_max: float = 40.0
    temp_c_min: float = 10.0
    humidity_pct_max: float = 90.0
    mq2_raw_warn: float = 0.4
    mq2_raw_alert: float = 0.7
    current_ma_max: float = 300.0
    lux_min: float = 10.0


class AIRuntimeConfig(BaseModel):
    enabled: bool = False
    model_id: str | None = None
    horizon_min: int = 15
    models_dir: Path = Path("models")


class RuntimeConfig(BaseModel):
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    fan_auto: bool = False
    fan_temp_threshold_c: float = 35.0
    ai: AIRuntimeConfig = Field(default_factory=AIRuntimeConfig)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        logger.warning("Config %s not found, using defaults", path)
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _env_is_set(name: str) -> bool:
    return name in os.environ and os.environ[name] != ""


def _apply_env_to_collector(config: CollectorConfig) -> CollectorConfig:
    env_config = load_env_config()
    if _env_is_set("SQLITE_DB_PATH") and env_config.database.backend == "sqlite":
        config.db_path = env_config.database.sqlite_db_path
    if _env_is_set("RUNTIME_CONFIG"):
        config.runtime_config_path = env_config.app.runtime_config
    return config


def _apply_env_to_runtime(config: RuntimeConfig) -> RuntimeConfig:
    env_config = load_env_config()
    if _env_is_set("AI_ENABLED"):
        config.ai.enabled = env_config.ai.enabled
    if _env_is_set("MODEL_ID"):
        config.ai.model_id = env_config.ai.model_id
    if _env_is_set("MODELS_DIR"):
        config.ai.models_dir = env_config.ai.models_dir
    return config


def load_collector_config(path: Path) -> CollectorConfig:
    """Load and validate configs/collector.yml."""
    return _apply_env_to_collector(CollectorConfig(**_load_yaml(path)))


def load_runtime_config(path: Path) -> RuntimeConfig:
    """Load and validate configs/runtime.yml (hot-reloadable)."""
    return _apply_env_to_runtime(RuntimeConfig(**_load_yaml(path)))
