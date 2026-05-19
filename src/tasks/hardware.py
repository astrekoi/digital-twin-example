"""Hardware-side Celery tasks + APScheduler-backed inline jobs.

This module is loaded only by hw-worker (`hostname=hw@%h`). All GPIO/I²C
ownership lives here as module-level singletons initialized once in
worker_init signal. APScheduler runs sample_sensors / apply_commands at
1 Hz inside the same process - no broker round-trip for hot-path work.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import load_collector_config, load_runtime_config
from src.env import load_project_env
from src.storage.factory import create_telemetry_store
from src.tasks.celery_app import celery_app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_project_env(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

# Module-level singletons - owned by the hw-worker process only.
_resources: dict[str, Any] = {
    "sensors": {},     # name -> sensor instance with .read() / .close()
    "actuators": {},   # name -> actuator instance with .close() (relays/leds/servo)
    "store": None,     # TelemetryStore
}
_scheduler: Any = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Lifecycle: init / cleanup
# ---------------------------------------------------------------------------


def init_hardware_resources(config_path: Path | None = None) -> None:
    """Initialize sensor/actuator singletons. Called from worker_init."""
    if _resources["sensors"]:
        logger.debug("hardware resources already initialized")
        return

    cfg = load_collector_config(config_path or PROJECT_ROOT / "configs" / "collector.yml")

    # storage
    store = create_telemetry_store()
    store.init_schema()
    _resources["store"] = store

    # sensors
    sensors = _resources["sensors"]
    if cfg.sensors.bme280.enabled:
        try:
            from src.sensors.bme280 import BME280Sensor
            # NB: param name is sensor_kind in BME280Sensor.__init__, NOT kind.
            sensors["bme280"] = BME280Sensor(
                address=cfg.sensors.bme280.i2c_address,
                sensor_kind=cfg.sensors.bme280.kind,
            )
            _emit_sensor_init_event(sensors["bme280"], "bme280", store)
        except Exception:
            logger.exception("bme280 init failed")
            _safe_emit_event(store, "sensor_error", "bme280 init failed", source="bme280")
    if cfg.sensors.bh1750.enabled:
        try:
            from src.sensors.bh1750 import BH1750Sensor
            sensors["bh1750"] = BH1750Sensor(address=cfg.sensors.bh1750.i2c_address)
            _safe_emit_event(store, "sensor_init", "BH1750 ready (lux)", source="bh1750")
        except Exception:
            logger.exception("bh1750 init failed")
            _safe_emit_event(store, "sensor_error", "bh1750 init failed", source="bh1750")
    if cfg.sensors.ina219.enabled:
        try:
            from src.sensors.ina219 import INA219Sensor
            sensors["ina219"] = INA219Sensor(address=cfg.sensors.ina219.i2c_address)
            _safe_emit_event(store, "sensor_init", "INA219 ready (current/voltage/power)", source="ina219")
        except Exception:
            logger.exception("ina219 init failed")
            _safe_emit_event(store, "sensor_error", "ina219 init failed", source="ina219")
    if cfg.sensors.ds18b20.enabled:
        try:
            from src.sensors.ds18b20 import DS18B20Sensor
            sensors["ds18b20"] = DS18B20Sensor()
            _safe_emit_event(store, "sensor_init", "DS18B20 ready (1-Wire temperature)", source="ds18b20")
        except Exception:
            logger.exception("ds18b20 init failed")
            _safe_emit_event(store, "sensor_error", "ds18b20 init failed", source="ds18b20")
    if cfg.sensors.mq2.enabled:
        try:
            from src.sensors.mq2 import MQ2Sensor
            sensors["mq2"] = MQ2Sensor(expected_vcc=cfg.sensors.mq2.expected_vcc)
            _safe_emit_event(
                store, "sensor_init",
                f"MQ-2 ready (Vcc={cfg.sensors.mq2.expected_vcc})",
                source="mq2",
            )
        except Exception:
            logger.exception("mq2 init failed")
            _safe_emit_event(store, "sensor_error", "mq2 init failed", source="mq2")
    if cfg.sensors.pir.enabled:
        try:
            from src.sensors.pir import PIRSensor
            sensors["pir"] = PIRSensor(bcm_pin=cfg.sensors.pir.bcm_pin)
            _safe_emit_event(
                store, "sensor_init",
                f"PIR ready (BCM {cfg.sensors.pir.bcm_pin})",
                source="pir",
            )
        except Exception:
            logger.exception("pir init failed")
            _safe_emit_event(store, "sensor_error", "pir init failed", source="pir")

    # actuators
    actuators = _resources["actuators"]
    if cfg.actuators.relay.enabled:
        try:
            from src.actuators.relay import Relay4Channel
            actuators["relay"] = Relay4Channel()
        except Exception:
            logger.exception("relay init failed")
    if cfg.actuators.servo.enabled:
        try:
            from src.actuators.servo import ServoDriver
            actuators["servo"] = ServoDriver(channels=cfg.actuators.servo.analog_channels)
        except Exception:
            logger.exception("servo init failed")
    if cfg.actuators.leds.enabled:
        try:
            from src.actuators.leds import TrafficLight
            actuators["leds"] = TrafficLight()
        except Exception:
            logger.exception("leds init failed")

    logger.info(
        "hw resources ready: sensors=%s actuators=%s",
        sorted(sensors.keys()),
        sorted(actuators.keys()),
    )


def cleanup_hardware_resources() -> None:
    """Close all sensor/actuator handles. Called from worker_shutdown."""
    for name, sensor in list(_resources["sensors"].items()):
        try:
            sensor.close()
        except Exception:
            logger.exception("error closing sensor %s", name)
    _resources["sensors"].clear()

    for name, act in list(_resources["actuators"].items()):
        try:
            if hasattr(act, "off_all"):
                act.off_all()
            if hasattr(act, "off"):
                try:
                    act.off()
                except Exception:
                    pass
            act.close()
        except Exception:
            logger.exception("error closing actuator %s", name)
    _resources["actuators"].clear()

    if _resources["store"] is not None:
        try:
            _resources["store"].close()
        except Exception:
            logger.exception("error closing store")
        _resources["store"] = None


# ---------------------------------------------------------------------------
# APScheduler - 1 Hz sample + 1 Hz apply_commands
# ---------------------------------------------------------------------------


def start_apscheduler(*, sample_period_s: float = 1.0, apply_commands_period_s: float = 1.0) -> None:
    """Start a BackgroundScheduler with two serialized 1-thread executor jobs."""
    global _scheduler
    if _scheduler is not None:
        logger.debug("apscheduler already running")
        return

    from apscheduler.executors.pool import ThreadPoolExecutor
    from apscheduler.schedulers.background import BackgroundScheduler

    _scheduler = BackgroundScheduler(
        executors={"default": ThreadPoolExecutor(max_workers=1)},
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 5},
        timezone="UTC",
    )
    _scheduler.add_job(
        sample_sensors_inline,
        "interval",
        seconds=sample_period_s,
        id="sample_sensors",
        replace_existing=True,
    )
    _scheduler.add_job(
        apply_pending_commands_inline,
        "interval",
        seconds=apply_commands_period_s,
        id="apply_commands",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info(
        "apscheduler started: sample=%.1fs apply_commands=%.1fs",
        sample_period_s,
        apply_commands_period_s,
    )


def stop_apscheduler() -> None:
    global _scheduler
    if _scheduler is None:
        return
    try:
        _scheduler.shutdown(wait=False)
    except Exception:
        logger.exception("apscheduler shutdown failed")
    _scheduler = None


# ---------------------------------------------------------------------------
# Inline jobs (NOT Celery tasks - invoked by APScheduler)
# ---------------------------------------------------------------------------


_READING_FIELDS = (
    "ts_iso", "ts_unix", "temp_c", "humidity_pct", "pressure_hpa", "lux",
    "current_ma", "bus_voltage", "shunt_mv", "power_mw", "ds18b20_c",
    "mq2_raw", "mq2_voltage", "pir_state",
)


def sample_sensors_inline() -> None:
    """Read all enabled sensors once and write a row to storage."""
    if not _lock.acquire(blocking=False):
        # Another job (apply_commands) is holding the I²C bus - skip this tick.
        return
    try:
        row: dict[str, Any] = {}
        ts = datetime.now(timezone.utc)
        row["ts_iso"] = ts.isoformat()
        row["ts_unix"] = ts.timestamp()
        for name, sensor in _resources["sensors"].items():
            try:
                data = sensor.read()
                if isinstance(data, dict):
                    for k, v in data.items():
                        if k in _READING_FIELDS:
                            row[k] = v
            except Exception:
                logger.warning("sensor %s read failed", name, exc_info=False)
                _emit_event("sensor_error", f"read failed: {name}", source=name)
        store = _resources.get("store")
        if store is None:
            return
        try:
            store.insert_reading(row)
        except Exception:
            logger.exception("insert_reading failed")
    finally:
        _lock.release()


def apply_pending_commands_inline() -> None:
    """Pull pending commands from storage and execute via singleton actuators."""
    store = _resources.get("store")
    if store is None:
        return
    try:
        pending = store.fetch_pending_commands()
    except Exception:
        logger.exception("fetch_pending_commands failed")
        return
    if not pending:
        return
    if not _lock.acquire(blocking=False):
        return
    try:
        for cmd in pending:
            cmd_id = cmd["id"] if hasattr(cmd, "__getitem__") else cmd.get("id")
            actuator_name = cmd["actuator"]
            action = cmd["action"]
            channel = cmd["channel"]
            value = cmd["value"]
            try:
                _execute_command(actuator_name, action, channel, value)
                store.update_command_status(cmd_id, "applied")
            except Exception as exc:
                logger.exception("command %s failed", cmd_id)
                store.update_command_status(cmd_id, "failed", error_msg=str(exc))
    finally:
        _lock.release()


def _execute_command(actuator: str, action: str, channel: int | None, value: float | None) -> None:
    actuators = _resources["actuators"]
    if actuator == "system" and action == "kill_switch":
        _kill_switch_inline()
        return
    if actuator == "relay":
        relay = actuators.get("relay")
        if relay is None:
            raise RuntimeError("relay not initialized")
        if action == "on":
            relay.on(int(channel) if channel is not None else 1)
        elif action == "off":
            relay.off(int(channel) if channel is not None else 1)
        else:
            raise ValueError(f"unknown relay action: {action}")
        return
    if actuator == "leds":
        leds = actuators.get("leds")
        if leds is None:
            raise RuntimeError("leds not initialized")
        if action == "red":
            leds.red()
        elif action == "yellow":
            leds.yellow()
        elif action == "green":
            leds.green()
        elif action == "off":
            leds.off()
        else:
            raise ValueError(f"unknown leds action: {action}")
        return
    if actuator == "servo":
        # Transient mode: if the servo is not initialized as a singleton (e.g.
        # MQ-2 currently owns Analog IO), build a ServoDriver just for this
        # command, run set_angle, wait ~1.5 s for the stroke to finish, then
        # close it - this releases Analog IO back to MQ-2 without restarting
        # the worker.
        servo = actuators.get("servo")
        transient = servo is None
        if transient:
            try:
                from src.actuators.servo import ServoDriver
                servo = ServoDriver()
            except Exception as exc:
                raise RuntimeError(f"failed to create transient servo: {exc}") from exc
        try:
            if action == "set_angle":
                if channel is None or value is None:
                    raise ValueError("servo set_angle needs channel and value")
                servo.set_angle(int(channel), float(value))
                if transient:
                    import time as _time
                    _time.sleep(1.5)  # enough for an SG90 full stroke
                    _emit_event(
                        "command", f"servo ch{channel} -> {value:.0f}° (transient)",
                        source="servo", value=float(value),
                    )
            elif action == "neutral":
                servo.neutral_all()
            else:
                raise ValueError(f"unknown servo action: {action}")
        finally:
            if transient:
                try:
                    servo.close()
                except Exception:
                    logger.exception("transient servo close failed")
        return
    raise ValueError(f"unknown actuator: {actuator}")


def _kill_switch_inline() -> None:
    """All relays OFF, LEDs OFF, servo to neutral. Best-effort."""
    actuators = _resources["actuators"]
    relay = actuators.get("relay")
    if relay is not None:
        try:
            relay.off_all()
        except Exception:
            logger.exception("relay off_all failed")
    leds = actuators.get("leds")
    if leds is not None:
        try:
            leds.off()
        except Exception:
            logger.exception("leds off failed")
    servo = actuators.get("servo")
    if servo is not None:
        try:
            servo.neutral_all()
        except Exception:
            logger.exception("servo neutral_all failed")
    _emit_event("kill_switch", "kill switch invoked", source="system")


def _emit_event(event_type: str, message: str, *, source: str | None = None, value: float | None = None) -> None:
    store = _resources.get("store")
    if store is None:
        return
    try:
        store.insert_event(event_type, message, source=source, value=value)
    except Exception:
        logger.exception("insert_event failed")


def _safe_emit_event(
    store: Any,
    event_type: str,
    message: str,
    *,
    source: str | None = None,
    value: float | None = None,
) -> None:
    """Like _emit_event but works during init when _resources['store'] not yet set."""
    if store is None:
        return
    try:
        store.insert_event(event_type, message, source=source, value=value)
    except Exception:
        logger.exception("safe_emit_event failed")


def _emit_sensor_init_event(sensor: Any, name: str, store: Any) -> None:
    """Persist BME280/BMP280 chip_id discovery to events table.

    Without this, sensor_type and chip_id were silently dropped by the
    _READING_FIELDS whitelist in sample_sensors_inline, so SQL had no
    way to tell whether a humidity_pct=NULL row meant "BMP280 detected"
    vs "BME280 read failed". This event closes that observability gap.
    """
    kind = getattr(sensor, "_sensor_kind", "unknown")
    chip = getattr(sensor, "_chip_id", 0)
    if name == "bme280":
        humidity_status = "real" if kind == "bme280" else "None (BMP280 has no humidity)"
        message = (
            f"{name} driver detected chip_id=0x{chip:02X} -> {kind}; "
            f"humidity_pct={humidity_status}"
        )
        _safe_emit_event(
            store, "sensor_init", message, source=name, value=float(chip)
        )
    else:
        _safe_emit_event(
            store, "sensor_init", f"{name} ready", source=name
        )


# ---------------------------------------------------------------------------
# Celery tasks (run on hw-worker only)
# ---------------------------------------------------------------------------


@celery_app.task(name="src.tasks.hardware.kill_switch_task", acks_late=True)
def kill_switch_task() -> dict:
    """High-priority task - gates all actuators to safe state."""
    if not _lock.acquire(blocking=True, timeout=2.0):
        return {"status": "busy"}
    try:
        _kill_switch_inline()
        return {"status": "ok"}
    finally:
        _lock.release()


@celery_app.task(name="src.tasks.hardware.retention_cleanup_task", acks_late=True)
def retention_cleanup_task() -> dict:
    """Hourly: delete readings older than RETENTION_DAYS."""
    store = _resources.get("store")
    if store is None:
        return {"status": "no_store"}
    try:
        cfg = load_runtime_config(PROJECT_ROOT / "configs" / "runtime.yml")
        days = getattr(cfg, "retention_days", 7)
    except Exception:
        days = 7
    deleted = store.cleanup_old_readings(retention_days=int(days))
    return {"status": "ok", "deleted": int(deleted)}


# Module-level convenience for tests
def get_resources() -> dict[str, Any]:
    return _resources
