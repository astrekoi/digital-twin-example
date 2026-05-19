"""Main telemetry collector: sensors -> SQLite + CSV.

Sole owner of GPIO and physical hardware.
The dashboard and predictor_daemon do not touch hardware.

Run modes:
  --once      - single read cycle and exit (debug / test)
  --dry-run   - stub data, no GPIO/I2C init
  <no flags>  - long-running 1 Hz loop (systemd-ready)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import signal
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    CollectorConfig,
    load_collector_config,
    load_runtime_config,
    setup_logging,
)
from src.db import (
    cleanup_old_readings,
    fetch_pending_commands,
    init_db,
    insert_event,
    insert_reading,
    update_command_status,
    vacuum,
)

logger = logging.getLogger(__name__)

COMMAND_RATE_LIMIT_S = 2.0

# Fixed CSV schema. Do not change without updating the training-data schema.
CSV_FIELDNAMES = [
    "ts_iso",
    "ts_unix",
    "temp_c",
    "humidity_pct",
    "pressure_hpa",
    "lux",
    "current_ma",
    "voltage_v",
    "mq2_raw",
    "mq2_voltage_v",
    "pir_state",
]

# Stub values for dry-run (no GPIO/I2C touched).
_STUB: dict[str, Any] = {
    "temp_c": 22.5,
    "humidity_pct": 55.0,
    "pressure_hpa": 1013.0,
    "lux": 100.0,
    "current_ma": 0.0,
    "bus_voltage": 5.0,
    "shunt_mv": 0.0,
    "power_mw": 0.0,
    "ds18b20_c": 22.3,
    "mq2_raw": 0.1,
    "mq2_voltage": 0.5,
    "pir_state": 0,
}

_shutdown_requested = False


def _request_shutdown(signum: int, frame: Any) -> None:
    global _shutdown_requested
    logger.info("Signal %s - shutdown requested", signum)
    _shutdown_requested = True


def create_sensor_suite(config: CollectorConfig, dry_run: bool) -> dict:
    """Init sensors per config. A failure of one sensor -> warning, not crash."""
    if dry_run:
        logger.info("dry-run: sensors not initialised")
        return {}

    sc = config.sensors
    sensors: dict[str, Any] = {}

    if sc.bme280.enabled:
        try:
            from src.sensors.bme280 import BME280Sensor

            sensors["bme280"] = BME280Sensor(
                address=sc.bme280.i2c_address,
                sensor_kind=sc.bme280.kind,
            )
        except Exception as exc:
            logger.warning("BME280 unavailable: %s", exc)

    if sc.bh1750.enabled:
        try:
            from src.sensors.bh1750 import BH1750Sensor

            sensors["bh1750"] = BH1750Sensor(address=sc.bh1750.i2c_address)
        except Exception as exc:
            logger.warning("BH1750 unavailable: %s", exc)

    if sc.ina219.enabled:
        try:
            from src.sensors.ina219 import INA219Sensor

            sensors["ina219"] = INA219Sensor(address=sc.ina219.i2c_address)
        except Exception as exc:
            logger.warning("INA219 unavailable: %s", exc)

    if sc.ds18b20.enabled:
        try:
            from src.sensors.ds18b20 import DS18B20Sensor

            sensors["ds18b20"] = DS18B20Sensor()
        except Exception as exc:
            logger.warning("DS18B20 unavailable: %s", exc)

    if sc.mq2.enabled:
        try:
            from src.sensors.mq2 import MQ2Sensor

            # expected_vcc comes from config; current stand uses 5V->V.
            sensors["mq2"] = MQ2Sensor(channel=0, expected_vcc=sc.mq2.expected_vcc)
        except Exception as exc:
            logger.warning("MQ-2 unavailable: %s", exc)

    if sc.pir.enabled:
        try:
            from src.sensors.pir import PIRSensor

            sensors["pir"] = PIRSensor(bcm_pin=sc.pir.bcm_pin)
        except Exception as exc:
            logger.warning("PIR unavailable: %s", exc)

    return sensors


def create_actuator_suite(config: CollectorConfig, dry_run: bool) -> dict:
    """Init actuators. dry-run -> empty dict."""
    if dry_run:
        logger.info("dry-run: actuators not initialised")
        return {}

    ac = config.actuators
    actuators: dict[str, Any] = {}

    if ac.relay.enabled:
        try:
            from src.actuators.relay import Relay4Channel

            actuators["relay"] = Relay4Channel()
        except Exception as exc:
            logger.warning("Relay unavailable: %s", exc)

    if ac.servo.enabled:
        try:
            from src.actuators.servo import ServoDriver

            actuators["servo"] = ServoDriver(expected_vcc=ac.servo.expected_vcc)
        except Exception as exc:
            logger.warning("ServoDriver unavailable: %s", exc)

    if ac.leds.enabled:
        try:
            from src.actuators.leds import TrafficLight

            actuators["leds"] = TrafficLight()
        except Exception as exc:
            logger.warning("TrafficLight unavailable: %s", exc)

    return actuators


def read_all_sensors(
    sensors: dict,
    dry_run: bool = False,
    conn: Any = None,
) -> dict:
    """Read all sensors and return a single telemetry row.

    Keys match the DB column names (for insert_reading).
    A sensor failure -> field is None and an event is written to the DB.
    """
    ts = datetime.now(timezone.utc)
    reading: dict[str, Any] = {
        "ts_iso": ts.isoformat(),
        "ts_unix": ts.timestamp(),
        "temp_c": None,
        "humidity_pct": None,
        "pressure_hpa": None,
        "lux": None,
        "current_ma": None,
        "bus_voltage": None,
        "shunt_mv": None,
        "power_mw": None,
        "ds18b20_c": None,
        "mq2_raw": None,
        "mq2_voltage": None,
        "pir_state": None,
    }

    if dry_run:
        reading.update(_STUB)
        return reading

    def _safe_read(name: str, fn: Any) -> Any:
        try:
            return fn()
        except Exception as exc:
            logger.warning("Sensor %s read error: %s", name, exc)
            if conn is not None:
                try:
                    insert_event(conn, "sensor_error", str(exc), source=name)
                except Exception:
                    pass
            return None

    if "bme280" in sensors:
        data = _safe_read("bme280", sensors["bme280"].read)
        if data:
            reading["temp_c"] = data.get("temp_c")
            reading["humidity_pct"] = data.get("humidity_pct")
            reading["pressure_hpa"] = data.get("pressure_hpa")

    if "bh1750" in sensors:
        data = _safe_read("bh1750", sensors["bh1750"].read)
        if data:
            reading["lux"] = data.get("lux")

    if "ina219" in sensors:
        data = _safe_read("ina219", sensors["ina219"].read)
        if data:
            reading["current_ma"] = data.get("current_ma")
            reading["bus_voltage"] = data.get("bus_voltage")
            reading["shunt_mv"] = data.get("shunt_mv")
            reading["power_mw"] = data.get("power_mw")

    if "ds18b20" in sensors:
        data = _safe_read("ds18b20", sensors["ds18b20"].read)
        if data:
            reading["ds18b20_c"] = data.get("ds18b20_c")

    if "mq2" in sensors:
        data = _safe_read("mq2", sensors["mq2"].read)
        if data:
            reading["mq2_raw"] = data.get("mq2_raw")
            reading["mq2_voltage"] = data.get("mq2_voltage")

    if "pir" in sensors:
        data = _safe_read("pir", sensors["pir"].read)
        if data:
            reading["pir_state"] = data.get("pir_state")

    return reading


def _to_csv_row(reading: dict) -> dict:
    """Map DB keys -> CSV columns (bus_voltage->voltage_v, mq2_voltage->mq2_voltage_v)."""
    return {
        "ts_iso": reading.get("ts_iso"),
        "ts_unix": reading.get("ts_unix"),
        "temp_c": reading.get("temp_c"),
        "humidity_pct": reading.get("humidity_pct"),
        "pressure_hpa": reading.get("pressure_hpa"),
        "lux": reading.get("lux"),
        "current_ma": reading.get("current_ma"),
        "voltage_v": reading.get("bus_voltage"),
        "mq2_raw": reading.get("mq2_raw"),
        "mq2_voltage_v": reading.get("mq2_voltage"),
        "pir_state": reading.get("pir_state"),
    }


def _check_thresholds(reading: dict, runtime_cfg: Any, conn: Any) -> None:
    """Write events when readings violate thresholds in runtime_cfg."""
    th = runtime_cfg.thresholds
    checks = [
        ("temp_c", reading.get("temp_c"), th.temp_c_max, ">"),
        ("temp_c", reading.get("temp_c"), th.temp_c_min, "<"),
        ("mq2_raw", reading.get("mq2_raw"), th.mq2_raw_alert, ">"),
        ("current_ma", reading.get("current_ma"), th.current_ma_max, ">"),
    ]
    for name, val, limit, op in checks:
        if val is None:
            continue
        triggered = val > limit if op == ">" else val < limit
        if triggered:
            msg = f"{name}={val:.4g} {op} threshold {limit}"
            logger.warning("Threshold violated: %s", msg)
            try:
                insert_event(conn, "threshold", msg, source=name, value=float(val))
            except Exception:
                pass


def _command_rate_key(
    actuator: str, action: str, channel: int | None
) -> tuple[str, int | str] | None:
    """Return rate-limit key for actuator commands; None means bypass."""
    if action == "kill_switch":
        return None
    if actuator == "relay":
        return ("relay", int(channel) if channel is not None else "unknown")
    if actuator == "servo":
        return ("servo", int(channel) if channel is not None else "unknown")
    if actuator == "leds":
        return ("traffic_light", "global")
    return (actuator, channel if channel is not None else "global")


def apply_pending_commands(
    conn: Any,
    actuators: dict,
    rate_state: dict[tuple[str, int | str], float] | None = None,
    now_fn: Any = time.monotonic,
) -> int:
    """Apply pending rows from the commands table. Return count processed."""
    commands = fetch_pending_commands(conn)
    if not commands:
        return 0

    if rate_state is None:
        rate_state = {}

    applied = 0
    for cmd in commands:
        cmd_id = cmd["id"]
        actuator = cmd["actuator"]
        action = cmd["action"]
        channel = cmd["channel"]
        value = cmd["value"]
        rate_key = _command_rate_key(actuator, action, channel)
        now = now_fn()

        if rate_key is not None:
            last_ts = rate_state.get(rate_key)
            if last_ts is not None and now - last_ts < COMMAND_RATE_LIMIT_S:
                errmsg = (
                    f"rate limited: {actuator}/{action} ch={channel} "
                    f"within {COMMAND_RATE_LIMIT_S:.1f}s"
                )
                logger.warning("Command %d rejected by rate limit: %s", cmd_id, errmsg)
                update_command_status(conn, cmd_id, "failed", error_msg=errmsg)
                try:
                    insert_event(
                        conn, "command", f"FAILED {actuator}/{action}: {errmsg}", source="collector"
                    )
                except Exception:
                    pass
                continue

        try:
            _execute_command(actuators, actuator, action, channel, value)
            if rate_key is not None:
                rate_state[rate_key] = now
            update_command_status(conn, cmd_id, "applied")
            insert_event(
                conn,
                "command",
                f"applied {actuator}/{action} ch={channel} val={value}",
                source="collector",
            )
            applied += 1
            logger.info(
                "Command %d applied: %s/%s ch=%s val=%s", cmd_id, actuator, action, channel, value
            )
        except Exception as exc:
            errmsg = str(exc)
            logger.error("Command %d (%s/%s) failed: %s", cmd_id, actuator, action, errmsg)
            update_command_status(conn, cmd_id, "failed", error_msg=errmsg)
            try:
                insert_event(
                    conn, "command", f"FAILED {actuator}/{action}: {errmsg}", source="collector"
                )
            except Exception:
                pass

    return applied


def _execute_command(
    actuators: dict,
    actuator: str,
    action: str,
    channel: int | None,
    value: float | None,
) -> None:
    """Execute one command. Raises ValueError/RuntimeError on bad arguments.

    Command schema (written by dashboard via db.insert_command):
      kill_switch    actuator="system"  action="kill_switch"
      relay_set      actuator="relay"   action="on"|"off"   channel=1..4
      traffic_light  actuator="leds"    action="red"|"yellow"|"green"|"off"
      servo_set      actuator="servo"   action="set_angle"  channel=1|2  value=0..180
    """
    if action == "kill_switch":
        logger.warning("KILL SWITCH: turning all actuators off")
        if "relay" in actuators:
            actuators["relay"].off_all()
        if "leds" in actuators:
            actuators["leds"].off()
        if "servo" in actuators:
            actuators["servo"].neutral_all()
        return

    if actuator == "relay":
        relay = actuators.get("relay")
        if relay is None:
            raise RuntimeError("Relay is not initialised")
        if channel not in (1, 2, 3, 4):
            raise ValueError(f"Invalid relay channel: {channel!r} (expected 1-4)")
        if action == "on":
            relay.on(int(channel))
        elif action == "off":
            relay.off(int(channel))
        else:
            raise ValueError(f"Unknown relay action: {action!r}")

    elif actuator == "leds":
        leds = actuators.get("leds")
        if leds is None:
            raise RuntimeError("TrafficLight is not initialised")
        if action not in ("red", "yellow", "green", "off"):
            raise ValueError(f"Unknown LED action: {action!r}")
        getattr(leds, action)()

    elif actuator == "servo":
        servo = actuators.get("servo")
        if servo is None:
            raise RuntimeError("ServoDriver is not initialised")
        if channel not in (1, 2):
            raise ValueError(f"Invalid servo channel: {channel!r} (expected 1 or 2)")
        if action != "set_angle":
            raise ValueError(f"Unknown servo action: {action!r}")
        angle = float(value if value is not None else 90.0)
        if not 0.0 <= angle <= 180.0:
            raise ValueError(f"Servo angle {angle} out of range 0..180")
        servo.set_angle(int(channel), angle)

    else:
        raise ValueError(f"Unknown actuator: {actuator!r}")


def _close_resources(sensors: dict, actuators: dict) -> None:
    """Move actuators to safe state and release resources."""
    if "relay" in actuators:
        try:
            actuators["relay"].off_all()
            actuators["relay"].close()
        except Exception as exc:
            logger.error("Relay close error: %s", exc)

    if "servo" in actuators:
        try:
            actuators["servo"].neutral_all()
            actuators["servo"].close()
        except Exception as exc:
            logger.error("Servo close error: %s", exc)

    if "leds" in actuators:
        try:
            actuators["leds"].off()
            actuators["leds"].close()
        except Exception as exc:
            logger.error("LED close error: %s", exc)

    for name, sensor in sensors.items():
        try:
            sensor.close()
        except Exception as exc:
            logger.error("Sensor %s close error: %s", name, exc)


def run_once(config: CollectorConfig, dry_run: bool = False) -> dict:
    """Single full cycle: init -> read -> DB -> CSV -> commands -> shutdown."""
    conn = init_db(config.db_path)
    sensors = create_sensor_suite(config, dry_run)
    actuators = create_actuator_suite(config, dry_run)

    try:
        reading = read_all_sensors(sensors, dry_run=dry_run, conn=conn)
        insert_reading(conn, reading)

        today = date.today().isoformat()
        csv_dir = config.csv_dir
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"{config.csv_prefix}_{today}.csv"
        new_file = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if new_file:
                writer.writeheader()
            writer.writerow(_to_csv_row(reading))
            f.flush()

        n_cmd = apply_pending_commands(conn, actuators)
        logger.info(
            "run_once: temp=%.1f°C rh=%.1f%% mq2_raw=%s pir=%s cmds=%d",
            reading.get("temp_c") or 0.0,
            reading.get("humidity_pct") or 0.0,
            reading.get("mq2_raw"),
            reading.get("pir_state"),
            n_cmd,
        )
        return reading
    finally:
        _close_resources(sensors, actuators)
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="PLC Twin - collector")
    parser.add_argument("--config", default="configs/collector.yml")
    parser.add_argument("--once", action="store_true", help="One cycle and exit")
    parser.add_argument("--dry-run", action="store_true", help="Stub data, no hardware")
    args = parser.parse_args()

    setup_logging()
    config = load_collector_config(Path(args.config))
    logger.info(
        "Collector starting: once=%s dry_run=%s config=%s",
        args.once,
        args.dry_run,
        args.config,
    )

    if args.once:
        result = run_once(config, dry_run=args.dry_run)
        safe = {k: v for k, v in result.items() if k not in ("ts_iso",)}
        print(json.dumps(safe, indent=2, default=str))
        return

    conn = init_db(config.db_path)
    sensors = create_sensor_suite(config, args.dry_run)
    actuators = create_actuator_suite(config, args.dry_run)

    # SIGTERM/SIGINT - register AFTER Relay4Channel.__init__ so we override
    # the relay's own SIGTERM handler.
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    if "pir" in sensors:

        def _pir_motion():
            try:
                insert_event(conn, "pir", "Motion detected", source="pir", value=1.0)
            except Exception:
                pass

        def _pir_no_motion():
            try:
                insert_event(conn, "pir", "Motion ended", source="pir", value=0.0)
            except Exception:
                pass

        sensors["pir"].on_motion = _pir_motion
        sensors["pir"].on_no_motion = _pir_no_motion

    csv_file: Any = None
    csv_writer: Any = None
    current_csv_date: str | None = None

    command_rate_state: dict[tuple[str, int | str], float] = {}
    last_cleanup_ts: float = 0.0
    last_vacuum_ts: float = 0.0
    last_config_reload_ts: float = 0.0
    runtime_cfg: Any = None

    try:
        runtime_cfg = load_runtime_config(config.runtime_config_path)
    except Exception:
        pass

    logger.info(
        "Collector running (%.1f Hz). SIGTERM/Ctrl+C to stop.", 1.0 / config.sample_period_s
    )

    try:
        while not _shutdown_requested:
            cycle_start = time.monotonic()
            now_ts = time.time()

            # CSV rotation per day
            today = date.today().isoformat()
            if today != current_csv_date:
                if csv_file is not None:
                    try:
                        csv_file.close()
                    except Exception:
                        pass
                csv_dir = config.csv_dir
                csv_dir.mkdir(parents=True, exist_ok=True)
                csv_path = csv_dir / f"{config.csv_prefix}_{today}.csv"
                new_file = not csv_path.exists()
                csv_file = csv_path.open("a", newline="")
                csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
                if new_file:
                    csv_writer.writeheader()
                current_csv_date = today
                logger.info("CSV: %s", csv_path)

            reading = read_all_sensors(sensors, dry_run=args.dry_run, conn=conn)

            if runtime_cfg is not None:
                _check_thresholds(reading, runtime_cfg, conn)

            try:
                insert_reading(conn, reading)
            except Exception as exc:
                logger.error("Reading DB write error: %s", exc)

            try:
                csv_writer.writerow(_to_csv_row(reading))
                csv_file.flush()
            except Exception as exc:
                logger.error("CSV write error: %s", exc)

            try:
                apply_pending_commands(conn, actuators, rate_state=command_rate_state)
            except Exception as exc:
                logger.error("Command apply error: %s", exc)

            # DB maintenance (1/hour)
            if now_ts - last_cleanup_ts >= 3600:
                try:
                    cleanup_old_readings(conn, config.retention_days)
                    last_cleanup_ts = now_ts
                except Exception as exc:
                    logger.error("cleanup_old_readings error: %s", exc)

            # VACUUM (1/7 days)
            if now_ts - last_vacuum_ts >= 7 * 86400:
                try:
                    vacuum(conn)
                    last_vacuum_ts = now_ts
                except Exception as exc:
                    logger.error("VACUUM error: %s", exc)

            # Hot-reload runtime.yml (every 60 s by default)
            if now_ts - last_config_reload_ts >= config.config_reload_interval_s:
                try:
                    runtime_cfg = load_runtime_config(config.runtime_config_path)
                    last_config_reload_ts = now_ts
                except Exception:
                    pass

            # Precise sleep until the next cycle
            elapsed = time.monotonic() - cycle_start
            sleep_s = config.sample_period_s - elapsed
            if sleep_s > 0 and not _shutdown_requested:
                time.sleep(sleep_s)

    finally:
        logger.info("Collector: shutting down...")
        if csv_file is not None:
            try:
                csv_file.close()
            except Exception:
                pass
        _close_resources(sensors, actuators)
        conn.close()
        logger.info("Collector stopped.")


if __name__ == "__main__":
    main()
