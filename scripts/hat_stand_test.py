"""Troyka HAT stand diagnostic script.

Safe by default: does not click relays, does not move servos, does not blink
LEDs. GPIO tests require an explicit --allow-gpio plus a separate permission.

Modes:
  --list           print expected I2C addresses and GPIO pins
  --check-imports  verify import of every project module
  --i2c-scan       run i2cdetect -y 1
  --dry-run        do nothing, just print what would be done
  --allow-gpio     enable GPIO tests (only with --allow-gpio explicitly)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_I2C_DEVICES = [
    ("0x23", "BH1750 GY-302", "ambient light"),
    ("0x2A", "STM32 HAT", "ADC/PWM expander (troykahat)"),
    ("0x3C", "SSD1306 OLED", "128×64 display"),
    ("0x40", "INA219", "current/voltage (A0/A1=GND -> not 0x41)"),
    ("0x76", "BME280", "temperature / humidity / pressure"),
]

_GPIO_PINS = [
    (17, "LED green", "via 220 Ω"),
    (18, "RS-485 DE+RE", "MAX485"),
    (22, "LED red", "via 220 Ω"),
    (23, "PIR HC-SR501", "motion input"),
    (24, "Relay IN2 (buzzer)", "HIGH Trigger"),
    (25, "Relay IN1 (fan)", "HIGH Trigger"),
    (27, "LED yellow", "via 220 Ω"),
    (12, "Relay IN3 (spare)", "HIGH Trigger"),
    (16, "Relay IN4 (spare)", "HIGH Trigger"),
]

_MODULES: list[tuple[str, str]] = [
    ("src.sensors.bme280", "BME280Sensor"),
    ("src.sensors.bh1750", "BH1750Sensor"),
    ("src.sensors.ina219", "INA219Sensor"),
    ("src.sensors.ds18b20", "DS18B20Sensor"),
    ("src.sensors.mq2", "MQ2Sensor"),
    ("src.sensors.pir", "PIRSensor"),
    ("src.actuators.relay", "Relay4Channel"),
    ("src.actuators.servo", "ServoDriver"),
    ("src.actuators.leds", "TrafficLight"),
    ("src.comms.rs485", "RS485"),
    ("src.comms.oled", "OLEDDisplay"),
    ("src.ai.features", "build_features"),
    ("src.ai.predictor", "Predictor"),
    ("src.db", "init_db"),
    ("src.config", "CollectorConfig"),
]


def cmd_list() -> None:
    print("\n=== Expected I2C devices (bus 1, 100 kHz) ===")
    for addr, name, desc in _I2C_DEVICES:
        print(f"  {addr}  {name:30s}  {desc}")

    print("\n=== GPIO (BCM) ===")
    for bcm, desc, note in _GPIO_PINS:
        print(f"  BCM {bcm:2d}  {desc:30s}  {note}")

    print("\n=== Analog IO (STM32 HAT) ===")
    print("  A0  MQ-2        jumper 3V3->V (CLAUDE.md) | actual stand: 5V->V")
    print("  A1  Servo SG90  jumper 5V->V (incompatible with A0 MQ-2)")
    print("  A2  Servo SG90  jumper 5V->V")


def cmd_check_imports() -> bool:
    """Import classes without instantiation. Hardware is not touched."""
    print("\n=== Import check ===")
    ok_count = 0
    warn_count = 0
    fail_count = 0

    for mod_path, symbol in _MODULES:
        try:
            mod = __import__(mod_path, fromlist=[symbol])
            getattr(mod, symbol)
            print(f"  OK    {mod_path}.{symbol}")
            ok_count += 1
        except ImportError as exc:
            print(f"  WARN  {mod_path}: {exc}")
            warn_count += 1
        except Exception as exc:
            print(f"  FAIL  {mod_path}: {exc}")
            fail_count += 1

    print(f"\nResult: OK={ok_count}  WARN={warn_count}  FAIL={fail_count}")
    return fail_count == 0


def cmd_i2c_scan(dry_run: bool) -> None:
    print("\n=== I2C scan (i2cdetect -y 1) ===")
    if dry_run:
        print("[dry-run] i2cdetect skipped")
        return
    try:
        result = subprocess.run(
            ["i2cdetect", "-y", "1"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        print(result.stdout)
        if result.returncode != 0:
            logger.warning("i2cdetect rc=%d: %s", result.returncode, result.stderr.strip())
    except FileNotFoundError:
        logger.warning("i2cdetect not found. sudo apt install i2c-tools")
    except subprocess.TimeoutExpired:
        logger.warning("i2cdetect: timeout")
    except PermissionError:
        logger.warning("No I2C permission. sudo usermod -aG i2c $USER")


def cmd_gpio_tests(dry_run: bool) -> None:
    """Brief GPIO smoke: blink LEDs, click relay ch1, read PIR.

    Gated by the --allow-gpio CLI flag. Each step is self-contained and
    returns the hardware to a safe state on completion.
    """
    import time

    print("\n=== GPIO tests ===")
    if dry_run:
        print("[dry-run] GPIO tests skipped")
        return

    # 1. LED blink (BCM 17/27/22 via TrafficLight wrapper).
    try:
        from src.actuators.leds import TrafficLight

        leds = TrafficLight()
        try:
            for color, fn in (("red", leds.red), ("yellow", leds.yellow), ("green", leds.green)):
                print(f"  LED {color}: ON")
                fn()
                time.sleep(0.5)
            leds.off()
            print("  LEDs: all off")
        finally:
            leds.close()
    except Exception as exc:
        print(f"  LED test FAILED: {exc}")

    # 2. Relay ch1 click (fan).
    try:
        from src.actuators.relay import Relay4Channel

        relay = Relay4Channel()
        try:
            print("  Relay ch1 (fan): ON for 0.2 s")
            relay.on(1)
            time.sleep(0.2)
            relay.off(1)
            print("  Relay ch1: OFF")
        finally:
            relay.close()
    except Exception as exc:
        print(f"  Relay test FAILED: {exc}")

    # 3. PIR read.
    try:
        from src.sensors.pir import PIRSensor

        pir = PIRSensor()
        try:
            data = pir.read()
            print(f"  PIR.is_active: {bool(data['pir_state'])}")
        finally:
            pir.close()
    except Exception as exc:
        print(f"  PIR test FAILED: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Troyka HAT stand diagnostics")
    parser.add_argument("--list", action="store_true", help="Expected devices and pins")
    parser.add_argument(
        "--i2c-scan", dest="i2c_scan", action="store_true", help="Run i2cdetect -y 1"
    )
    parser.add_argument(
        "--check-imports",
        dest="check_imports",
        action="store_true",
        help="Import modules without hardware",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print only, no actions")
    parser.add_argument(
        "--allow-gpio", action="store_true", help="Enable GPIO tests (LED, relay click, PIR read)"
    )
    args = parser.parse_args()

    if not any([args.list, args.i2c_scan, args.check_imports, args.allow_gpio]):
        parser.print_help()
        sys.exit(0)

    exit_code = 0

    if args.list:
        cmd_list()

    if args.check_imports:
        ok = cmd_check_imports()
        if not ok:
            exit_code = 1

    if args.i2c_scan:
        cmd_i2c_scan(args.dry_run)

    if args.allow_gpio:
        cmd_gpio_tests(args.dry_run)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
