"""Environmental sensor at 0x76: BME280 or BMP280 with auto-detection.

Auto-detect: reads chip_id from register 0xD0 via smbus2.
  chip_id 0x60 -> BME280: temp + humidity + pressure.
  chip_id 0x58 -> BMP280: temp + pressure only; humidity_pct = None.

The public class is still BME280Sensor so no callers need to change.
_read_chip_id is a module-level function to make it easily mockable in tests.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import busio

from src.sensors import i2c_retry

logger = logging.getLogger(__name__)

_ADDRESS = 0x76
_I2C_BUS = 1
_REG_CHIP_ID = 0xD0
_CHIP_ID_BME280 = 0x60
_CHIP_ID_BMP280 = 0x58

SensorKind = Literal["auto", "bme280", "bmp280"]


def _read_chip_id(address: int, bus_num: int = _I2C_BUS) -> int:
    """Read chip_id from register 0xD0 using smbus2.

    Separated into a module-level function so tests can patch it without
    opening real hardware.
    """
    import smbus2
    with smbus2.SMBus(bus_num) as bus:
        return bus.read_byte_data(address, _REG_CHIP_ID)


class BME280Sensor:
    """Environmental sensor wrapper supporting BME280 and BMP280.

    sensor_kind="auto" (default): reads chip_id via smbus2 to decide driver.
    sensor_kind="bme280": forces BME280 driver (adafruit_bme280).
    sensor_kind="bmp280": forces BMP280 driver (adafruit_bmp280).

    BMP280 has no humidity: read() returns humidity_pct=None for BMP280.
    """

    i2c_address: int = _ADDRESS

    def __init__(
        self,
        i2c: busio.I2C | None = None,
        address: int = _ADDRESS,
        sensor_kind: SensorKind = "auto",
    ) -> None:
        import board
        import busio as _busio

        self._owns_i2c = i2c is None
        if i2c is None:
            i2c = _busio.I2C(board.SCL, board.SDA)
        self._i2c = i2c
        self.i2c_address = address

        # Resolve actual sensor kind
        if sensor_kind == "auto":
            chip_id = _read_chip_id(address)
            if chip_id == _CHIP_ID_BME280:
                resolved_kind = "bme280"
            elif chip_id == _CHIP_ID_BMP280:
                resolved_kind = "bmp280"
                logger.warning(
                    "BMP280 detected (chip_id=0x58) at I²C 0x%02X: "
                    "humidity_pct will be None",
                    address,
                )
            else:
                raise RuntimeError(
                    f"Unknown chip_id=0x{chip_id:02X} at I²C 0x{address:02X}. "
                    f"Expected BME280 (chip_id=0x60) or BMP280 (chip_id=0x58)."
                )
        else:
            resolved_kind = sensor_kind
            chip_id = _CHIP_ID_BME280 if resolved_kind == "bme280" else _CHIP_ID_BMP280

        self._sensor_kind = resolved_kind
        self._chip_id = chip_id

        if resolved_kind == "bme280":
            import adafruit_bme280.basic as _bme
            self._sensor = _bme.Adafruit_BME280_I2C(i2c, address=address)
        else:  # bmp280
            try:
                import adafruit_bmp280 as _bmp
            except ImportError as exc:
                raise ImportError(
                    "BMP280 detected (chip_id=0x58) but adafruit-circuitpython-bmp280 "
                    "is not installed. Install with: "
                    "pip install adafruit-circuitpython-bmp280"
                ) from exc
            self._sensor = _bmp.Adafruit_BMP280_I2C(i2c, address=address)

        logger.debug(
            "%s initialized at I²C 0x%02X (chip_id=0x%02X)",
            resolved_kind.upper(),
            address,
            chip_id,
        )

    def read(self) -> dict:
        """Read sensor values.

        Returns dict with keys:
          ts_iso, ts_unix, temp_c, humidity_pct (None for BMP280),
          pressure_hpa, sensor_type ("bme280" or "bmp280"), chip_id ("0x60"/"0x58").
        """
        ts = datetime.now(timezone.utc)
        temp_c = i2c_retry(lambda: self._sensor.temperature)
        pressure_hpa = i2c_retry(lambda: self._sensor.pressure)

        if self._sensor_kind == "bme280":
            humidity_raw = i2c_retry(lambda: self._sensor.relative_humidity)
            humidity_pct: float | None = round(humidity_raw, 2)
        else:
            humidity_pct = None

        return {
            "ts_iso": ts.isoformat(),
            "ts_unix": ts.timestamp(),
            "temp_c": round(temp_c, 2),
            "humidity_pct": humidity_pct,
            "pressure_hpa": round(pressure_hpa, 2),
            "sensor_type": self._sensor_kind,
            "chip_id": f"0x{self._chip_id:02X}",
        }

    def close(self) -> None:
        if self._owns_i2c:
            try:
                self._i2c.deinit()
            except Exception:
                pass
