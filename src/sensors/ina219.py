"""INA219: current, voltage, power via adafruit-circuitpython-ina219."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import busio

from src.sensors import i2c_retry

logger = logging.getLogger(__name__)

# A0/A1=GND -> 0x40. Do not change without re-soldering jumpers A0/A1.
_ADDRESS = 0x40


class INA219Sensor:
    """High-side current / voltage via INA219 at I2C 0x40.

    Wiring: HAT 5V -> COM1 -> NO1 -> VIN+ -> [0.1Ω] -> VIN− -> (+)load -> GND.
    Negative current = VIN+/VIN− swapped; do NOT hide via abs() - it is a
    diagnostic indicator of incorrect wiring.
    """

    i2c_address: int = _ADDRESS

    def __init__(
        self,
        i2c: busio.I2C | None = None,
        address: int = _ADDRESS,
    ) -> None:
        import adafruit_ina219 as _ina
        import board
        import busio as _busio

        self._owns_i2c = i2c is None
        if i2c is None:
            i2c = _busio.I2C(board.SCL, board.SDA)
        self._i2c = i2c
        self._sensor = _ina.INA219(i2c, addr=address)
        self.i2c_address = address
        logger.debug("INA219 initialised at 0x%02X", address)

    def read(self) -> dict:
        """Return bus_voltage (V), shunt_mv (mV), current_ma (mA), power_mw (mW)."""
        ts = datetime.now(timezone.utc)
        bus_voltage = i2c_retry(lambda: self._sensor.bus_voltage)
        shunt_mv = i2c_retry(lambda: self._sensor.shunt_voltage)
        current_ma = i2c_retry(lambda: self._sensor.current)
        power_w = i2c_retry(lambda: self._sensor.power)
        return {
            "ts_iso": ts.isoformat(),
            "ts_unix": ts.timestamp(),
            "bus_voltage": round(bus_voltage, 3),
            "shunt_mv": round(shunt_mv, 3),
            "current_ma": round(current_ma, 2),
            "power_mw": round(power_w * 1000, 2),
        }

    def close(self) -> None:
        if self._owns_i2c:
            try:
                self._i2c.deinit()
            except Exception:
                pass
