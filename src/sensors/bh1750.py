"""BH1750 GY-302 ambient light sensor via adafruit-circuitpython-bh1750."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import busio

from src.sensors import i2c_retry

logger = logging.getLogger(__name__)

_ADDRESS = 0x23


class BH1750Sensor:
    """Ambient light (lux) via adafruit-circuitpython-bh1750 at I2C 0x23."""

    i2c_address: int = _ADDRESS

    def __init__(
        self,
        i2c: busio.I2C | None = None,
        address: int = _ADDRESS,
    ) -> None:
        import adafruit_bh1750 as _bh1750
        import board
        import busio as _busio

        self._owns_i2c = i2c is None
        if i2c is None:
            i2c = _busio.I2C(board.SCL, board.SDA)
        self._i2c = i2c
        self._sensor = _bh1750.BH1750(i2c, address=address)
        self.i2c_address = address
        logger.debug("BH1750 initialised at 0x%02X", address)

    def read(self) -> dict:
        """Read illuminance. Returns dict with ts_iso, ts_unix, lux."""
        ts = datetime.now(timezone.utc)
        lux = i2c_retry(lambda: self._sensor.lux)
        return {
            "ts_iso": ts.isoformat(),
            "ts_unix": ts.timestamp(),
            "lux": round(lux, 1),
        }

    def close(self) -> None:
        if self._owns_i2c:
            try:
                self._i2c.deinit()
            except Exception:
                pass
