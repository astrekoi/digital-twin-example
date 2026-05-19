"""SSD1306 OLED 128×64 via luma.oled, I2C 0x3C, <=2 Hz."""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_I2C_ADDRESS = 0x3C
_I2C_PORT = 1
_MIN_INTERVAL_S = 0.5   # 2 Hz cap - Pi 5 has no rendering headroom above this
_LINE_HEIGHT_PX = 12    # pixels per line (5 lines fit in 64px)
_MAX_LINES = 5


class OLEDDisplay:
    """SSD1306 128×64 over I2C 0x3C via luma.oled.

    Refresh rate <=2 Hz: show_* calls more frequent than _MIN_INTERVAL_S are
    skipped without raising (debug log only). This guards against accidental
    polling.

    Lazy import of luma / PIL: the module loads in CI without a display.
    """

    def __init__(
        self,
        i2c_address: int = _I2C_ADDRESS,
        i2c_port: int = _I2C_PORT,
    ) -> None:
        from luma.core.interface.serial import i2c
        from luma.oled.device import ssd1306

        serial = i2c(port=i2c_port, address=i2c_address)
        self._device = ssd1306(serial)
        self._last_update: float = 0.0
        logger.info(
            "OLEDDisplay initialised: SSD1306 128×64 at I2C 0x%02X",
            i2c_address,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_lines(self, lines: list[str]) -> None:
        """Render a list of strings (up to 5 lines, 12 px each)."""
        if not self._rate_ok():
            return
        from luma.core.render import canvas

        with canvas(self._device) as draw:
            for i, line in enumerate(lines[:_MAX_LINES]):
                draw.text((0, i * _LINE_HEIGHT_PX), line, fill="white")

    def show_telemetry(self, data: dict[str, Any]) -> None:
        """Render a dict as 'key: value', one entry per line."""
        lines = [f"{k}: {v}" for k, v in data.items()]
        self.show_lines(lines)

    def clear(self) -> None:
        """Clear the screen."""
        if not self._rate_ok():
            return
        self._device.clear()

    def close(self) -> None:
        try:
            self._device.cleanup()
        except Exception:
            pass
        logger.info("OLEDDisplay: resources released")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_ok(self) -> bool:
        now = time.monotonic()
        if now - self._last_update >= _MIN_INTERVAL_S:
            self._last_update = now
            return True
        logger.debug("OLED: update skipped - call too frequent (<%g s)", _MIN_INTERVAL_S)
        return False
