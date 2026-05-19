"""Traffic-light block: three LEDs via gpiozero.LED."""
from __future__ import annotations

import logging

from gpiozero import LED

logger = logging.getLogger(__name__)

# All three LEDs are wired through 220 Ω current-limiting resistors.
_BCM_RED = 22     # WP 3
_BCM_YELLOW = 27  # WP 2
_BCM_GREEN = 17   # WP 0


class TrafficLight:
    """Three LEDs (red / yellow / green) used as a traffic light.

    Each red()/yellow()/green() call lights one LED and turns the other two
    off. off() turns all off. close() turns all off and releases GPIO.
    """

    def __init__(self) -> None:
        self._red = LED(_BCM_RED)
        self._yellow = LED(_BCM_YELLOW)
        self._green = LED(_BCM_GREEN)
        self.off()
        logger.info(
            "TrafficLight initialised: red=BCM%d, yellow=BCM%d, green=BCM%d",
            _BCM_RED, _BCM_YELLOW, _BCM_GREEN,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def red(self) -> None:
        self._red.on()
        self._yellow.off()
        self._green.off()
        logger.debug("LED: red")

    def yellow(self) -> None:
        self._red.off()
        self._yellow.on()
        self._green.off()
        logger.debug("LED: yellow")

    def green(self) -> None:
        self._red.off()
        self._yellow.off()
        self._green.on()
        logger.debug("LED: green")

    def off(self) -> None:
        self._red.off()
        self._yellow.off()
        self._green.off()
        logger.debug("LED: all off")

    def close(self) -> None:
        self.off()
        self._red.close()
        self._yellow.close()
        self._green.close()
        logger.info("TrafficLight: resources released")
