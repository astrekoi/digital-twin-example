"""HC-SR501 PIR motion sensor via gpiozero.MotionSensor."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger(__name__)

# WP 4 = BCM 23. The Amperka PDF has a typo (BCM 27 listed twice) -
# verified empirically.
_BCM_PIN = 23


class PIRSensor:
    """PIR HC-SR501 on BCM 23 via gpiozero.MotionSensor.

    Edges (motion_detected / motion_ended) are logged via callbacks, not
    polling. To persist events to the DB, set on_motion / on_no_motion from
    outside (e.g. in collector.py).
    """

    def __init__(self, bcm_pin: int = _BCM_PIN) -> None:
        from gpiozero import MotionSensor

        self._sensor = MotionSensor(bcm_pin)
        self._sensor.when_motion = self._on_motion
        self._sensor.when_no_motion = self._on_no_motion
        # External callbacks set by the collector to persist events in the DB.
        self.on_motion: Callable[[], None] | None = None
        self.on_no_motion: Callable[[], None] | None = None
        logger.info("PIR initialised on BCM %d", bcm_pin)

    def _on_motion(self) -> None:
        logger.info("PIR: motion detected")
        if self.on_motion:
            self.on_motion()

    def _on_no_motion(self) -> None:
        logger.debug("PIR: motion ended")
        if self.on_no_motion:
            self.on_no_motion()

    def read(self) -> dict:
        """Current state: pir_state=1 if motion is active right now."""
        ts = datetime.now(timezone.utc)
        return {
            "ts_iso": ts.isoformat(),
            "ts_unix": ts.timestamp(),
            "pir_state": int(self._sensor.is_active),
        }

    def close(self) -> None:
        self._sensor.close()
        logger.debug("PIR: resources released")
