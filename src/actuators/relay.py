"""Four-channel SRD-05VDC-SL-C relay board via gpiozero.OutputDevice."""
from __future__ import annotations

import logging
import signal
from typing import FrozenSet

from gpiozero import OutputDevice

logger = logging.getLogger(__name__)

# WP->BCM mapping per CLAUDE.md. BCM 7/8 must NOT be used (HW pull-up keeps
# the relay latched HIGH after .close()).
# HIGH-trigger jumper; `gpio=12,16,24,25=op,dl` in config.txt holds LOW at boot.
_CHANNEL_PINS: dict[int, int] = {
    1: 25,  # WP 6 - IN1, RQD 4010MS fan
    2: 24,  # WP 5 - IN2, buzzer
    3: 12,  # WP 26 - IN3, spare
    4: 16,  # WP 27 - IN4, spare
}

VALID_CHANNELS: FrozenSet[int] = frozenset(_CHANNEL_PINS)


class Relay4Channel:
    """Control of four relay channels.

    All channels: active_high=True (HIGH-trigger jumper), initial_value=False.
    SIGTERM -> off_all() + close() so that relays do not remain HIGH on
    `systemctl stop`.
    """

    def __init__(self) -> None:
        self._relays: dict[int, OutputDevice] = {}
        for ch, pin in _CHANNEL_PINS.items():
            self._relays[ch] = OutputDevice(pin, active_high=True, initial_value=False)
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        logger.info(
            "Relay4Channel initialised: ch1=BCM25(fan), ch2=BCM24(buzzer), "
            "ch3=BCM12(rsv), ch4=BCM16(rsv)"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on(self, ch: int) -> None:
        self._check(ch)
        self._relays[ch].on()
        logger.debug("Relay ch%d ON", ch)

    def off(self, ch: int) -> None:
        self._check(ch)
        self._relays[ch].off()
        logger.debug("Relay ch%d OFF", ch)

    def off_all(self) -> None:
        for ch, dev in self._relays.items():
            dev.off()
        logger.info("All relays off")

    def state(self, ch: int) -> bool:
        self._check(ch)
        return bool(self._relays[ch].value)

    def close(self) -> None:
        self.off_all()
        for dev in self._relays.values():
            dev.close()
        self._relays.clear()
        logger.info("Relay4Channel: resources released")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check(self, ch: int) -> None:
        if ch not in VALID_CHANNELS:
            raise ValueError(f"Channel {ch} invalid; allowed: {sorted(VALID_CHANNELS)}")

    def _sigterm_handler(self, signum: int, frame: object) -> None:  # noqa: ARG002
        logger.warning("SIGTERM received - turning all relays off")
        self.close()
