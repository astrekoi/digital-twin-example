"""SG90 servos via Troyka HAT A1/A2 (jumper 5V->V).

Empirical control model (verified against the ~/cg.py probe):
- troykahat.analog_io().analogWrite(pin, duty) with duty in 0.0..1.0
- pin 1 -> A1, pin 2 -> A2
- The default troykahat PWM frequency (~1 kHz) does NOT match the nominal
  50 Hz of an SG90, but on this stand the servos respond correctly within
  the duty range 0.05..0.95 (no _setPwmFreq call - calling it created more
  problems than it solved).

Hardware constraint: the Analog IO jumper must be in 5V->V; the stand cannot
drive MQ-2 and the servos simultaneously (single physical mode, not a code
bug).
"""
from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

# Duty range for SG90 at the default troykahat frequency (see ~/cg.py).
_DC_MIN = 0.05      # -> 0 deg
_DC_MAX = 0.95      # -> 180 deg
_DC_NEUTRAL = 0.50  # -> ~90 deg (middle of the range)

# Troyka HAT Analog IO: pin 1 = A1, pin 2 = A2.
_VALID_CHANNELS = (1, 2)


class ServoDriver:
    """SG90 servos on Troyka HAT, channels A1 (ch=1) and A2 (ch=2).

    Angle range 0-180 deg, neutral at 90 deg. Duty mapping 0.05..0.95 is
    empirical.

    The `channels` kwarg is accepted for backwards compatibility with
    `configs/collector.yml -> actuators.servo.analog_channels` but is in fact
    ignored: both channels are always initialised (cheap operation).
    """

    def __init__(
        self,
        expected_vcc: Literal["5V"] = "5V",
        channels: list[int] | tuple[int, ...] | None = None,  # noqa: ARG002 - back-compat
    ) -> None:
        # Lazy import - protects pytest CI without the patched troykahat.
        from troykahat import analog_io

        self._io = analog_io()
        self._expected_vcc = expected_vcc

        # Set both servos to neutral on init (= duty 0.5).
        for ch in _VALID_CHANNELS:
            self._write_dc(ch, _DC_NEUTRAL)

        logger.info(
            "ServoDriver initialised: A1/A2, VCC=%s, duty 0.05..0.95 (empirical, ~1 kHz)",
            expected_vcc,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_angle(self, channel: int, angle: float) -> None:
        """Set angle 0-180 deg. channel=1 -> A1, channel=2 -> A2."""
        self._check(channel)
        angle = max(0.0, min(180.0, angle))
        dc = _DC_MIN + (angle / 180.0) * (_DC_MAX - _DC_MIN)
        self._write_dc(channel, dc)
        logger.info("Servo ch%d -> %.1f deg (duty=%.3f)", channel, angle, dc)

    def neutral(self, channel: int) -> None:
        self._check(channel)
        self._write_dc(channel, _DC_NEUTRAL)
        logger.debug("Servo ch%d -> neutral", channel)

    def neutral_all(self) -> None:
        for ch in _VALID_CHANNELS:
            self._write_dc(ch, _DC_NEUTRAL)
        logger.info("Both servos -> neutral")

    def close(self) -> None:
        self.neutral_all()
        logger.info("ServoDriver: released (servos in neutral)")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check(self, channel: int) -> None:
        if channel not in _VALID_CHANNELS:
            raise ValueError(f"Channel {channel} invalid; allowed: {_VALID_CHANNELS}")

    def _write_dc(self, channel: int, dc: float) -> None:
        self._io.analogWrite(channel, dc)
