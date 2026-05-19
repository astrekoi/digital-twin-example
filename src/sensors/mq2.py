"""MQ-2: gas sensor via Troyka HAT A0.

Current stand: Analog IO jumper is 5V->V and the default expected_vcc is "5V".
Legacy/alternate "3V3" mode remains supported explicitly.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

logger = logging.getLogger(__name__)


class MQ2Sensor:
    """Raw MQ-2 ADC channel via Troyka HAT, A0.

    Current stand uses Analog IO jumper 5V->V. expected_vcc="5V" therefore
    maps voltage to raw * 5.0. expected_vcc="3V3" remains available as an
    explicit alternate mode and maps voltage to raw * 3.3.

    Using MQ-2 and SG90 servo through the same Analog IO jumper at the same
    time is impossible: this is a physical stand mode, not a software fault.

    raw - normalised ADC value 0.0..1.0 (12-bit / 4095).
    voltage - raw × Vref (V). ppm conversion is not possible without a
    calibration table from a gas chamber.
    """

    def __init__(
        self,
        channel: int = 0,
        expected_vcc: Literal["3V3", "5V"] = "5V",
    ) -> None:
        # Current stand (verified 2026-04-26): MQ-2 uses Analog IO jumper 5V->V.
        # Keep 3V3 as an explicit supported mode for alternate hardware setup.
        # Do not convert to ppm without a calibration table.

        # Lazy import: protects against ImportError in CI/pytest without the
        # patched troykahat. analog_io is a factory - call analog_io() to get
        # an object with analogRead.
        from troykahat import analog_io

        self._io = analog_io()
        self._channel = channel
        self._expected_vcc = expected_vcc
        self._vref = 3.3 if expected_vcc == "3V3" else 5.0
        logger.info(
            "MQ-2 initialised on A%d (expected VCC=%s, Vref=%.1f V)",
            channel,
            expected_vcc,
            self._vref,
        )

    def read(self) -> dict:
        """Return mq2_raw (0.0-1.0) and mq2_voltage (V). No ppm conversion."""
        ts = datetime.now(timezone.utc)
        raw: float = self._io.analogRead(self._channel)
        return {
            "ts_iso": ts.isoformat(),
            "ts_unix": ts.timestamp(),
            "mq2_raw": round(raw, 4),
            "mq2_voltage": round(raw * self._vref, 4),
        }

    def close(self) -> None:
        pass
