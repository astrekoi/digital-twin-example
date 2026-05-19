"""DS18B20: temperature via /sys/bus/w1. No third-party library."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_W1_BASE = Path("/sys/bus/w1/devices")


class DS18B20Sensor:
    """DS18B20 1-Wire via /sys/bus/w1/devices/28-*/w1_slave.

    Does not use w1thermsensor - avoids an extra dependency.
    Returns None if the device is missing or the CRC check fails.
    """

    def __init__(self) -> None:
        logger.debug("DS18B20 initialised (1-Wire, %s)", _W1_BASE)

    def _find_device_path(self) -> Path | None:
        devices = sorted(_W1_BASE.glob("28-*/w1_slave"))
        if not devices:
            logger.warning("DS18B20: no 28-* device found in %s", _W1_BASE)
            return None
        if len(devices) > 1:
            logger.warning(
                "DS18B20: %d devices found, using the first: %s",
                len(devices),
                devices[0],
            )
        return devices[0]

    def _read_raw(self) -> float | None:
        path = self._find_device_path()
        if path is None:
            return None
        try:
            text = path.read_text(encoding="ascii")
        except OSError as e:
            logger.error("DS18B20: read error on %s: %s", path, e)
            return None
        if "YES" not in text:
            logger.warning("DS18B20: CRC check failed (%s)", path.parent.name)
            return None
        try:
            t_idx = text.index("t=")
            # Text after "t=" may include a newline; take only the digits.
            raw_str = text[t_idx + 2 :].strip().split()[0]
            return round(int(raw_str) / 1000.0, 3)
        except (ValueError, IndexError) as e:
            logger.error("DS18B20: parse error on '%s': %s", text.strip(), e)
            return None

    def read(self) -> dict:
        """Return dict with ts_iso, ts_unix, ds18b20_c (None on error)."""
        ts = datetime.now(timezone.utc)
        return {
            "ts_iso": ts.isoformat(),
            "ts_unix": ts.timestamp(),
            "ds18b20_c": self._read_raw(),
        }

    def close(self) -> None:
        pass  # no hardware resources to release
