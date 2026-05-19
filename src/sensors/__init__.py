"""Sensor wrappers for Raspberry Pi 5 / Troyka HAT."""
from __future__ import annotations

import time


def i2c_retry(fn, retries: int = 3, delay: float = 0.01):
    """Retry an I2C operation with exponential backoff on OSError."""
    for i in range(retries):
        try:
            return fn()
        except OSError:
            if i == retries - 1:
                raise
            time.sleep(delay * (2**i))
