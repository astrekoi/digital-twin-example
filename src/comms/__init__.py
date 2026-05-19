"""Communication interfaces: RS-485 and OLED display."""
from __future__ import annotations

from .oled import OLEDDisplay
from .rs485 import RS485

__all__ = ["RS485", "OLEDDisplay"]
