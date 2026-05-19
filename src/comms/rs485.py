"""MAX485 RS-485 half-duplex via pyserial + gpiozero.OutputDevice."""
from __future__ import annotations

import logging
import time
from types import TracebackType

import serial
from gpiozero import OutputDevice

logger = logging.getLogger(__name__)

# BCM 18 = WP 1, DE+RE are tied together on the MAX485 board.
_BCM_DE_RE = 18

# UART0 on Pi 5 with dtparam=uart0=on. /dev/serial0 is an alternative symlink.
_DEFAULT_PORT = "/dev/ttyAMA0"


def _byte_time(baudrate: int) -> float:
    """Transmission time of a single byte (8N1: 10 bits)."""
    return 10.0 / baudrate


class RS485:
    """Half-duplex MAX485 on Pi 5 UART0.

    DE/RE initial_value=False - RX mode by default.

    Strict order in send() per CLAUDE.md:
      de_re.on() -> sleep(0.002) -> write() -> flush()
      -> sleep(BYTE_TIME × len + 0.002) -> de_re.off()
    The delay before de_re.off() is critical - otherwise the last byte is
    lost.
    """

    def __init__(
        self,
        port: str = _DEFAULT_PORT,
        baudrate: int = 9600,
        timeout: float = 0.1,
    ) -> None:
        self._de_re = OutputDevice(_BCM_DE_RE, active_high=True, initial_value=False)
        self._ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self._byte_time = _byte_time(baudrate)
        logger.info(
            "RS485 initialised: %s, %d baud, DE+RE=BCM%d",
            port,
            baudrate,
            _BCM_DE_RE,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, data: bytes) -> None:
        """TX mode: DE/RE HIGH -> write -> flush -> wait for buffer drain -> RX."""
        self._de_re.on()
        time.sleep(0.002)
        self._ser.write(data)
        self._ser.flush()
        time.sleep(self._byte_time * len(data) + 0.002)
        self._de_re.off()
        logger.debug("RS485: sent %d bytes", len(data))

    def recv(self, size: int = 1, timeout: float | None = None) -> bytes:
        """Read from the RX buffer. DE/RE is already LOW - bus is being listened to."""
        if timeout is not None:
            saved = self._ser.timeout
            self._ser.timeout = timeout
            data = self._ser.read(size)
            self._ser.timeout = saved
            return data
        return self._ser.read(size)

    def flush_rx(self) -> None:
        """Reset the input buffer (junk after RX->TX->RX switch)."""
        self._ser.reset_input_buffer()

    def close(self) -> None:
        self._de_re.off()
        self._ser.close()
        self._de_re.close()
        logger.info("RS485: resources released")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> RS485:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
