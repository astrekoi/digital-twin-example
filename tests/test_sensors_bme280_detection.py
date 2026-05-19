"""BME280Sensor: chip_id auto-detection + BMP280 humidity guard.

Both sensors share I²C 0x76. BMP280 (chip_id 0x58) lacks humidity; BME280
(chip_id 0x60) provides it. Wrong dispatch corrupts data silently - these
tests prevent regression of the BMP280/BME280 conflict that broke the
Reflex-era pipeline.

All tests mock smbus2 (chip_id read) and the adafruit drivers (no real I²C).
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Common test fixtures: stub board/busio so import works on a non-Pi host.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_circuitpython_globals(monkeypatch):
    """board + busio + adafruit drivers are stubbed; tests never touch I²C."""
    fake_board = types.ModuleType("board")
    fake_board.SCL = object()
    fake_board.SDA = object()
    monkeypatch.setitem(sys.modules, "board", fake_board)

    fake_busio = types.ModuleType("busio")
    fake_busio.I2C = MagicMock()
    monkeypatch.setitem(sys.modules, "busio", fake_busio)

    fake_bme = types.ModuleType("adafruit_bme280")
    fake_bme.basic = types.ModuleType("adafruit_bme280.basic")
    fake_bme.basic.Adafruit_BME280_I2C = MagicMock(name="Adafruit_BME280_I2C")
    monkeypatch.setitem(sys.modules, "adafruit_bme280", fake_bme)
    monkeypatch.setitem(sys.modules, "adafruit_bme280.basic", fake_bme.basic)

    fake_bmp = types.ModuleType("adafruit_bmp280")
    fake_bmp.Adafruit_BMP280_I2C = MagicMock(name="Adafruit_BMP280_I2C")
    monkeypatch.setitem(sys.modules, "adafruit_bmp280", fake_bmp)

    yield


# ---------------------------------------------------------------------------
# Detection: chip_id -> driver routing
# ---------------------------------------------------------------------------


def test_chip_id_0x60_routes_to_bme280():
    from src.sensors import bme280

    with patch.object(bme280, "_read_chip_id", return_value=0x60):
        s = bme280.BME280Sensor()
    assert s._sensor_kind == "bme280"
    assert s._chip_id == 0x60


def test_chip_id_0x58_routes_to_bmp280():
    from src.sensors import bme280

    with patch.object(bme280, "_read_chip_id", return_value=0x58):
        s = bme280.BME280Sensor()
    assert s._sensor_kind == "bmp280"
    assert s._chip_id == 0x58


def test_chip_id_unknown_raises_runtime_error():
    from src.sensors import bme280

    with patch.object(bme280, "_read_chip_id", return_value=0xFF):
        with pytest.raises(RuntimeError, match="Unknown chip_id"):
            bme280.BME280Sensor()


def test_forced_kind_overrides_chip_id():
    """sensor_kind="bme280" must NOT call _read_chip_id at all."""
    from src.sensors import bme280

    with patch.object(bme280, "_read_chip_id") as mock_read:
        s = bme280.BME280Sensor(sensor_kind="bme280")
    mock_read.assert_not_called()
    assert s._sensor_kind == "bme280"
    assert s._chip_id == 0x60


# ---------------------------------------------------------------------------
# Read path: BMP280 must NEVER call .relative_humidity
# ---------------------------------------------------------------------------


def test_bmp280_read_never_calls_humidity():
    from src.sensors import bme280

    with patch.object(bme280, "_read_chip_id", return_value=0x58):
        s = bme280.BME280Sensor()
    # The bmp280 driver MagicMock has no real attrs; spy on relative_humidity.
    s._sensor.temperature = 22.5
    s._sensor.pressure = 1013.0
    # NB: do NOT set relative_humidity; if read() touches it, MagicMock would
    # auto-create the attr, but we assert directly via mock_calls list.

    out = s.read()
    assert out["temp_c"] == 22.5
    assert out["humidity_pct"] is None
    assert out["pressure_hpa"] == 1013.0
    assert out["sensor_type"] == "bmp280"
    assert out["chip_id"] == "0x58"

    # Critical: relative_humidity must NOT have been accessed on the BMP280 driver.
    accessed = [
        c for c in s._sensor.mock_calls
        if "relative_humidity" in str(c)
    ]
    assert accessed == [], f"BMP280 driver accessed humidity: {accessed}"


def test_bme280_read_returns_real_humidity():
    from src.sensors import bme280

    with patch.object(bme280, "_read_chip_id", return_value=0x60):
        s = bme280.BME280Sensor()
    s._sensor.temperature = 23.0
    s._sensor.pressure = 1010.5
    s._sensor.relative_humidity = 45.0

    out = s.read()
    assert out["temp_c"] == 23.0
    assert out["humidity_pct"] == 45.0
    assert out["pressure_hpa"] == 1010.5
    assert out["sensor_type"] == "bme280"
    assert out["chip_id"] == "0x60"
