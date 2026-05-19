"""Pi 5 compatibility patch.

Replaces wiringpi.I2C with smbus2.SMBus (wiringpi does not work on the Pi 5
RP1 chip). The public GpioExpander API is unchanged.
"""
from smbus2 import SMBus


class GpioExpander(object):
    """Drive the Troyka HAT I2C expander pins (STM32F030F4P6).

    Pins are labeled "Analog IO" on the board.
    """

    UID = 0x00
    RESET = 0x01
    CHANGE_I2C_ADDR = 0x02
    SAVE_I2C_ADDR = 0x03
    INPUT = 0x04
    INPUT_PULLUP = 0x05
    INPUT_PULLDOWN = 0x06
    OUTPUT = 0x07
    DIGITAL_READ = 0x08
    DIGITAL_WRITE_HIGH = 0x09
    DIGITAL_WRITE_LOW = 0x0A
    ANALOG_WRITE = 0x0B
    ANALOG_READ = 0x0C
    PWM_FREQ = 0x0D
    ADC_SPEED = 0x0E
    MASTER_READED_UID = 0x0F
    CHANGE_I2C_ADDR_IF_UID_OK = 0x10
    SAY_SLOT = 0x11
    ADC_LOWPASS_FILTER_ON = 0x20
    ADC_LOWPASS_FILTER_OFF = 0x21
    ADC_AS_DIGITAL_PORT_SET_TRESHOLD = 0x22
    ADC_AS_DIGITAL_PORT_READ = 0x23

    def __init__(self, i2c_address, bus_number=1):
        self._addr = i2c_address
        self._bus = SMBus(bus_number)

    # --- low-level replacements for the wiringpi methods ---

    def _writeReg16(self, reg, data):
        """Equivalent of wiringpi.writeReg16: write a 16-bit word to register
        reg. Byte order matches wiringpi (little-endian)."""
        low = data & 0xFF
        high = (data >> 8) & 0xFF
        self._bus.write_i2c_block_data(self._addr, reg, [low, high])

    def _readReg16(self, reg):
        """Equivalent of wiringpi.readReg16."""
        data = self._bus.read_i2c_block_data(self._addr, reg, 2)
        return data[0] | (data[1] << 8)

    def _write(self, reg):
        """Equivalent of wiringpi.write with a single command byte."""
        self._bus.write_byte(self._addr, reg)

    # --- public API (unchanged) ---

    def pinMode(self, pin, mode):
        data = self._reverse_uint16(1 << pin)
        self._writeReg16(mode, data)

    def digitalRead(self, pin):
        mask = 1 << pin
        return 1 if (self._digitalReadPort() & mask) else 0

    def digitalWrite(self, pin, value):
        data = self._reverse_uint16(1 << pin)
        state = (
            GpioExpander.DIGITAL_WRITE_HIGH if value
            else GpioExpander.DIGITAL_WRITE_LOW
        )
        self._writeReg16(state, data)

    def analogRead(self, pin):
        return self._analogRead16(pin) / 4095.0

    def analogWrite(self, pin, value):
        value = int(value * 255)
        data = (pin & 0xFF) | ((value & 0xFF) << 8)
        self._writeReg16(GpioExpander.ANALOG_WRITE, data)

    def changeAddress(self, newAddress):
        self._writeReg16(GpioExpander.CHANGE_I2C_ADDR, newAddress)

    def saveAddress(self):
        self._write(GpioExpander.SAVE_I2C_ADDR)

    # --- internal helpers ---

    def _reset(self):
        self._write(GpioExpander.RESET)

    def _reverse_uint16(self, data):
        return ((data & 0xFF) << 8) | ((data >> 8) & 0xFF)

    def _digitalReadPort(self):
        return self._reverse_uint16(
            self._readReg16(GpioExpander.DIGITAL_READ)
        )

    def _digitalWritePort(self, value):
        value = self._reverse_uint16(value)
        self._writeReg16(GpioExpander.DIGITAL_WRITE_HIGH, value)
        self._writeReg16(GpioExpander.DIGITAL_WRITE_LOW, ~value & 0xFFFF)

    def _analogRead16(self, pin):
        self._writeReg16(GpioExpander.ANALOG_READ, pin)
        return self._reverse_uint16(
            self._readReg16(GpioExpander.ANALOG_READ)
        )

    def _setPwmFreq(self, freq):
        self._writeReg16(
            GpioExpander.PWM_FREQ, self._reverse_uint16(freq)
        )
