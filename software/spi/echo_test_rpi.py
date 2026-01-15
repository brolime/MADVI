#SLAVE SPI INTERFACE
# echo_test_rpi.py
import spidev
import time

# SPI setup
spi = spidev.SpiDev()
spi.open(0, 0)  # bus 0, chip select 0
spi.max_speed_hz = 500000
spi.mode = 0b00

# Memory buffer to send
buffer = [10, 20, 30, 40, 50, 60, 70, 80]

# Registers in memory (simulate with a dict)
registers = {
    0x00: 0,  # READY_REG
    0x04: buffer  # BUFFER_REG (list)
}

def read_register(offset):
    if offset == 0x00:
        return registers[0x00]
    elif offset == 0x04:
        return registers[0x04]
    else:
        return 0

def write_register(offset, val):
    if offset == 0x00:
        registers[0x00] = val

print("Raspberry Pi SPI Slave starting...")

# Set ready flag to 1 to indicate data is ready
write_register(0x00, 1)

# Wait until PYNQ clears ready flag
while read_register(0x00) == 1:
    # simulate SPI transaction
    # Normally PYNQ would read BUFFER_REG via SPI
    # Here we just print the buffer
    print("Slave buffer ready:", registers[0x04])
    time.sleep(0.5)

print("Slave done, ready flag cleared by master.")
spi.close()
