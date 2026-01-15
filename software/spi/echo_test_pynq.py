#MASTER SPI INTERFACE
# echo_test_pynq.py
import sys
import time
from pynq import Overlay, MMIO

# -----------------------------
# Command-line argument parsing
# -----------------------------
if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <path_to_bitfile>")
    sys.exit(1)

bitfile_path = sys.argv[1]

print(f"Loading bitstream from: {bitfile_path}")
ol = Overlay(bitfile_path)

# -----------------------------
# Memory-mapped registers
# -----------------------------
# Let's assume the AXI IP is at this base address
spi_mmio = MMIO(0x40000000, 0x1000)  # adjust if different

# Register offsets
READY_REG = 0x00   # ready flag from slave
BUFFER_REG = 0x04  # data register
BUFFER_LEN = 8     # number of 32-bit words in buffer

print("PYNQ SPI Master starting...")

# -----------------------------
# Poll READY_REG until 1
# -----------------------------
while True:
    ready = spi_mmio.read(READY_REG)
    if ready == 1:
        print("Slave ready! Reading buffer...")
        data = []
        for i in range(BUFFER_LEN):
            val = spi_mmio.read(BUFFER_REG + i*4)
            data.append(val)
        print("Data read from slave:", data)

        # Clear ready flag
        spi_mmio.write(READY_REG, 0)
        break
    time.sleep(0.01)  # small delay to avoid busy wait

print("PYNQ Master done.")

