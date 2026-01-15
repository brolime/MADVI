#MASTER SPI INTERFACE
# echo_test_pynq.py
import sys
import time
from pynq import Overlay, MMIO
import numpy as np

# -----------------------------
# Command-line argument parsing
# -----------------------------
if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <path_to_bitfile>")
    sys.exit(1)

bitfile_path = sys.argv[1]

print(f"Loading bitstream from: {bitfile_path}")
ol = Overlay(bitfile_path)

print(ol.ip_dict)
s = ol.axi_quad_spi_0
print(s)

val = s.read(0x60)
print("Initial Register Setup: ")
print(np.binary_repr(val))
print(np.binary_repr(val))
s.write(0x60,0b00_00011110)
val = s.read(0x60)
print("Setup Now: ")
print(np.binary_repr(val))
#select device 0
s.write(0x70,0b1111_1110) #write teh lowest slave low (Active low so this one gets turned on.)
val = s.read(0x70)
print("SSelect: ")

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

