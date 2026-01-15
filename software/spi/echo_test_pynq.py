#MASTER SPI INTERFACE
# echo_test_pynq.py
from pynq import Overlay, MMIO
import time

# Load overlay with AXI SPI / register interface
ol = Overlay("axi_spi.bit")  # Replace with your actual bitstream
# Let's say your AXI slave address space starts at 0x40000000
spi_mmio = MMIO(0x40000000, 0x1000)  # size 0x1000 bytes

# Register offsets (all in bytes)
READY_REG = 0x00   # ready flag from slave
BUFFER_REG = 0x04  # data register

# Number of data elements to read
BUFFER_LEN = 8

print("PYNQ SPI Master starting...")

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
