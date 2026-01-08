#!/usr/bin/env python3

import argparse
import time
from pynq import Overlay
from pynq.lib.iic import AxiIIC

# -----------------------------
# BME280 Constants
# -----------------------------
BME280_ADDR = 0x77        # change to 0x77 if needed
REG_CTRL_HUM = 0xF2       # humidity control register
REG_CTRL_MEAS = 0xF4      # measurement control register

rx_data = []
# -----------------------------
# Argument parsing
# -----------------------------
parser = argparse.ArgumentParser(description="BME280 I2C R/W test on PYNQ")
parser.add_argument(
    "--bitfile",
    required=True,
    help="Path to bitstream (.bit) file"
)
parser.add_argument(
    "--i2c",
    default=1,
    type=int,
    help="I2C bus index (default: 1)"
)
args = parser.parse_args()

# -----------------------------
# Load FPGA bitstream
# -----------------------------
print(f"[INFO] Loading bitstream: {args.bitfile}")
ol = Overlay(args.bitfile)
ol.download()
print("[OK] Bitstream loaded")

# -----------------------------
# Initialize I2C
# -----------------------------
print(f"[INFO] Opening I2C bus {args.i2c}")

i2c_bus = AxiIIC(ol.ip_dict["axi_iic_0"])

# -----------------------------
# Helper functions
# -----------------------------
def i2c_read_reg(bus, addr, reg, read_data):
    bus.send(addr, [reg], len([reg]),1)
    bus.receive(addr,read_data,1,0)
    return read_data

def i2c_write_reg(bus, addr, reg, value):
    bus.send(addr, bytes([reg, value]), length=2)

# -----------------------------
# Test sequence
# -----------------------------
#print("[INFO] Reading original CTRL_HUM register")
#rx_data = i2c_read_reg(i2c_bus, BME280_ADDR, REG_CTRL_HUM , rx_data)
#print(f"  Original value: 0x{rx_data:02X}")
#
## Write a new value (valid values: 0x00–0x07)
#test_val = [(rx_data + 1) & 0x07]
#print(f"[INFO] Writing test value: 0x{test_val:02X}")
#i2c_write_reg(i2c_bus, BME280_ADDR, REG_CTRL_HUM, test_val)
#
#time.sleep(0.05)
#
#print("[INFO] Reading back register")
#readback_val = i2c_read_reg(i2c_bus, BME280_ADDR, REG_CTRL_HUM)
#print(f"  Readback value: 0x{readback_val:02X}")

iic_data = [0xD0]
read_data = []

i2c_bus.send(0x77,iic_data,len(iic_data),1)
i2c_bus.receive(0x77,rx_data,2,0)

print(rx_data)


# -----------------------------
# Verify
# -----------------------------
#if readback_val == test_val:
#    print("[SUCCESS] Register write verified ✅")
#else:
#    print("[ERROR] Register write FAILED ❌")
#    print(f"Expected 0x{test_val:02X}, got 0x{readback_val:02X}")
#    exit(1)
