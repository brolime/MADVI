#!/usr/bin/env python3

import argparse
import time
from pynq import Overlay
from pynq.lib.iic import AxiIIC

# -----------------------------
# BME280 Constants
# -----------------------------
BME280_ADDR = 0x76        # change to 0x77 if needed
REG_CTRL_HUM = 0xF2       # humidity control register
REG_CTRL_MEAS = 0xF4      # measurement control register

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
print(ol.ip_dict)
i2c_bus = AxiIIC(ol.ip_dict["axi_iic_0"])

# -----------------------------
# Helper functions
# -----------------------------
def i2c_read_reg(bus, addr, reg):
    bus.send(addr, bytes([reg]), length=1)
    return bus.receive(addr, 1)[0]

def i2c_write_reg(bus, addr, reg, value):
    bus.send(addr, bytes([reg, value]), length=2)

# -----------------------------
# Test sequence
# -----------------------------
print("[INFO] Reading original CTRL_HUM register")
orig_val = i2c_read_reg(i2c_bus, BME280_ADDR, REG_CTRL_HUM)
print(f"  Original value: 0x{orig_val:02X}")

# Write a new value (valid values: 0x00–0x07)
test_val = (orig_val + 1) & 0x07
print(f"[INFO] Writing test value: 0x{test_val:02X}")
i2c_write_reg(i2c_bus, BME280_ADDR, REG_CTRL_HUM, test_val)

time.sleep(0.05)

print("[INFO] Reading back register")
readback_val = i2c_read_reg(i2c_bus, BME280_ADDR, REG_CTRL_HUM)
print(f"  Readback value: 0x{readback_val:02X}")

# -----------------------------
# Verify
# -----------------------------
if readback_val == test_val:
    print("[SUCCESS] Register write verified ✅")
else:
    print("[ERROR] Register write FAILED ❌")
    print(f"Expected 0x{test_val:02X}, got 0x{readback_val:02X}")
    exit(1)
