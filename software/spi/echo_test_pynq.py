# mmio_interactive.py
import sys
from pynq import Overlay, MMIO
import numpy as np

# -----------------------------
# Check for bitfile argument
# -----------------------------
if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <path_to_bitfile>")
    sys.exit(1)

bitfile_path = sys.argv[1]
print(f"Loading bitstream: {bitfile_path}")
ol = Overlay(bitfile_path)

# -----------------------------
# Interactive setup
# -----------------------------
while True:
    try:
        mmio_base = input("Enter MMIO base address (hex, e.g., 0x40000000): ").strip()
        mmio_base = int(mmio_base, 16)
        mmio_range = input("Enter MMIO range (hex, e.g., 0x1000): ").strip()
        mmio_range = int(mmio_range, 16)
        break
    except ValueError:
        print("Invalid hex value. Please try again.")

mmio = MMIO(mmio_base, mmio_range)
print(f"MMIO initialized: base=0x{mmio_base:X}, range=0x{mmio_range:X}\n")

# -----------------------------
# Interactive read/write loop
# -----------------------------
while True:
    action = input("Enter action (read/write/exit): ").strip().lower()
    if action == "exit":
        print("Exiting program.")
        break
    elif action not in ["read", "write"]:
        print("Invalid action. Please enter 'read', 'write', or 'exit'.")
        continue

    try:
        reg_offset = input("Enter register offset (hex, e.g., 0x04): ").strip()
        reg_offset = int(reg_offset, 16)

        if action == "write":
            value = input("Enter value to write (hex, e.g., 0x12345678): ").strip()
            value = int(value, 16)
            mmio.write(reg_offset, value)
            print(f"Wrote 0x{value:08X} to register 0x{reg_offset:X}")
            val = mmio.read(reg_offset)
            print(f"Verify readback: 0x{val:08X}")
            print("Binary:", np.binary_repr(val, width=32))
        else:  # read
            val = mmio.read(reg_offset)
            print(f"Read 0x{val:08X} from register 0x{reg_offset:X}")
            print("Binary:", np.binary_repr(val, width=32))

        print("-" * 40)

    except ValueError:
        print("Invalid hex input. Try again.")
    except Exception as e:
        print(f"Error during MMIO access: {e}")
