import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
input_dir = "letter_crops_v2"          # EAST output directory
output_dir = "letter_crops_v2_npy"     # FINN-ready output
img_size = (28, 28)
dtype = np.uint8

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# TRANSFORM FUNCTION
# -----------------------------
def transform(arr, bg_threshold=120, darken_factor=1.0):
    """
    - Forces background to white (255)
    - Keeps dark foreground
    - Input/Output range: [0, 255]
    """
    arr = arr.copy()

    # Force background to white
    arr[arr > bg_threshold] = 255

    # Optional contrast scaling
    arr = 255 - (255 - arr) * darken_factor
    arr = np.clip(arr, 0, 255)

    return arr

# -----------------------------
# CONVERSION
# -----------------------------
for fname in sorted(os.listdir(input_dir)):
    if not fname.lower().endswith(".png"):
        continue

    png_path = os.path.join(input_dir, fname)

    # Load original image
    img_orig = Image.open(png_path).convert("L")

    # Resize to 28x28
    img = img_orig.resize(img_size)

    # Convert to numpy
    arr = np.array(img, dtype=dtype)

    # Apply transform
    arr = transform(arr)

    # Binarize (match FINN training)
    arr = (arr < 128).astype(np.uint8)

    # Flatten for FINN: (1, 784)
    arr = arr.reshape(1, 28 * 28)

    # Save NPY
    npy_name = os.path.splitext(fname)[0] + ".npy"
    np.save(os.path.join(output_dir, npy_name), arr)

    print(
        f"Saved {npy_name} | shape={arr.shape} | "
        f"dtype={arr.dtype} | min={arr.min()} max={arr.max()}"
    )
