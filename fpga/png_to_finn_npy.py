#!/usr/bin/env python3
import argparse
import numpy as np
from PIL import Image

def convert_one(img_path: str, out_path: str, thresh: int = 50):
    # Load image
    img = Image.open(img_path)

    # Convert to grayscale and resize
    img = img.convert("L").resize((28, 28))

    # Convert to numpy
    arr = np.array(img, dtype=np.uint8)

    # Invert: black-on-white -> white-on-black
    arr = 255 - arr

    # Binarize to remove gray blur
    arr = (arr > thresh).astype(np.uint8)#values 0 or 1

    # Flatten to (1, 784)
    arr = arr.reshape(1, 784)

    # Save
    np.save(out_path, arr)

    print(f"Saved {out_path}")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print(f"  min/max: {arr.min()} / {arr.max()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert letter PNG/JPEG into FINN FCNN input (1x784)"
    )
    ap.add_argument("--infile", required=True, help="Input image (.png/.jpg)")
    ap.add_argument("--outfile", required=True, help="Output .npy file")
    ap.add_argument("--thresh", type=int, default=50, help="Binarization threshold (0-255)")
    args = ap.parse_args()

    convert_one(args.infile, args.outfile, thresh=args.thresh)

