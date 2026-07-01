"""
Pixel-diffs our reimplementation of the preprocessing pipeline (in
classifier.py: pad_to_square + resize to 1024) against a REAL preprocessed
sample copied down from the training Drive folder, for the same source
image. This settles definitively whether the preprocessing reimplementation
matches what the model was actually trained/evaluated on, rather than
relying on a visual "looks about right" check.

Usage:
    python3 compare_preprocessing.py <raw_jpg_path> <real_preprocessed_png_path> <laterality: L|R>
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def pad_to_square(img: Image.Image, laterality: str, size: int = 1024) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 0)
    paste_x = 0 if laterality == "L" else (side - w)
    canvas.paste(img, (paste_x, 0))
    return canvas.resize((size, size), Image.BILINEAR)


def compare(raw_jpg: Path, real_preprocessed: Path, laterality: str, out_path: Path) -> None:
    raw = Image.open(raw_jpg).convert("L")
    ours = pad_to_square(raw, laterality, size=1024)

    real = Image.open(real_preprocessed).convert("L")
    if real.size != ours.size:
        print(f"SIZE MISMATCH: ours={ours.size}, real={real.size}")
        print("Resizing 'real' to match ours for a pixel diff -- but this size difference")
        print("itself is likely meaningful and worth noting, not just resolved away.")
        real_for_diff = real.resize(ours.size, Image.BILINEAR)
    else:
        print(f"Sizes match: {ours.size}")
        real_for_diff = real

    ours_arr = np.array(ours, dtype=np.float32)
    real_arr = np.array(real_for_diff, dtype=np.float32)

    diff = np.abs(ours_arr - real_arr)
    print(f"\nPixel diff stats (0-255 scale):")
    print(f"  mean abs diff: {diff.mean():.2f}")
    print(f"  max abs diff:  {diff.max():.2f}")
    print(f"  % pixels with diff > 10:  {100 * (diff > 10).mean():.1f}%")
    print(f"  % pixels with diff > 50:  {100 * (diff > 50).mean():.1f}%")

    if diff.mean() < 2:
        print("\n  -> VERDICT: near-identical. Preprocessing reimplementation is correct.")
    elif diff.mean() < 15:
        print("\n  -> VERDICT: minor differences (likely just interpolation rounding). Probably fine.")
    else:
        print("\n  -> VERDICT: substantial difference. Preprocessing reimplementation does NOT")
        print("     match the real pipeline -- this is very likely the bug. Do not trust")
        print("     classifier predictions until this is resolved.")

    combined = Image.new("L", (ours.size[0] * 2 + 20, ours.size[1] + 40), 255)
    combined.paste(ours, (0, 40))
    combined.paste(real_for_diff, (ours.size[0] + 20, 40))
    draw = ImageDraw.Draw(combined)
    draw.text((10, 10), "OURS (reimplemented)", fill=0)
    draw.text((ours.size[0] + 30, 10), "REAL (from Drive)", fill=0)
    combined.save(out_path)
    print(f"\nSaved side-by-side comparison to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    compare(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path("/tmp/preprocessing_comparison.png"))
