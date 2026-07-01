"""
Confirms (or refutes) whether Grad-CAM's peak attention reliably aligns
with the real lesion location, using 2 more known-location cases beyond
the first (patient 40 R, which showed a mismatch). If these also mismatch,
that's a real pattern -- Grad-CAM location doesn't reliably correspond to
lesion location for this model -- not a fluke from one case.

Usage:
    python3 cam_verify_batch.py
"""

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import sys

sys.path.insert(0, "src/generation")

import torch
from PIL import Image

from classifier import load_classifier, preprocess_image, pad_to_square, PAD_SIZE
from vision_report import resolve_image_path
from cam import grad_cam, cam_to_location, verify_cam_location

IMAGES_DIR = Path.home() / "data/cdd-cesm/CDD-CESM"
CHECKPOINT_DIR = Path.home() / "data/cdd-cesm/checkpoints"

# (patient_id, side, modality, real_location_description)
CASES = [
    ("28", "R", "DM", "upper outer"),
    ("35", "L", "DM", "lower inner"),
]

model = load_classifier(CHECKPOINT_DIR / "best_birads_dm.pt", num_outputs=3)

for patient_id, side, modality, real_location in CASES:
    img_path = resolve_image_path(IMAGES_DIR, patient_id, side, modality)
    x = preprocess_image(img_path, laterality=side)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(probs.argmax())

    cam_arr = grad_cam(model, x, target_class=pred_idx, binary=False)
    loc = cam_to_location(cam_arr, laterality=side)

    print(f"=== Patient {patient_id} {side} {modality} ===")
    print(f"  Predicted class index: {pred_idx}")
    print(f"  CAM-derived location:  {loc.describe()} (row_frac={loc.row_frac:.2f}, col_frac={loc.col_frac:.2f})")
    print(f"  Real ground truth:     {real_location}")
    print()

    pil_img = Image.open(img_path).convert("L")
    padded = pad_to_square(pil_img, side, size=PAD_SIZE).resize((384, 384))
    out_path = Path(f"/tmp/cam_verify_p{patient_id}_{side}.png")
    verify_cam_location(out_path, padded, cam_arr)
    print()
