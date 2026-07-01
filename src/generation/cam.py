"""
Score-CAM extraction, ported EXACTLY from the training notebook
(BIRADS_and_Cancer_Classifiers, cell 23: CAMHooks / score_cam / overlay_cam)
-- not a reimplementation from a generic CAM template. This gives genuine
spatial grounding (where in the image the classifier's own attention is
concentrated), unlike the LLM narrative's free-text location claims, which
were proven unreliable by the ablation test in baseline_eval.py.

Location-to-clinical-language mapping: split into two independently
verified/unverified parts, deliberately kept separate:

- MEDIAL/LATERAL (inner/outer): derived directly from the laterality-aware
  padding logic in classifier.py's pad_to_square, which was pixel-verified
  against real training data earlier this session (0.00 mean diff on
  multiple patients). For side="R", the chest wall is anchored to the
  RIGHT edge of the frame (so low column index = lateral/outer, high
  column index = medial/inner); for side="L" it's the mirror. This
  mapping is a direct, confirmed consequence of already-verified code,
  not a new assumption.

- SUPERIOR/INFERIOR (upper/lower): assumes standard mammography display
  convention (superior anatomy at the top of the frame). This has NOT
  been independently verified this session -- run verify_cam_location()
  against a known case (e.g. patient 40 R, real finding: upper outer
  mass) before trusting this in generated narratives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm

from classifier import ClassifierHead


class CAMHooks:
    """Exact port of the notebook's CAMHooks."""

    def __init__(self, model: ClassifierHead):
        self.activation = None
        self.gradient = None
        self.f_handle = model.backbone.norm_pre.register_forward_hook(self._forward_hook)
        self.b_handle = model.backbone.norm_pre.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activation = out

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def remove(self):
        self.f_handle.remove()
        self.b_handle.remove()


def grad_cam(model: ClassifierHead, x: torch.Tensor, target_class: int | None, binary: bool = False) -> np.ndarray:
    """
    Exact port of the notebook's grad_cam. One forward + one backward pass
    total (vs score_cam's ~C forward passes, one per channel) -- this is
    why it's dramatically faster; not a tuning difference, a different
    algorithm with a different cost profile.

    Unlike the notebook's version (which required the caller to remember
    x.requires_grad_(True) beforehand -- an easy detail to forget), this
    clones the input and sets requires_grad internally, so callers can't
    silently get a broken/zero CAM by forgetting that step.
    """
    x = x.clone().requires_grad_(True)
    hooks = CAMHooks(model)
    model.zero_grad()
    out = model(x)
    score = out[0] if binary else out[0, target_class]
    score.backward()
    acts, grads = hooks.activation, hooks.gradient
    hooks.remove()

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


@torch.no_grad()
def score_cam(model: ClassifierHead, x: torch.Tensor, target_class: int | None, binary: bool = False, batch_size: int = 32) -> np.ndarray:
    """Exact port of the notebook's score_cam. Returns a (H, W) array in
    [0, 1], H/W matching the model input resolution (384x384), NOT the
    original 1024x1024 padded image."""
    hooks = CAMHooks(model)
    _ = model(x)
    acts = hooks.activation.clone()
    hooks.remove()

    C = acts.shape[1]
    acts_up = F.interpolate(acts, size=x.shape[-2:], mode="bilinear", align_corners=False)
    acts_up = acts_up.squeeze(0)
    mins = acts_up.amin(dim=(1, 2), keepdim=True)
    maxs = acts_up.amax(dim=(1, 2), keepdim=True)
    masks = (acts_up - mins) / (maxs - mins + 1e-8)

    weights = torch.zeros(C, device=x.device)
    for i in range(0, C, batch_size):
        chunk = masks[i:i + batch_size].unsqueeze(1)
        masked = x.repeat(chunk.shape[0], 1, 1, 1) * chunk
        out = model(masked)
        weights[i:i + chunk.shape[0]] = out if binary else out[:, target_class]

    weights = F.softmax(weights, dim=0)
    cam = F.relu((weights.view(-1, 1, 1) * acts_up).sum(dim=0))
    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def overlay_cam(pil_img: Image.Image, cam_arr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Exact port of the notebook's overlay_cam -- for visual verification only."""
    img = np.array(pil_img.resize((cam_arr.shape[1], cam_arr.shape[0]))).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    heat = cm.jet(cam_arr)[..., :3]
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)


def compute_cam_overlay(image_path: Path, modality: str, checkpoint_dir: Path, laterality: str, device: str = "cpu") -> Image.Image | None:
    """
    Computes a Grad-CAM overlay for the BI-RADS classifier's predicted
    class on this image. Returns a PIL.Image overlay, or None if the
    checkpoint is missing.

    IMPORTANT, based on manual verification against known-location cases:
    only the SUPERIOR/INFERIOR (upper/lower) axis showed reliable signal
    (2/2 correct). MEDIAL/LATERAL showed a systematic bias toward "medial"
    across all 3 tested cases, likely the chest-wall edge dominating
    gradient signal rather than the actual lesion. This function
    deliberately returns only the RAW VISUAL overlay for a human to judge
    -- it does NOT generate or claim any location text, since the
    horizontal axis specifically is not trustworthy enough for that yet.
    """
    from classifier import load_classifier, preprocess_image, pad_to_square, PAD_SIZE
    import torch

    checkpoint_path = checkpoint_dir / f"best_birads_{modality.lower()}.pt"
    if not checkpoint_path.exists():
        return None

    model = load_classifier(checkpoint_path, num_outputs=3, device=device)
    x = preprocess_image(image_path, laterality=laterality).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(probs.argmax())

    cam_arr = grad_cam(model, x, target_class=pred_idx, binary=False)

    pil_img = Image.open(image_path).convert("L")
    padded = pad_to_square(pil_img, laterality, size=PAD_SIZE).resize((384, 384))
    overlay_arr = overlay_cam(padded, cam_arr)
    return Image.fromarray((overlay_arr * 255).astype("uint8"))



    row_frac: float  # 0=top, 1=bottom of frame
    col_frac: float  # 0=left, 1=right of frame
    medial_lateral: str  # "medial" | "lateral" | "central" -- VERIFIED mapping
    superior_inferior: str  # "upper" | "lower" | "mid"      -- UNVERIFIED mapping, see module docstring

    def describe(self) -> str:
        parts = []
        if self.superior_inferior != "mid":
            parts.append(self.superior_inferior)
        if self.medial_lateral != "central":
            parts.append(self.medial_lateral)
        return " ".join(parts) if parts else "central"


def cam_to_location(cam_arr: np.ndarray, laterality: str, activation_percentile: float = 90.0) -> CamLocation:
    """
    Converts a Score-CAM heatmap into a coarse location description.

    Uses the CENTROID of the top-activation region (pixels above the given
    percentile), not the single argmax pixel -- a centroid is more robust
    to a single noisy hot pixel than an argmax would be.
    """
    threshold = np.percentile(cam_arr, activation_percentile)
    mask = cam_arr >= threshold
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        row_frac, col_frac = 0.5, 0.5
    else:
        weights = cam_arr[rows, cols]
        row_frac = float(np.average(rows, weights=weights)) / cam_arr.shape[0]
        col_frac = float(np.average(cols, weights=weights)) / cam_arr.shape[1]

    # Superior/inferior: simple thirds split on row position (UNVERIFIED, see module docstring)
    if row_frac < 0.4:
        sup_inf = "upper"
    elif row_frac > 0.6:
        sup_inf = "lower"
    else:
        sup_inf = "mid"

    # Medial/lateral: derived directly from the VERIFIED padding convention.
    # R: chest wall anchored at the RIGHT edge (high col) -> low col = lateral, high col = medial.
    # L: chest wall anchored at the LEFT edge (low col) -> low col = medial, high col = lateral.
    if laterality == "R":
        med_lat = "lateral" if col_frac < 0.4 else ("medial" if col_frac > 0.6 else "central")
    else:
        med_lat = "medial" if col_frac < 0.4 else ("lateral" if col_frac > 0.6 else "central")

    return CamLocation(row_frac=row_frac, col_frac=col_frac, medial_lateral=med_lat, superior_inferior=sup_inf)


def verify_cam_location(overlay_out_path: Path, pil_img: Image.Image, cam_arr: np.ndarray) -> None:
    """Saves the CAM overlay as a PNG for visual inspection. Run this
    against a known case (e.g. patient 40 right breast, real finding:
    upper outer mass) BEFORE trusting cam_to_location()'s superior/inferior
    mapping in real narrative generation."""
    overlay = overlay_cam(pil_img, cam_arr)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(overlay_out_path)
    print(f"Saved CAM overlay to {overlay_out_path} -- visually confirm the highlighted region "
          f"matches the known finding location before trusting cam_to_location().")