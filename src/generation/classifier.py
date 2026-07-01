"""
Inference wrapper for the 4 trained classifiers (BIRADS-group + cancer,
each for DM and CESM), producing structured JSON to ground report generation.

Design notes
------------
- Preprocessing here is an EXACT replication of two notebooks:
  1. Virtual_CE_Preprocessing: raw JPG -> grayscale -> laterality-aware
     pad-to-square (chest wall anchored to a consistent edge) -> resize to
     1024x1024, BILINEAR. (Note: that notebook's markdown claims Lanczos
     interpolation, but the actual code uses BILINEAR -- this replicates
     the code, not the comment, since a mismatch here would silently feed
     the model out-of-distribution input.)
  2. BIRADS_and_Cancer_Classifiers: resize 1024 -> 384 (cfg.IMAGE_SIZE),
     convert to RGB (replicates grayscale into 3 channels for the
     ImageNet-pretrained backbone), ToTensor, Normalize with ImageNet
     mean/std.
  Verified against real sample images (laterality-aware padding produces
  the expected chest-wall-anchored pixel distribution) -- see the
  accompanying test run. NOT yet verified against a live checkpoint,
  since no .pt files or torch/timm were available in the environment this
  was written in. Run sanity_check() first on real checkpoints before
  trusting predictions.

- The BIRADS head predicts a COARSE 3-way group (GROUP3 = {1:0, 2:0, 3:1,
  4:1, 5:2, 6:2} -> Benign/Suspicious/Malignant), NOT the individual
  BIRADS digit. This is a genuinely different granularity than the
  `birads_bucket()` used in baseline_eval.py (which keeps 3 as its own
  bucket, and groups 4-6 together). birads_digit_to_group() below converts
  ground-truth BIRADS digits into this classifier's group space so the two
  can be compared fairly -- do not compare group predictions directly
  against baseline_eval's bucket() output, they are not the same grouping.

- The cancer head is a separate binary model (Pathology == "Malignant"),
  not derived from the BIRADS head -- the two can and sometimes will
  disagree (e.g. BIRADS group "Suspicious" but cancer model says
  not-cancer), which is itself useful signal to surface, not an error to
  reconcile away.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

try:
    import timm
except ImportError as e:
    raise ImportError("timm is required: uv add timm torch torchvision") from e

Modality = Literal["DM", "CESM"]
Laterality = Literal["L", "R"]

BACKBONE = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"
PAD_SIZE = 1024
IMAGE_SIZE = 384

BIRADS_GROUP_NAMES = ["Benign", "Suspicious", "Malignant"]
GROUP3 = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def birads_digit_to_group(birads: str) -> str:
    """Converts a ground-truth BIRADS value (e.g. '4A', '2') into this
    classifier's 3-way group space, for fair comparison against
    predictions. Strips any A/B/C subtype suffix before mapping."""
    digit = int(birads[0])
    return BIRADS_GROUP_NAMES[GROUP3[digit]]


# ---------------------------------------------------------------------------
# Model architecture -- exact copy of ClassifierHead from the training
# notebook. Must match exactly or state_dict loading will fail or silently
# load into the wrong shapes.
# ---------------------------------------------------------------------------
class ClassifierHead(nn.Module):
    def __init__(self, backbone_name: str = BACKBONE, num_outputs: int = 3, drop_rate: float = 0.1):
        super().__init__()
        # pretrained=True to exactly match training-time construction (cell 8 of the
        # training notebook), even though load_state_dict() below overwrites every
        # matching weight regardless -- matching this removes one more variable rather
        # than assuming the difference is harmless.
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, drop_rate=drop_rate)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(drop_rate), nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        out = self.head(self.backbone(x))
        return out.squeeze(-1) if out.shape[-1] == 1 else out


def load_classifier(checkpoint_path: Path, num_outputs: int, device: str = "cpu") -> ClassifierHead:
    """
    Note: this downloads ImageNet-pretrained backbone weights on first call
    (matching training-time construction exactly), which are then fully
    overwritten by the checkpoint's state_dict -- slightly wasteful but
    removes any doubt about architecture/init parity with training.
    """
    model = ClassifierHead(num_outputs=num_outputs).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Preprocessing -- exact replication of the two source notebooks
# ---------------------------------------------------------------------------
def _get_laterality_from_filename(fname: str) -> Laterality:
    for part in fname.split("_"):
        if part in ("L", "R"):
            return part  # type: ignore[return-value]
    raise ValueError(f"No laterality tag (L/R) in filename: {fname}")


def pad_to_square(img: Image.Image, laterality: Laterality, size: int = PAD_SIZE) -> Image.Image:
    """Exact replication of Virtual_CE_Preprocessing's pad_to_square.
    img must already be 'L' mode (grayscale)."""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 0)
    paste_x = 0 if laterality == "L" else (side - w)
    canvas.paste(img, (paste_x, 0))
    return canvas.resize((size, size), Image.BILINEAR)


_normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])


def preprocess_image(image_path: Path, laterality: Laterality | None = None) -> torch.Tensor:
    """
    Full pipeline: raw image -> model input tensor (1, 3, 384, 384).
    laterality: if None, inferred from the filename (expects the standard
    P{id}_{L|R}_{modality}_{view}.jpg naming used throughout this project).
    """
    if laterality is None:
        laterality = _get_laterality_from_filename(image_path.name)

    img = Image.open(image_path).convert("L")
    img = pad_to_square(img, laterality, size=PAD_SIZE)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR).convert("RGB")
    x = _normalize(img)
    return x.unsqueeze(0)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@dataclass
class ClassifierResult:
    modality: Modality
    birads_group: str
    birads_group_probs: dict[str, float]
    cancer: bool
    cancer_prob: float

    def to_json_dict(self) -> dict:
        return {
            "modality": self.modality,
            "birads_group": self.birads_group,
            "birads_group_probs": {k: round(v, 4) for k, v in self.birads_group_probs.items()},
            "cancer": self.cancer,
            "cancer_prob": round(self.cancer_prob, 4),
        }


@torch.no_grad()
def classify_image(
    image_path: Path,
    modality: Modality,
    checkpoint_dir: Path,
    device: str = "cpu",
    laterality: Laterality | None = None,
) -> ClassifierResult:
    x = preprocess_image(image_path, laterality).to(device)

    birads_ckpt = checkpoint_dir / f"best_birads_{modality.lower()}.pt"
    cancer_ckpt = checkpoint_dir / f"best_cancer_{modality.lower()}.pt"

    birads_model = load_classifier(birads_ckpt, num_outputs=3, device=device)
    logits = birads_model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    group_idx = int(probs.argmax())

    cancer_model = load_classifier(cancer_ckpt, num_outputs=1, device=device)
    logit = cancer_model(x)
    cancer_prob = torch.sigmoid(logit).item()

    return ClassifierResult(
        modality=modality,
        birads_group=BIRADS_GROUP_NAMES[group_idx],
        birads_group_probs={name: float(p) for name, p in zip(BIRADS_GROUP_NAMES, probs)},
        cancer=cancer_prob > 0.5,
        cancer_prob=float(cancer_prob),
    )


# ---------------------------------------------------------------------------
# Sanity check -- run this FIRST on real checkpoints before trusting
# anything else in this module. Uses patient 3, whose ground truth we've
# independently validated multiple times this session:
#   right breast: DM/CESM BIRADS 2  -> group "Benign"
#   left breast:  DM/CESM BIRADS 5  -> group "Malignant"
# If this doesn't come back roughly right, something in checkpoint loading
# or preprocessing is broken -- fix that before running anything at scale.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Diagnostics -- run these FIRST when predictions look wrong. A systematic
# collapse to one class with high confidence on cases with very different
# ground truth is a much stronger signal of a preprocessing/checkpoint bug
# than of genuine model inaccuracy, which usually looks like spread-out
# errors rather than confident agreement on the wrong answer.
# ---------------------------------------------------------------------------
def inspect_checkpoint(checkpoint_path: Path, device: str = "cpu") -> None:
    """Prints whatever metrics were saved alongside the weights, to sanity
    check this is a real, reasonably-trained checkpoint and not something
    corrupted or barely-trained. Compare against what your training run
    actually printed for that model's best epoch."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"{checkpoint_path.name}:")
    print(f"  keys in checkpoint: {list(ckpt.keys())}")
    print(f"  epoch: {ckpt.get('epoch')}")
    state_dict = ckpt["model"]
    other_keys = [k for k in ckpt if k not in ("model", "epoch")]
    for k in other_keys:
        v = ckpt[k]
        # numpy scalars (e.g. from roc_auc_score) have a .shape attribute
        # even though they're not arrays -- check ndim/size instead of
        # just hasattr(v, "shape") to avoid silently hiding them
        is_scalar = not hasattr(v, "shape") or getattr(v, "size", 1) == 1
        print(f"  {k}: {float(v) if is_scalar else v}")
    n_params = sum(t.numel() for t in state_dict.values())
    print(f"  state_dict: {len(state_dict)} tensors, {n_params/1e6:.2f}M params total")


def save_preprocessed_for_inspection(image_path: Path, laterality: Laterality, out_path: Path) -> None:
    """
    Un-normalizes the exact tensor that would be fed to the model and
    saves it as a viewable PNG. If this image doesn't look like a
    recognizable mammogram (blank, garbled, wrong orientation), the bug
    is in preprocessing, not the model or checkpoint.
    """
    x = preprocess_image(image_path, laterality).squeeze(0)  # (3, 384, 384), normalized
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    unnorm = (x * std + mean).clamp(0, 1)
    img = transforms.ToPILImage()(unnorm)
    img.save(out_path)
    print(f"Saved preprocessed input to {out_path} -- open it and confirm it looks like a real mammogram")


def sanity_check(
    images_dir: Path,
    checkpoint_dir: Path,
    device: str = "cpu",
    cases: list[tuple[str, str, str, bool]] | None = None,
) -> None:
    """
    cases: list of (patient_id, side, expected_birads_group, expected_cancer).
    Defaults to patient 3 (right=Benign/BIRADS2, left=Malignant/BIRADS5),
    ground truth independently validated multiple times earlier this
    session -- but any patient with known, ideally MIXED (not same class
    on both sides) ground truth works just as well here.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from vision_report import resolve_image_path  # reuse the same image-finding logic

    if cases is None:
        cases = [("3", "R", "Benign", False), ("3", "L", "Malignant", True)]

    for patient_id, side, expected_group, expected_cancer in cases:
        img_path = resolve_image_path(images_dir, patient_id, side, "DM")
        result = classify_image(img_path, "DM", checkpoint_dir, device=device, laterality=side)
        match = "OK" if result.birads_group == expected_group else "MISMATCH"
        print(
            f"patient {patient_id} {side}: predicted={result.birads_group!r} "
            f"(expected {expected_group!r}) [{match}], cancer={result.cancer} "
            f"(expected {expected_cancer}), probs={result.birads_group_probs}"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python -m src.generation.classifier <images_dir> <checkpoint_dir>")
        sys.exit(1)

    images_dir, checkpoint_dir = Path(sys.argv[1]), Path(sys.argv[2])

    # Patient 40: DM right = BIRADS 4 ("suspicious looking mass") -> Suspicious,
    # DM left = BIRADS 2 ("benign IMLN") -> Benign. Mixed ground truth (not the
    # same class on both sides), independently validated via report_parser.py
    # earlier this session.
    TEST_CASES = [("40", "R", "Suspicious", False), ("40", "L", "Benign", False)]

    print("=== Checkpoint inspection ===")
    for fname in ("best_birads_dm.pt", "best_cancer_dm.pt", "best_birads_cesm.pt", "best_cancer_cesm.pt"):
        path = checkpoint_dir / fname
        if path.exists():
            inspect_checkpoint(path)
        else:
            print(f"{fname}: MISSING at {path}")
        print()

    print("=== Preprocessed input inspection ===")
    sys.path.insert(0, str(Path(__file__).parent))
    from vision_report import resolve_image_path

    for patient_id, side, _, _ in TEST_CASES:
        img_path = resolve_image_path(images_dir, patient_id, side, "DM")
        out_path = Path(f"/tmp/preprocessed_p{patient_id}_{side}.png")
        save_preprocessed_for_inspection(img_path, side, out_path)

    print("\n=== Prediction sanity check ===")
    sanity_check(images_dir, checkpoint_dir, cases=TEST_CASES)