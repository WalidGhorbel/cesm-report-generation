"""
Ablation test: does the narrative generation actually use the image, or
mostly follow the stated classifier category regardless of what's shown?

Method: hold the CATEGORY artificially fixed ("Suspicious", same
confidence) across two genuinely different real images -- patient 3's
right breast (real finding: a benign dilated duct) and left breast (real
finding: a large malignant mass with a spiculated margin). If the model
is meaningfully reading the image, these two narratives should describe
visibly different things even though both are told the same category. If
the narratives come back interchangeable, that's direct evidence the
narrative mostly follows the given label rather than the pixels.

This directly probes the same limitation compare_report.py already
surfaced anecdotally (patient 3 CESM-left: classifier said Benign,
narrative dutifully described "no enhancement" despite the real image
containing an obvious malignant mass) -- this test makes it a designed,
repeatable check instead of a one-off observation.
"""

from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path

sys.path.insert(0, "src/generation")
from full_report import _generate_narrative
from vision_report import resolve_image_path
from classifier import ClassifierResult

GUIDELINE = """BI-RADS terminology reference: masses are described by shape (oval/round/irregular),
margin (circumscribed/obscured/indistinct/microlobulated/spiculated), and density.
Enhancement on CESM is described as mass or non-mass, with pattern (segmental/ductal/regional/diffuse)."""

FIXED_RESULT = ClassifierResult(
    modality="DM",
    birads_group="Suspicious",
    birads_group_probs={"Benign": 0.02, "Suspicious": 0.95, "Malignant": 0.03},
    cancer=False,
    cancer_prob=0.10,
)

IMAGES_DIR = Path.home() / "data/cdd-cesm/CDD-CESM"


def main():
    img_r = resolve_image_path(IMAGES_DIR, "3", "R", "DM")
    img_l = resolve_image_path(IMAGES_DIR, "3", "L", "DM")

    narrative_r = _generate_narrative(img_r, "DM", FIXED_RESULT, GUIDELINE, None)
    narrative_l = _generate_narrative(img_l, "DM", FIXED_RESULT, GUIDELINE, None)

    print("=" * 100)
    print("Patient 3 RIGHT breast")
    print("  Real finding (from report_parser.py, NOT shown to the model): benign dilated duct")
    print("  Category TOLD to the model (forced, same for both images): Suspicious")
    print(f"  GENERATED: {narrative_r!r}")
    print()
    print("Patient 3 LEFT breast")
    print("  Real finding (from report_parser.py, NOT shown to the model): malignant mass, spiculated margin")
    print("  Category TOLD to the model (forced, same for both images): Suspicious")
    print(f"  GENERATED: {narrative_l!r}")
    print("=" * 100)

    words_r = set(narrative_r.lower().split())
    words_l = set(narrative_l.lower().split())
    overlap = len(words_r & words_l) / max(1, min(len(words_r), len(words_l)))
    print(f"\nWord overlap ratio: {overlap:.0%}")
    if overlap > 0.5:
        print("HIGH overlap -- narratives are largely interchangeable despite very different real")
        print("pathology. This is evidence the narrative follows the STATED CATEGORY, not the image.")
    else:
        print("LOW overlap -- narratives differ substantially despite being told the same category.")
        print("This is evidence the model IS responding to genuine visual differences between images,")
        print("not just parroting the category.")


if __name__ == "__main__":
    main()
