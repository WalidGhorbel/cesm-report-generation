"""
Runs generate_full_report + compare_report across multiple patients and
aggregates the category-match rate, instead of eyeballing one patient at a
time. Costs real API credits: roughly 4 LLM narrative calls per bilateral
patient (fewer for single-sided patients), so the default list below is
deliberately small (5 patients) -- widen PATIENT_IDS once you've confirmed
this batch behaves as expected.

Usage:
    python3 run_comparison_batch.py
"""

from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path

sys.path.insert(0, "src/generation")
sys.path.insert(0, "src/ingestion")

from full_report import generate_full_report
from report_parser import parse_report
from compare_report import compare_report

# Deliberately picked for a mix of bilateral/single-sided and benign/suspicious
# cases, distinct from the few-shot-reserved patients used elsewhere (30/31/34/39)
# -- not that it matters for this script (no few-shot examples are used in
# full_report.py), but keeping patient selection consistent avoids confusion.
PATIENT_IDS = ["4", "5", "8", "29", "38"]

REPORTS_DIR = Path.home() / "data/cdd-cesm/Medical reports for cases"
IMAGES_DIR = Path.home() / "data/cdd-cesm/CDD-CESM"
CHECKPOINT_DIR = Path.home() / "data/cdd-cesm/checkpoints"

GUIDELINE = """BI-RADS terminology reference: masses are described by shape (oval/round/irregular),
margin (circumscribed/obscured/indistinct/microlobulated/spiculated), and density.
Enhancement on CESM is described as mass or non-mass, with pattern (segmental/ductal/regional/diffuse)."""


def main():
    total_agree, total_count = 0, 0
    failures = []

    for pid in PATIENT_IDS:
        print(f"\n\n{'#'*100}")
        print(f"# Patient {pid}")
        print(f"{'#'*100}")
        try:
            report_path = REPORTS_DIR / f"P{pid}.docx"
            rec = parse_report(report_path)

            generated = generate_full_report(
                patient_id=pid,
                images_dir=IMAGES_DIR,
                checkpoint_dir=CHECKPOINT_DIR,
                guideline_context=GUIDELINE,
            )

            agree, count = 0, 0
            for modality, gen_r_attr, gen_l_attr in (("DM", "dm_right", "dm_left"), ("CESM", "cesm_right", "cesm_left")):
                for side, gen_attr in (("R", gen_r_attr), ("L", gen_l_attr)):
                    from compare_report import _real_finding_and_group
                    real_text, real_group = _real_finding_and_group(rec, side, modality)
                    gen_section = getattr(generated, gen_attr)
                    if real_group is not None and gen_section is not None:
                        count += 1
                        agree += int(real_group == gen_section.classifier_result.birads_group)

            compare_report(rec, generated)
            total_agree += agree
            total_count += count

        except Exception as e:
            print(f"FAILED for patient {pid}: {e}")
            failures.append((pid, str(e)))

    print(f"\n\n{'='*100}")
    print(f"BATCH SUMMARY across {len(PATIENT_IDS)} patients")
    print(f"{'='*100}")
    if total_count:
        print(f"Overall category-level agreement: {total_agree}/{total_count} ({100*total_agree/total_count:.0f}%)")
    if failures:
        print(f"\n{len(failures)} patient(s) failed to process:")
        for pid, err in failures:
            print(f"  patient {pid}: {err}")


if __name__ == "__main__":
    main()
