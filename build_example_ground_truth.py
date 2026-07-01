"""
Generates examples/<name>/ground_truth.json from the REAL source report
(docx), using report_parser.py + classifier.py's birads_digit_to_group --
NOT manually transcribed.

Why this matters: hand-typing ground truth risks silently diverging from
the actual validated source (typos, copy-paste errors), and doesn't scale
if more example patients get added later. This script makes adding a new
example a one-command, source-of-truth-derived operation instead.

Usage:
    python3 build_example_ground_truth.py <patient_id> <docx_report_path> <examples_dir>

Example:
    python3 build_example_ground_truth.py 3 \
        ~/data/cdd-cesm/"Medical reports for cases"/P3.docx \
        examples/patient_3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src/ingestion")
sys.path.insert(0, "src/generation")

from report_parser import parse_report  # noqa: E402
from classifier import birads_digit_to_group  # noqa: E402


def build_ground_truth(patient_id: str, docx_path: Path) -> dict:
    rec = parse_report(docx_path)
    if rec.patient_id != patient_id:
        raise ValueError(f"docx patient_id {rec.patient_id!r} does not match requested {patient_id!r}")
    if not rec.is_clean:
        raise ValueError(f"Report has parsing warnings, refusing to build ground truth from it: {rec.warnings}")

    result: dict = {"DM": {}, "CESM": {}}

    for side, breast_dm, birads_dm, breast_cesm in (
        ("R", rec.dm_right, rec.dm_birads_right, rec.cesm_right),
        ("L", rec.dm_left, rec.dm_birads_left, rec.cesm_left),
    ):
        if birads_dm is not None and breast_dm.findings:
            result["DM"][side] = {
                "finding": " ".join(breast_dm.findings),
                "birads": birads_dm,
                "group": birads_digit_to_group(birads_dm),
            }
        if breast_cesm.birads is not None and breast_cesm.findings:
            result["CESM"][side] = {
                "finding": " ".join(breast_cesm.findings),
                "birads": breast_cesm.birads,
                "group": birads_digit_to_group(breast_cesm.birads),
            }

    return result


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    patient_id, docx_path, examples_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    gt = build_ground_truth(patient_id, docx_path)

    examples_dir.mkdir(parents=True, exist_ok=True)
    out_path = examples_dir / "ground_truth.json"
    out_path.write_text(json.dumps(gt, indent=2))
    print(f"Wrote {out_path}\n")
    print(json.dumps(gt, indent=2))
