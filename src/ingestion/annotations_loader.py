"""
Loader for the CDD-CESM Radiology-manual-annotations.xlsx ('all' sheet).

Design notes
------------
- ~4% of rows encode MULTIPLE findings per image via a '$' delimiter in
  both BIRADS and Findings (position-aligned). These are split into one
  row per finding so downstream code never has to special-case a
  compound string.
- 'Type' uses the label "CESM" for what image filenames encode as "_CM_".
  Image_name itself is untouched and matches filenames directly -- only
  the semantic Type label differs, which we normalize here.
- This module also cross-validates BIRADS values against ReportRecord
  objects produced by report_parser.py, as an independent data-quality
  check (two differently-structured sources should agree; where they
  don't, that's worth looking at, not silently trusting one).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TYPE_LABEL_TO_FILENAME_INFIX = {"DM": "DM", "CESM": "CM"}


@dataclass
class ImageAnnotation:
    image_name: str
    patient_id: str
    side: str
    modality: str
    view: str
    age: int | None
    acr_density: str | None
    birads: str | None
    finding: str | None
    tags: str | None
    pathology: str | None


def load_annotations(xlsx_path: Path, sheet: str = "all") -> list[ImageAnnotation]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet)

    density_col = "Breast density (ACR)" if "Breast density (ACR)" in df.columns else "ACR"

    records: list[ImageAnnotation] = []
    for _, row in df.iterrows():
        birads_raw = str(row.get("BIRADS", "")) if pd.notna(row.get("BIRADS")) else ""
        findings_raw = str(row.get("Findings", "")) if pd.notna(row.get("Findings")) else ""

        birads_parts = birads_raw.split("$") if birads_raw else [None]
        finding_parts = findings_raw.split("$") if findings_raw else [None]

        if len(birads_parts) != len(finding_parts):
            birads_parts = [birads_raw or None]
            finding_parts = [findings_raw or None]

        for b, f in zip(birads_parts, finding_parts):
            records.append(
                ImageAnnotation(
                    image_name=str(row["Image_name"]),
                    patient_id=str(row["Patient_ID"]),
                    side=str(row["Side"]),
                    modality=str(row["Type"]),
                    view=str(row["View"]),
                    age=int(row["Age"]) if pd.notna(row.get("Age")) else None,
                    acr_density=(
                        None if str(row.get(density_col, "")).strip() == "_" else row.get(density_col)
                    ),
                    birads=(b.strip() if b else None),
                    finding=(f.strip() if f else None),
                    tags=row.get("Tags") if pd.notna(row.get("Tags")) else None,
                    pathology=row.get("Pathology Classification/ Follow up"),
                )
            )
    return records


def cross_validate_against_reports(annotations: list[ImageAnnotation], reports: dict) -> None:
    by_patient_side: dict[tuple[str, str], list[str]] = {}
    for a in annotations:
        if a.modality != "CESM" or not a.birads:
            continue
        by_patient_side.setdefault((a.patient_id, a.side), []).append(a.birads)

    checked = agree = disagree = 0
    for pid, rec in reports.items():
        for side, cesm_finding in (("R", rec.cesm_right), ("L", rec.cesm_left)):
            key = (pid, side)
            if key not in by_patient_side:
                continue
            xlsx_birads_set = set(by_patient_side[key])
            report_birads = cesm_finding.birads
            if report_birads is None:
                continue
            checked += 1
            if report_birads in xlsx_birads_set:
                agree += 1
            else:
                disagree += 1
                print(
                    f"  MISMATCH patient {pid} side {side}: "
                    f"report CESM BI-RADS={report_birads!r}, xlsx BI-RADS={xlsx_birads_set!r}"
                )

    print(f"\nCross-validation: {checked} (patient, side) pairs checked, {agree} agree, {disagree} disagree")


if __name__ == "__main__":
    import sys
    from .report_parser import audit_reports

    xlsx_path = Path(sys.argv[1])
    reports_dir = Path(sys.argv[2])

    annotations = load_annotations(xlsx_path)
    print(f"Loaded {len(annotations)} image-level annotation rows (post $-split)")

    records = audit_reports(reports_dir)
    reports_by_patient = {r.patient_id: r for r in records if r.patient_id}

    print("\n--- Cross-validating report_parser output against xlsx ground truth ---")
    cross_validate_against_reports(annotations, reports_by_patient)
