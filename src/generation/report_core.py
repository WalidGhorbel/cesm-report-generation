"""
Core report-generation logic for the Streamlit app, deliberately separated
from any Streamlit import. This module can be unit-tested without a
Streamlit runtime (which this environment doesn't have) and without a
browser -- app.py is a thin UI layer on top of these functions.

Unlike generate_full_report() in full_report.py (which looks up all 4
images for a known patient_id by filename convention), this module is
built for USER-UPLOADED single images: one side (R or L) at a time, an LE
file and/or a DES file, with no assumption that both exist or that a
patient_id/filename convention applies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from classifier import ClassifierResult, Modality, classify_image
from full_report import ACR_DISCLAIMER, FindingSection, PatientReport, _generate_narrative

CATEGORY_COLOR = {
    "Benign": "#1e7d32",       # green
    "Suspicious": "#b8860b",   # amber
    "Malignant": "#b71c1c",    # red
}
CATEGORY_EMOJI = {
    "Benign": "🟢",
    "Suspicious": "🟠",
    "Malignant": "🔴",
}


def category_color(group: str) -> str:
    return CATEGORY_COLOR.get(group, "#666666")


def category_emoji(group: str) -> str:
    return CATEGORY_EMOJI.get(group, "⚪")


@dataclass
class SideResult:
    side: str  # "R" | "L"
    dm: FindingSection | None = None
    cesm: FindingSection | None = None
    errors: list[str] = field(default_factory=list)


def generate_side_report(
    le_path: Path | None,
    des_path: Path | None,
    side: str,
    checkpoint_dir: Path,
    guideline_context: str,
    client=None,
    device: str = "cpu",
) -> SideResult:
    """
    Runs classification + narrative generation for whichever of LE/DES was
    provided for this side. Missing files are skipped, not errored on --
    a user may legitimately only have one modality for a given breast.
    Genuine failures (e.g. a corrupt/unreadable image) are captured in
    .errors rather than raised, so the UI can display a clear message
    instead of crashing.
    """
    result = SideResult(side=side)

    for modality, path in (("DM", le_path), ("CESM", des_path)):
        if path is None:
            continue
        try:
            clf_result = classify_image(path, modality, checkpoint_dir, device=device, laterality=side)
            narrative = _generate_narrative(path, modality, clf_result, guideline_context, client)
            section = FindingSection(modality=modality, classifier_result=clf_result, narrative=narrative)
            if modality == "DM":
                result.dm = section
            else:
                result.cesm = section
        except Exception as e:
            result.errors.append(f"{modality}: {e}")

    return result


def merge_into_report(report: PatientReport, side_result: SideResult) -> PatientReport:
    """Merges a single side's results into an accumulating PatientReport
    (e.g. right breast run now, left breast run later -> one combined report)."""
    if side_result.side == "R":
        if side_result.dm is not None:
            report.dm_right = side_result.dm
        if side_result.cesm is not None:
            report.cesm_right = side_result.cesm
    else:
        if side_result.dm is not None:
            report.dm_left = side_result.dm
        if side_result.cesm is not None:
            report.cesm_left = side_result.cesm
    return report


def report_is_empty(report: PatientReport) -> bool:
    return all(
        getattr(report, attr) is None
        for attr in ("dm_right", "dm_left", "cesm_right", "cesm_left")
    )
