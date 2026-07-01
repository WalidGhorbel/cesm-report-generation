"""
Structured parser + auditor for CDD-CESM style radiology reports (.docx).

Design rationale
-----------------
Case reports and general clinical literature (guidelines) are fundamentally
different document types and must not be chunked the same way:

- Reports are semi-structured records (patient / laterality / modality /
  BI-RADS as first-class fields). They should be parsed into structured
  chunks, NOT split by a generic character/token splitter, or a BI-RADS
  score can get separated from the finding it belongs to.
- Guidelines are prose and belong behind a generic recursive text splitter.

This module only handles the report side. It is intentionally built as a
*parse + audit* pair rather than a bare parser: report formatting was only
verified on 3 sample files, so before trusting this across a full corpus we
need a pass that tells us which files silently fail to match the expected
template, rather than a parser that guesses and produces bad metadata.

Run as a script for a one-shot audit:
    python -m src.ingestion.report_parser /path/to/reports_dir
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import docx


BIRADS_RE = re.compile(r"BIRADS?\s*(III|II|IV|VI|I|V|\d[A-C]?)\b", re.IGNORECASE)
ROMAN_TO_ARABIC = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5", "VI": "6"}

def _normalize_birads(raw: str) -> str:
    up = raw.upper()
    return ROMAN_TO_ARABIC.get(up, up)
ACR_RE = re.compile(r"ACR\s+([A-D])\s*:\s*(.+)", re.IGNORECASE)
PATIENT_RE = re.compile(r"PATIENT\s*NO\.?\s*(\d+)", re.IGNORECASE)

DM_HEADER_MARKERS = ("SOFT TISSUE MAMMOGRAPHY", "LOW DOSE")
CESM_HEADER_MARKERS = ("CONTRAST ENHANCED", "CESM")
OPINION_HEADER = "OPINION"
RIGHT_MARKER = "RIGHT BREAST"
LEFT_MARKER = "LEFT BREAST"


@dataclass
class BreastFinding:
    findings: list[str] = field(default_factory=list)
    birads: str | None = None


@dataclass
class ReportRecord:
    source_file: str
    patient_id: str | None = None
    acr_category: str | None = None
    acr_description: str | None = None
    laterality_scope: str | None = None
    dm_right: BreastFinding = field(default_factory=BreastFinding)
    dm_left: BreastFinding = field(default_factory=BreastFinding)
    dm_birads_right: str | None = None
    dm_birads_left: str | None = None
    cesm_right: BreastFinding = field(default_factory=BreastFinding)
    cesm_left: BreastFinding = field(default_factory=BreastFinding)
    birads_changed_right: bool = False
    birads_changed_left: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.warnings) == 0


def _paragraph_texts(path: Path) -> list[str]:
    d = docx.Document(str(path))
    return [p.text.strip() for p in d.paragraphs]


def _split_sections(paras: list[str]) -> dict[str, list[str]]:
    idx_dm = idx_opinion = idx_cesm = None
    for i, p in enumerate(paras):
        up = p.upper()
        if idx_dm is None and any(m in up for m in DM_HEADER_MARKERS) and "REVEALED" in up:
            idx_dm = i
        elif idx_opinion is None and up.startswith(OPINION_HEADER):
            idx_opinion = i
        elif idx_cesm is None and any(m in up for m in CESM_HEADER_MARKERS) and "REVEALED" in up:
            idx_cesm = i

    sections: dict[str, list[str]] = {}
    sections["header"] = paras[0 : idx_dm if idx_dm is not None else len(paras)]
    if idx_dm is not None:
        end = idx_opinion if idx_opinion is not None else len(paras)
        sections["dm"] = paras[idx_dm:end]
    if idx_opinion is not None:
        end = idx_cesm if idx_cesm is not None else len(paras)
        sections["opinion"] = paras[idx_opinion:end]
    if idx_cesm is not None:
        sections["cesm"] = paras[idx_cesm:]
    return sections


def _split_laterality(block: list[str]) -> tuple[list[str], list[str]]:
    right, left = [], []
    current = None
    for line in block:
        if not line:
            continue
        up = line.upper()
        if RIGHT_MARKER in up:
            current = "right"
            continue
        if LEFT_MARKER in up:
            current = "left"
            continue
        if current == "right":
            right.append(line)
        elif current == "left":
            left.append(line)
    return right, left


def _extract_finding(lines: list[str]) -> BreastFinding:
    bf = BreastFinding()
    for line in lines:
        if ACR_RE.search(line):
            continue
        m = BIRADS_RE.search(line)
        if m:
            bf.birads = _normalize_birads(m.group(1))
        bf.findings.append(line)
    return bf


def parse_report(path: Path) -> ReportRecord:
    rec = ReportRecord(source_file=path.name)
    paras = [p for p in _paragraph_texts(path)]
    non_empty = [p for p in paras if p]

    pm = next((PATIENT_RE.search(p) for p in non_empty if PATIENT_RE.search(p)), None)
    if pm:
        rec.patient_id = pm.group(1)
    else:
        rec.warnings.append("patient_id not found")

    sections = _split_sections(paras)

    all_text_upper = "\n".join(non_empty).upper()
    has_right = RIGHT_MARKER in all_text_upper
    has_left = LEFT_MARKER in all_text_upper
    if has_right and has_left:
        rec.laterality_scope = "bilateral"
    elif has_right:
        rec.laterality_scope = "right_only"
    elif has_left:
        rec.laterality_scope = "left_only"
    else:
        rec.laterality_scope = "unknown"
        rec.warnings.append("no 'Right Breast' / 'Left Breast' marker found anywhere")

    expect_right = has_right
    expect_left = has_left

    if "dm" not in sections:
        rec.warnings.append("DM ('...REVEALED:') section header not found")
    else:
        acr_line = next((p for p in sections["dm"] if ACR_RE.search(p)), None)
        if acr_line:
            m = ACR_RE.search(acr_line)
            rec.acr_category = m.group(1).upper()
            rec.acr_description = m.group(2).strip()
        else:
            is_postop = any(
                "POSTOPERATIVE" in p.upper() or "POST OPERATIVE" in p.upper()
                for p in sections["dm"]
            )
            if not is_postop:
                rec.warnings.append("ACR density category not found in DM section")
        r, l = _split_laterality(sections["dm"])
        rec.dm_right = _extract_finding(r)
        rec.dm_left = _extract_finding(l)
        if expect_right and not r:
            rec.warnings.append("DM: 'Right Breast' expected (seen elsewhere) but no content found")
        if expect_left and not l:
            rec.warnings.append("DM: 'Left Breast' expected (seen elsewhere) but no content found")

    op_right_birads = op_left_birads = None
    if "opinion" not in sections:
        rec.warnings.append("OPINION section header not found")
    else:
        r, l = _split_laterality(sections["opinion"])
        op_right = _extract_finding(r)
        op_left = _extract_finding(l)
        op_right_birads = op_right.birads
        op_left_birads = op_left.birads
        rec.dm_birads_right = op_right_birads
        rec.dm_birads_left = op_left_birads
        if expect_right and op_right_birads is None:
            rec.warnings.append("OPINION: expected right-breast BI-RADS not found")
        if expect_left and op_left_birads is None:
            rec.warnings.append("OPINION: expected left-breast BI-RADS not found")

    if "cesm" not in sections:
        rec.warnings.append("CESM ('...REVEALED:') section header not found")
    else:
        r, l = _split_laterality(sections["cesm"])
        rec.cesm_right = _extract_finding(r)
        rec.cesm_left = _extract_finding(l)
        if expect_right and rec.cesm_right.birads is None:
            rec.warnings.append("CESM: expected right-breast BI-RADS not found")
        if expect_left and rec.cesm_left.birads is None:
            rec.warnings.append("CESM: expected left-breast BI-RADS not found")

    if op_right_birads and rec.cesm_right.birads and op_right_birads != rec.cesm_right.birads:
        rec.birads_changed_right = True
    if op_left_birads and rec.cesm_left.birads and op_left_birads != rec.cesm_left.birads:
        rec.birads_changed_left = True

    return rec


def audit_reports(directory: Path) -> list[ReportRecord]:
    files = sorted(f for f in directory.glob("*.docx") if not f.name.startswith("~$"))
    if not files:
        print(f"No .docx files found in {directory}")
        return []

    records = [parse_report(f) for f in files]

    clean = [r for r in records if r.is_clean]
    flagged = [r for r in records if not r.is_clean]

    print(f"Audited {len(records)} report(s): {len(clean)} clean, {len(flagged)} flagged\n")

    if flagged:
        print("--- Flagged files (genuine structural issues only) ---")
        for r in flagged:
            print(f"\n{r.source_file} (patient {r.patient_id}):")
            for w in r.warnings:
                print(f"  - {w}")

    birads_seen: dict[str, int] = {}
    acr_seen: dict[str, int] = {}
    scope_seen: dict[str, int] = {}
    changed_count = 0
    breast_count = 0
    for r in records:
        scope_seen[r.laterality_scope or "unknown"] = scope_seen.get(r.laterality_scope or "unknown", 0) + 1
        for bf in (r.dm_right, r.dm_left, r.cesm_right, r.cesm_left):
            if bf.birads:
                birads_seen[bf.birads] = birads_seen.get(bf.birads, 0) + 1
        if r.acr_category:
            acr_seen[r.acr_category] = acr_seen.get(r.acr_category, 0) + 1
        for changed, present in (
            (r.birads_changed_right, r.dm_birads_right is not None),
            (r.birads_changed_left, r.dm_birads_left is not None),
        ):
            if present:
                breast_count += 1
                if changed:
                    changed_count += 1

    print("\n--- Corpus summary ---")
    print(f"Laterality scope: {dict(sorted(scope_seen.items()))}")
    print(f"BI-RADS distribution (CESM+DM findings combined): {dict(sorted(birads_seen.items()))}")
    print(f"ACR density distribution: {dict(sorted(acr_seen.items()))}")
    if breast_count:
        pct = 100 * changed_count / breast_count
        print(
            f"BI-RADS changed from DM-opinion to CESM: {changed_count}/{breast_count} breasts ({pct:.0f}%)"
            "  -- expected/clinically meaningful, not an error"
        )

    return records


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.ingestion.report_parser <reports_dir>")
        sys.exit(1)
    audit_reports(Path(sys.argv[1]))
