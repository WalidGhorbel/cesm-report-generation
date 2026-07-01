"""
Regression test: locks in the full-corpus parsing result now that it's
been validated end-to-end against the xlsx ground truth. If this test
starts failing, something in report_parser.py changed behavior -- check
whether that's an intentional fix or an accidental regression before
touching the assertions.
"""
from pathlib import Path

from src.ingestion.report_parser import audit_reports
from src.ingestion.annotations_loader import load_annotations, cross_validate_against_reports

DATA_DIR = Path.home() / "data" / "cdd-cesm"
REPORTS_DIR = DATA_DIR / "Medical reports for cases"
XLSX_PATH = DATA_DIR / "Radiology-manual-annotations.xlsx"


def test_full_corpus_parses_clean():
    records = audit_reports(REPORTS_DIR)
    assert len(records) == 326
    flagged = [r for r in records if not r.is_clean]
    assert flagged == [], f"{len(flagged)} report(s) failed to parse cleanly: {[r.source_file for r in flagged]}"


def test_cross_validation_against_xlsx_ground_truth(capsys):
    annotations = load_annotations(XLSX_PATH)
    records = audit_reports(REPORTS_DIR)
    reports_by_patient = {r.patient_id: r for r in records if r.patient_id}
    cross_validate_against_reports(annotations, reports_by_patient)
    out = capsys.readouterr().out
    assert "566 agree, 0 disagree" in out
