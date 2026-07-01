"""
Baseline eval: measures zero-shot / few-shot vision generation accuracy
against ground truth (ReportRecord from report_parser.py) across MULTIPLE
cases, for either modality (DM = LE, CESM = DES).

Design rationale
-----------------
A single generated report tells you almost nothing -- it could be a lucky
hit or an unlucky miss. Tuning the prompt against one example is
overfitting to that example, not improving the system. This script exists
to turn "it got patient 3 wrong" into an actual number ("it agreed with
ground truth on N/M breasts"), so that future prompt/retrieval changes can
be judged by whether they move that number, not by whether they happen to
fix one anecdote.

Two accuracy metrics are reported, deliberately:
- Exact BI-RADS match: strict, matches the digit exactly.
- Bucket match (benign 1-2 vs suspicious 4-6, with 3 as its own bucket):
  a looser, clinically-motivated metric -- getting BI-RADS 4 vs 5 wrong
  is a much smaller error than getting 1 vs 5 wrong, and collapsing both
  into "wrong" would hide that distinction.

Few-shot grounding examples are pulled DYNAMICALLY from real parsed report
data for a small set of permanently-reserved patients (FEWSHOT_PATIENT_IDS),
rather than hand-typed strings -- hand-typed few-shot text risks silently
diverging from the actual source-of-truth wording over time. The reserved
patients are excluded from the eval set STRUCTURALLY (by ID, unconditionally)
rather than by picking IDs that happen to fall outside today's eval range --
that "safe range" approach breaks silently the moment max_patients grows
large enough to reach those IDs. run_eval() also asserts no overlap at
runtime as a second, independent safety net.

Costs real API credits (one call per breast per modality, plus the few-shot
images resent on every call). Defaults to a small subset, not the full
326-patient corpus, for exactly that reason -- scale up deliberately once
the harness itself is confirmed to work.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from ..ingestion.report_parser import ReportRecord, audit_reports
from ..generation.vision_report import FewShotExample, GeneratedReport, Modality, generate_report, resolve_image_path

# Permanently reserved for few-shot grounding. NEVER used as eval cases,
# regardless of how large max_patients grows -- see build_eval_cases().
FEWSHOT_PATIENT_IDS = ["30", "31", "34", "39"]


def birads_bucket(birads: str) -> str:
    digit = birads[0]
    if digit in ("1", "2"):
        return "benign"
    if digit == "3":
        return "probably_benign"
    return "suspicious"  # 4, 4A/B/C, 5, 6


@dataclass
class EvalCase:
    patient_id: str
    side: str
    modality: Modality
    ground_truth_birads: str
    ground_truth_finding: str
    generated: GeneratedReport | None = None
    error: str | None = None

    @property
    def exact_match(self) -> bool | None:
        if self.generated is None or not self.generated.is_valid:
            return None
        return self.generated.birads == self.ground_truth_birads

    @property
    def bucket_match(self) -> bool | None:
        if self.generated is None or not self.generated.is_valid:
            return None
        return birads_bucket(self.generated.birads) == birads_bucket(self.ground_truth_birads)


def _ground_truth_for(rec: ReportRecord, side: str, modality: Modality):
    breast = (rec.dm_right if (modality == "DM" and side == "R") else
              rec.dm_left if (modality == "DM" and side == "L") else
              rec.cesm_right if side == "R" else rec.cesm_left)
    birads = (rec.dm_birads_right if (modality == "DM" and side == "R") else
              rec.dm_birads_left if (modality == "DM" and side == "L") else
              breast.birads)
    return breast, birads


def build_eval_cases(
    records: list[ReportRecord], modality: Modality, exclude_patient_ids: set[str] | None = None
) -> list[EvalCase]:
    exclude = exclude_patient_ids or set()
    cases = []
    for rec in records:
        if not rec.patient_id or rec.patient_id in exclude:
            continue
        for side in ("R", "L"):
            breast, gt_birads = _ground_truth_for(rec, side, modality)
            if gt_birads is None or not breast.findings:
                continue
            cases.append(
                EvalCase(
                    patient_id=rec.patient_id,
                    side=side,
                    modality=modality,
                    ground_truth_birads=gt_birads,
                    ground_truth_finding=" ".join(breast.findings),
                )
            )
    return cases


def build_fewshot_examples(
    records_by_id: dict[str, ReportRecord],
    images_dir: Path,
    modality: Modality,
    patient_ids: list[str] = FEWSHOT_PATIENT_IDS,
    max_examples: int = 3,
) -> list[FewShotExample]:
    """
    Pulls one example per BI-RADS bucket (benign / probably_benign /
    suspicious) from real parsed report data for the reserved patients,
    up to max_examples. If a bucket isn't available among the reserved
    patients for this modality, that bucket is simply skipped -- fewer,
    real examples beat padding with a fabricated one.
    """
    pool = []
    for pid in patient_ids:
        rec = records_by_id.get(pid)
        if rec is None:
            continue
        for side in ("R", "L"):
            breast, birads = _ground_truth_for(rec, side, modality)
            if birads is None or not breast.findings:
                continue
            try:
                img = resolve_image_path(images_dir, pid, side, modality)
            except FileNotFoundError:
                continue
            pool.append((birads_bucket(birads), FewShotExample(
                image_path=img, modality=modality, finding=" ".join(breast.findings), birads=birads)))

    seen_buckets = set()
    examples = []
    for bucket, ex in pool:
        if bucket in seen_buckets or len(examples) >= max_examples:
            continue
        examples.append(ex)
        seen_buckets.add(bucket)
    return examples


def run_eval(
    cases: list[EvalCase],
    images_dir: Path,
    guideline_context: str,
    client=None,
    few_shot_examples: list[FewShotExample] | None = None,
) -> list[EvalCase]:
    if few_shot_examples:
        eval_patient_ids = {c.patient_id for c in cases}
        fewshot_patient_ids = {
            e.image_path.stem.split("_")[0].lstrip("P") for e in few_shot_examples
        }
        overlap = eval_patient_ids & fewshot_patient_ids
        assert not overlap, (
            f"Few-shot leakage: patient(s) {overlap} appear in BOTH the eval set and the "
            f"few-shot examples. This would inflate accuracy dishonestly -- fix by choosing "
            f"different few-shot patients, not by ignoring this."
        )

    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] patient {case.patient_id} side {case.side} modality {case.modality} ...", end=" ")
        try:
            img = resolve_image_path(images_dir, case.patient_id, case.side, case.modality)
            case.generated = generate_report(
                img, case.modality, guideline_context, client=client, few_shot_examples=few_shot_examples
            )
            status = "OK" if case.exact_match else ("bucket-match" if case.bucket_match else "MISS")
            print(f"generated={case.generated.birads!r} gt={case.ground_truth_birads!r} [{status}]")
        except Exception as e:
            case.error = str(e)
            print(f"ERROR: {e}")
    return cases


def report_results(cases: list[EvalCase]) -> None:
    scored = [c for c in cases if c.exact_match is not None]
    errored = [c for c in cases if c.error is not None]
    invalid = [c for c in cases if c.generated is not None and not c.generated.is_valid]

    print(f"\n--- Results ---")
    print(f"Total cases: {len(cases)}, scored: {len(scored)}, errored: {len(errored)}, invalid parse: {len(invalid)}")

    if scored:
        exact = sum(1 for c in scored if c.exact_match)
        bucket = sum(1 for c in scored if c.bucket_match)
        print(f"Exact BI-RADS match: {exact}/{len(scored)} ({100*exact/len(scored):.0f}%)")
        print(f"Bucket match (benign/probably-benign/suspicious): {bucket}/{len(scored)} ({100*bucket/len(scored):.0f}%)")

        from collections import Counter
        gt_dist = Counter(c.ground_truth_birads[0] for c in scored)
        gen_dist = Counter(c.generated.birads[0] for c in scored)
        print(f"Ground truth digit distribution: {dict(sorted(gt_dist.items()))}")
        print(f"Generated digit distribution:    {dict(sorted(gen_dist.items()))}")

        print("\nPer-case detail:")
        for c in scored:
            marker = "OK   " if c.exact_match else ("~ok  " if c.bucket_match else "MISS ")
            print(f"  {marker} patient {c.patient_id:>4} {c.side} {c.modality}: "
                  f"generated={c.generated.birads!r:>4} gt={c.ground_truth_birads!r:>4}")

    if errored:
        print("\nErrors:")
        for c in errored:
            print(f"  patient {c.patient_id} {c.side} {c.modality}: {c.error}")


GUIDELINE_TEXT = """BI-RADS 1 (Negative): Normal fibroglandular tissue only. No mass, no \
architectural distortion, no suspicious calcifications, symmetric with the other breast.
Base rate note: most breasts in a general screening population are BI-RADS 1 or 2.

BI-RADS 2 (Benign): A finding IS present but is unambiguously benign -- e.g. coarse \
"popcorn" or round macrocalcifications, a simple dilated duct, a stable surgical scar, \
a lymph node. The image is NOT normal, but nothing about the finding's shape or margin \
suggests malignancy.

BI-RADS 3 (Probably benign): A mass with a smooth, well-circumscribed, rounded or oval \
margin and uniform internal density. The margin smoothness is the deciding feature here,
not simply "a mass is present."

BI-RADS 4 (Suspicious): Mass or calcifications with an indistinct, obscured, or \
microlobulated margin (the margin partially blends into surrounding tissue, or has small \
undulations rather than being smooth). Biopsy would be considered. Subdivided 4A (low \
suspicion) / 4B (moderate) / 4C (high).

BI-RADS 5 (Highly suggestive of malignancy): Reserve this category ONLY for a mass with a \
clearly spiculated margin (distinct thin lines radiating outward into surrounding tissue) \
AND/OR segmental or ductal non-mass enhancement that clearly follows a duct system. Do \
NOT assign BI-RADS 5 for density variation, asymmetry, or enhancement alone without a \
clearly spiculated or architecturally distorted margin -- that pattern is BI-RADS 4 at most. \
BI-RADS 5 is rare in a general population; assigning it should be the exception, not the \
default when uncertain.

BI-RADS 6: Known biopsy-proven malignancy, imaged for treatment monitoring.

General calibration guidance: when genuinely uncertain between two adjacent categories, \
prefer the LOWER (less severe) one unless a specific criterion above (spiculated margin, \
segmental/ductal enhancement pattern) is clearly met. Over-calling suspicion on ambiguous \
findings is itself a documented error mode to avoid, not a "safe default." """


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    if len(sys.argv) < 3:
        print("Usage: python -m src.eval.baseline_eval <reports_dir> <images_dir> [max_patients] [modality: DM|CESM]")
        sys.exit(1)

    reports_dir = Path(sys.argv[1])
    images_dir = Path(sys.argv[2])
    max_patients = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    modality: Modality = sys.argv[4] if len(sys.argv) > 4 else "DM"
    assert modality in ("DM", "CESM"), f"modality must be DM or CESM, got {modality!r}"

    all_records = audit_reports(reports_dir)
    records_by_id = {r.patient_id: r for r in all_records if r.patient_id}

    few_shot_examples = build_fewshot_examples(records_by_id, images_dir, modality)
    print(f"Built {len(few_shot_examples)} few-shot example(s) for {modality} "
          f"from reserved patients {FEWSHOT_PATIENT_IDS}")

    eligible = [
        r for r in all_records
        if r.patient_id and r.patient_id not in FEWSHOT_PATIENT_IDS and int(r.patient_id) <= max_patients + 5
    ][:max_patients]

    print(f"Building eval cases for {len(eligible)} patients ({modality} modality)...")
    cases = build_eval_cases(eligible, modality=modality, exclude_patient_ids=set(FEWSHOT_PATIENT_IDS))
    print(f"Built {len(cases)} eval cases (one per breast with a known BI-RADS)")

    cases = run_eval(cases, images_dir, GUIDELINE_TEXT, few_shot_examples=few_shot_examples)
    report_results(cases)