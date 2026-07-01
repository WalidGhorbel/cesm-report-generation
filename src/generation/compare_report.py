"""
Side-by-side comparison between a real report (parsed via report_parser.py)
and a generated report (full_report.py's PatientReport), for a single
patient. Prints ground truth vs generated narrative + category agreement
per breast per modality.

This is deliberately a repeatable tool, not a one-off manual comparison --
we'll want to run this across more patients as the pipeline evolves, and
eyeballing output by hand doesn't scale or stay consistent.
"""

from __future__ import annotations

from pathlib import Path

from classifier import birads_digit_to_group


def _real_finding_and_group(rec, side: str, modality: str) -> tuple[str | None, str | None]:
    """Pulls the real finding text + BI-RADS group (mapped into the classifier's
    3-way space via birads_digit_to_group) for one breast/modality from a
    ReportRecord, mirroring the same field logic used in report_parser.py."""
    if modality == "DM":
        breast = rec.dm_right if side == "R" else rec.dm_left
        birads = rec.dm_birads_right if side == "R" else rec.dm_birads_left
    else:
        breast = rec.cesm_right if side == "R" else rec.cesm_left
        birads = breast.birads

    if not breast.findings:
        return None, None
    finding_text = " ".join(breast.findings)
    group = birads_digit_to_group(birads) if birads else None
    return finding_text, group


def compare_report(rec, generated_report) -> None:
    """
    rec: a ReportRecord from report_parser.parse_report()
    generated_report: a PatientReport from full_report.generate_full_report()
    """
    print(f"{'='*100}")
    print(f"PATIENT {rec.patient_id}: real vs generated")
    print(f"{'='*100}")

    agree_count = 0
    total_count = 0

    for modality, gen_r_attr, gen_l_attr in (("DM", "dm_right", "dm_left"), ("CESM", "cesm_right", "cesm_left")):
        for side, gen_attr in (("R", gen_r_attr), ("L", gen_l_attr)):
            real_text, real_group = _real_finding_and_group(rec, side, modality)
            gen_section = getattr(generated_report, gen_attr)

            if real_text is None and gen_section is None:
                continue  # genuinely not imaged on either side, nothing to compare

            print(f"\n--- {modality} {side} ---")
            print(f"  REAL:      {real_text!r}")
            print(f"  REAL group: {real_group!r}")
            if gen_section is not None:
                print(f"  GENERATED: {gen_section.narrative!r}")
                print(f"  GEN group:  {gen_section.classifier_result.birads_group!r} "
                      f"(confidence {gen_section.classifier_result.birads_group_probs[gen_section.classifier_result.birads_group]:.0%})")
                if real_group is not None:
                    total_count += 1
                    match = real_group == gen_section.classifier_result.birads_group
                    agree_count += int(match)
                    print(f"  CATEGORY MATCH: {'YES' if match else 'NO'}")
            else:
                print(f"  GENERATED: (no image found / not generated for this side)")

    if total_count:
        print(f"\n{'='*100}")
        print(f"Category-level agreement: {agree_count}/{total_count} ({100*agree_count/total_count:.0f}%)")
        print(f"(Narrative TEXT agreement is not scored here -- that's a qualitative read, not a")
        print(f" metric, since we have no ground-truth-labeled 'correct narrative' to compare against,")
        print(f" only the real radiologist's own specific wording, which the model was never shown.)")
