"""
Generates a report matching the structure of the real CDD-CESM reports
(PATIENT NO. / DM REVEALED / OPINION / CESM REVEALED, per breast), using:

- The trained classifier (classifier.py) for the CATEGORY -- this is the
  grounded, verified part. Confirmed correct on real cases this session
  (patient 40: both sides, both modalities, >99.9% confidence, matching
  independently-validated ground truth).
- The vision LLM for the NARRATIVE finding description (location, shape,
  margin language) -- explicitly conditioned on the classifier's category
  rather than guessing independently, since we already rigorously proved
  (baseline_eval.py) that unconditioned zero/few-shot category guessing
  from a general vision model is unreliable.

Deliberate limitations, stated in the output rather than hidden:
- The classifier gives a COARSE 3-way group (Benign / Suspicious /
  Malignant), not the individual BI-RADS digit (e.g. it cannot
  distinguish BI-RADS 3 from 4, or 4A from 4B/4C -- both fall under
  "Suspicious"). The generated report states the group, explicitly
  labeled as a model estimate, not a real BI-RADS digit assignment.
- ACR breast density is not covered by any trained classifier in this
  project. Rather than have the LLM freely assign an ACR letter grade
  with false confidence, the report states this is out of scope for the
  current model suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from classifier import ClassifierResult, Modality, classify_image
from vision_report import encode_image_base64, resolve_image_path

Laterality = str  # "R" | "L"

ACR_DISCLAIMER = "ACR density: not assessed (no trained classifier covers breast density in this project; requires manual review)."

NARRATIVE_PROMPT_TEMPLATE = """You are assisting with a RESEARCH project on automated CESM report \
generation. This is NOT a clinical diagnostic tool.

A TRAINED, VALIDATED classifier has already assessed this image. Its category assessment is \
GROUND TRUTH -- do not second-guess or contradict it:

  Category: {category}
  Confidence: {confidence:.1%}

IMPORTANT, based on a controlled test: this system's narrative generation has been shown NOT to \
reliably read fine-grained visual details (specific quadrant/location, exact margin type) from \
the image -- when tested, two genuinely different images (one benign, one malignant) produced \
near-identical location and margin language when given the same category. Do not claim a \
specific quadrant, location, or precise margin type as if confirmed for this image -- that would \
be false precision.

Instead, write ONE sentence of GENERAL, TYPICAL descriptive language for what a "{category}" \
finding characteristically looks like in {modality_name} imaging, in the terminology style of \
the lexicon below. This is illustrative/educational phrasing about the category, not a specific \
claim about a location or feature you have confirmed in this particular image.

Do NOT state a specific BI-RADS digit or subcategory (e.g. "BI-RADS 4" or "4A").
Do NOT state a specific quadrant or location.

--- LEXICON STYLE REFERENCE ---
{guideline_context}
--- END LEXICON ---

{modality_instructions}

Respond with ONLY the one-sentence typical description, no preamble, no JSON, no BI-RADS number, \
no specific location claim."""

MODALITY_INSTRUCTIONS = {
    "DM": "This is a Low Energy (LE) mammography image (pre-contrast). Describe density/architecture findings only.",
    "CESM": "This is a Dual-Energy Subtracted (DES) contrast image. Describe enhancement findings only.",
}
MODALITY_NAMES = {"DM": "low energy (LE) mammography", "CESM": "contrast-enhanced (CESM/DES)"}


@dataclass
class FindingSection:
    modality: Modality
    classifier_result: ClassifierResult
    narrative: str

    def format_lines(self) -> list[str]:
        return [
            f"- [Typical/illustrative language for this category, NOT a confirmed finding for "
            f"this specific image -- see note below] {self.narrative}",
            f"- (Model-estimated category: {self.classifier_result.birads_group}, "
            f"confidence {self.classifier_result.birads_group_probs[self.classifier_result.birads_group]:.0%} "
            f"-- NOT an official BI-RADS digit assignment)",
        ]


@dataclass
class PatientReport:
    patient_id: str
    dm_right: FindingSection | None = None
    dm_left: FindingSection | None = None
    cesm_right: FindingSection | None = None
    cesm_left: FindingSection | None = None

    def format(self) -> str:
        lines = [
            f"PATIENT NO. {self.patient_id}",
            "",
            "[NOTE: category labels below come from a trained, validated classifier (verified",
            " against real ground truth this session). Descriptive narrative text is GENERAL,",
            " TYPICAL language for that category, not a confirmed finding specific to this image --",
            " a controlled test showed this system does not reliably read fine-grained visual",
            " detail (location, exact margin type) from the image itself.]",
            "",
        ]
        lines.append("DIGITALIZED LOW DOSE SOFT TISSUE MAMMOGRAPHY REVEALED:")
        lines.append(ACR_DISCLAIMER)
        lines.append("")
        for side, section in (("Right", self.dm_right), ("Left", self.dm_left)):
            if section is None:
                continue
            lines.append(f"{side} Breast:")
            lines.extend(section.format_lines())
            lines.append("")

        lines.append("OPINION:")
        for side, section in (("Right", self.dm_right), ("Left", self.dm_left)):
            if section is None:
                continue
            lines.append(f"{side} Breast:")
            lines.append(
                f"- {section.classifier_result.birads_group} "
                f"(model estimate, confidence {section.classifier_result.birads_group_probs[section.classifier_result.birads_group]:.0%})."
            )
            if section.classifier_result.cancer:
                lines.append(
                    f"  Cancer flag: POSITIVE (prob={section.classifier_result.cancer_prob:.2f})."
                )
        lines.append("")

        lines.append("CONTRAST ENHANCED SPECTRAL MAMMOGRAPHY REVEALED:")
        for side, section in (("Right", self.cesm_right), ("Left", self.cesm_left)):
            if section is None:
                continue
            lines.append(f"{side} Breast:")
            lines.extend(section.format_lines())
            if section.classifier_result.cancer:
                lines.append(
                    f"  Cancer flag: POSITIVE (prob={section.classifier_result.cancer_prob:.2f})."
                )
            lines.append("")

        return "\n".join(lines)


def _generate_narrative(
    image_path: Path, modality: Modality, classifier_result: ClassifierResult, guideline_context: str, client
) -> str:
    if client is None:
        from anthropic import Anthropic

        client = Anthropic()

    image_data, media_type = encode_image_base64(image_path)
    prompt = NARRATIVE_PROMPT_TEMPLATE.format(
        category=classifier_result.birads_group,
        confidence=classifier_result.birads_group_probs[classifier_result.birads_group],
        guideline_context=guideline_context.strip(),
        modality_instructions=MODALITY_INSTRUCTIONS[modality],
        modality_name=MODALITY_NAMES[modality],
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        system=prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                    {"type": "text", "text": "Describe the finding per the instructions above."},
                ],
            }
        ],
    )
    return response.content[0].text.strip()


def generate_full_report(
    patient_id: str,
    images_dir: Path,
    checkpoint_dir: Path,
    guideline_context: str,
    client=None,
    device: str = "cpu",
) -> PatientReport:
    report = PatientReport(patient_id=patient_id)

    for modality, attr_r, attr_l in (("DM", "dm_right", "dm_left"), ("CESM", "cesm_right", "cesm_left")):
        for side, attr in ((("R", attr_r)), (("L", attr_l))):
            try:
                img_path = resolve_image_path(images_dir, patient_id, side, modality)
            except FileNotFoundError:
                continue  # this breast/modality genuinely wasn't imaged -- omit, don't fabricate
            result = classify_image(img_path, modality, checkpoint_dir, device=device, laterality=side)
            narrative = _generate_narrative(img_path, modality, result, guideline_context, client)
            setattr(report, attr, FindingSection(modality=modality, classifier_result=result, narrative=narrative))

    return report