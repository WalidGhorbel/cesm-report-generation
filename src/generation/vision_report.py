"""
Vision-based report generation: given a single mammography image (LE or
DES), generate a structured finding + BI-RADS assessment, grounded in the
standardized CEM lexicon retrieved from the guideline corpus.

Design notes
------------
- LE and DES are generated INDEPENDENTLY (one API call each), not as two
  sections of one call. This mirrors how the radiologist actually produces
  two separate opinions in the source reports (DM section vs CESM section)
  -- and those two opinions genuinely disagree ~35% of the time in this
  corpus, which is the whole clinical point of doing CESM. Merging them
  into one call would blur that distinction.
- The model is asked for structured JSON output (finding + birads), not
  free text, so generated reports are directly comparable to the ground
  truth ReportRecord fields (dm_right.birads, cesm_right.birads, etc.)
  that report_parser.py already extracts and that annotations_loader.py
  already cross-validated at 566/566 against independent ground truth.
- guideline_context is a plain string parameter, not fetched internally.
  This module does NOT do retrieval itself -- it's deliberately decoupled
  from however the Qdrant retrieval step is implemented, so it can be
  built and unit-tested (prompt construction, response parsing) before
  that retrieval pipeline exists, and swapped freely once it does.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Modality = Literal["DM", "CESM"]

MODALITY_INSTRUCTIONS = {
    "DM": (
        "This is a Low Energy (LE) digital mammography image -- physically "
        "equivalent to a standard mammogram, taken BEFORE contrast injection. "
        "Assess tissue density and architecture only: masses, asymmetries, "
        "architectural distortion, calcifications, skin/contour changes. "
        "Do NOT reference contrast enhancement -- it is not visible in this image."
    ),
    "CESM": (
        "This is a Dual-Energy Subtracted (DES/recombined) contrast-enhanced "
        "mammography image -- a computed recombination of a low-energy and a "
        "post-contrast high-energy exposure, which cancels background tissue "
        "and isolates iodine contrast uptake. Assess enhancement only: presence, "
        "type (mass vs non-mass), intensity, and pattern of enhancement. "
        "Background parenchyma appears suppressed by design; do not describe it "
        "as absent tissue."
    ),
}

SYSTEM_PROMPT_TEMPLATE = """You are assisting with a RESEARCH project on automated CESM report \
generation. This is NOT a clinical diagnostic tool and output must never be presented \
as a clinical read. You are given one mammography image and must produce a finding \
description and BI-RADS category using the standardized lexicon below, extracted from \
peer-reviewed CEM interpretation guidelines.

--- LEXICON / GUIDELINE CONTEXT ---
{guideline_context}
--- END LEXICON ---

{modality_instructions}

Respond with ONLY a JSON object, no markdown fences, no preamble:
{{"finding": "<one or two sentence finding description, matching the style and \
terminology of the lexicon above>", "birads": "<BI-RADS category as a single \
Arabic digit 1-6, with an optional trailing A/B/C for category 4 subtypes>"}}
"""


@dataclass
class GeneratedReport:
    modality: Modality
    image_path: str
    finding: str | None
    birads: str | None
    raw_response: str
    parse_error: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.parse_error is None and self.birads is not None


def encode_image_base64(image_path: Path) -> tuple[str, str]:
    """Returns (base64_data, media_type)."""
    suffix = image_path.suffix.lower()
    media_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(suffix)
    if media_type is None:
        raise ValueError(f"Unsupported image type: {suffix}")
    data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    return data, media_type


def build_system_prompt(modality: Modality, guideline_context: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        guideline_context=guideline_context.strip(),
        modality_instructions=MODALITY_INSTRUCTIONS[modality],
    )


def parse_model_response(raw: str) -> tuple[str | None, str | None, str | None]:
    """
    Returns (finding, birads, parse_error). Tolerant of accidental code
    fences even though the prompt asks the model not to use them, since
    trusting a prompt instruction to always hold is exactly the kind of
    assumption this project has been testing at every other stage.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, None, f"JSON parse error: {e}"

    finding = obj.get("finding")
    birads = obj.get("birads")
    if not finding or not birads:
        return finding, birads, "missing 'finding' or 'birads' key in response"

    birads = str(birads).strip().upper()
    if not re.fullmatch(r"[1-6][A-C]?", birads):
        return finding, birads, f"BI-RADS value {birads!r} doesn't match expected format"

    return finding, birads, None


@dataclass
class FewShotExample:
    image_path: Path
    modality: Modality
    finding: str
    birads: str


def _build_example_turns(examples: list["FewShotExample"]) -> list[dict]:
    turns = []
    for ex in examples:
        image_data, media_type = encode_image_base64(ex.image_path)
        turns.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": image_data},
                    },
                    {"type": "text", "text": "Assess this image per the instructions above."},
                ],
            }
        )
        turns.append(
            {
                "role": "assistant",
                "content": json.dumps({"finding": ex.finding, "birads": ex.birads}),
            }
        )
    return turns


def generate_report(
    image_path: Path,
    modality: Modality,
    guideline_context: str,
    client=None,
    model: str = "claude-sonnet-4-6",
    few_shot_examples: list["FewShotExample"] | None = None,
) -> GeneratedReport:
    """
    client: an anthropic.Anthropic() instance, passed in rather than
    constructed here so this function stays testable without a live API
    key / network access.

    few_shot_examples: optional grounding examples shown as prior
    user/assistant turns before the real query. Caller is responsible for
    ensuring these come from patients NOT in whatever eval set this is
    being used for -- this function does not and cannot enforce that,
    since it has no notion of "the eval set".
    """
    if client is None:
        from anthropic import Anthropic

        client = Anthropic()

    image_data, media_type = encode_image_base64(image_path)
    system_prompt = build_system_prompt(modality, guideline_context)

    messages = []
    if few_shot_examples:
        # only use examples matching this call's modality -- an LE example
        # doesn't help ground a DES assessment and vice versa
        matching = [e for e in few_shot_examples if e.modality == modality]
        messages.extend(_build_example_turns(matching))

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": image_data},
                },
                {"type": "text", "text": "Assess this image per the instructions above."},
            ],
        }
    )

    response = client.messages.create(
        model=model,
        max_tokens=500,
        system=system_prompt,
        messages=messages,
    )

    raw = response.content[0].text
    finding, birads, err = parse_model_response(raw)
    return GeneratedReport(
        modality=modality,
        image_path=str(image_path),
        finding=finding,
        birads=birads,
        raw_response=raw,
        parse_error=err,
    )


def resolve_image_path(
    images_dir: Path, patient_id: str, side: str, modality: Modality, view: str = "MLO"
) -> Path:
    """
    modality here is "DM" or "CESM" (matches annotations_loader.Type);
    filenames use "DM" and "CM" respectively -- this is the same label
    mismatch documented in annotations_loader.py.

    Falls back to the other standard view (CC or MLO) if the requested
    one is not present -- not every breast in this corpus was captured in
    both views (e.g. patient 4's right breast only has a CC image), and
    that is a real data gap, not something worth hard-failing the whole
    case over when a usable image exists under the other view.
    """
    filename_infix = {"DM": "DM", "CESM": "CM"}[modality]
    other_view = "CC" if view == "MLO" else "MLO"

    for v in (view, other_view):
        pattern = f"P{patient_id}_{side}_{filename_infix}_{v}.jpg"
        matches = list(images_dir.rglob(pattern))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous: {len(matches)} images matched {pattern}: {matches}")

    raise FileNotFoundError(
        f"No image found for patient {patient_id} side {side} modality {modality} "
        f"in either view ({view} or {other_view}) under {images_dir}"
    )