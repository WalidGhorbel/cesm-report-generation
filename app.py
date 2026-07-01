"""
RadRAG -- CESM Report Assistant (Streamlit demo app)

RESEARCH PROTOTYPE. NOT a clinical diagnostic tool. See the status bar
below and the "Methodology & Validation" section for what this system
does and does not reliably do -- both are backed by real evaluation
numbers produced during development of this project, not marketing claims.

Flow (mirrors the real report structure): Low Energy (LE/DM) is imaged
and reported on FIRST for both breasts, then Contrast-Enhanced (DES/CESM)
SECOND for both breasts -- not per-side, per-modality, matching how the
source dataset's actual reports are structured (one DM section covering
both breasts, then one CESM section covering both breasts).

Run with: streamlit run app.py
"""

from __future__ import annotations

import sys
import tempfile
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src" / "generation"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "ingestion"))

from full_report import PatientReport  # noqa: E402
from report_core import (  # noqa: E402
    category_color,
    category_emoji,
    generate_side_report,
    merge_into_report,
    report_is_empty,
)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
CHECKPOINT_DIR = Path.home() / "data/cdd-cesm/checkpoints"
EXAMPLES_DIR = Path(__file__).parent / "examples"

GUIDELINE_CONTEXT = """BI-RADS terminology reference: masses are described by shape (oval/round/irregular),
margin (circumscribed/obscured/indistinct/microlobulated/spiculated), and density.
Enhancement on CESM is described as mass or non-mass, with pattern (segmental/ductal/regional/diffuse)."""

st.set_page_config(page_title="RadRAG -- CESM Report Assistant", page_icon="\U0001fa7b", layout="wide")

# --------------------------------------------------------------------------
# Clinical theme -- dark reading-room palette, IBM Plex type pairing
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #0B0E14;
    --panel: #131824;
    --panel-2: #171E2C;
    --border: #232B3D;
    --text: #E8EDF4;
    --muted: #8B98AC;
    --accent: #3DA9FC;
    --benign: #3ECF8E;
    --suspicious: #F2A93B;
    --malignant: #F2545B;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: var(--bg); color: var(--text); }
section[data-testid="stSidebar"] { background-color: var(--panel); border-right: 1px solid var(--border); }
[data-testid="stHeader"] { background-color: var(--bg); }

.rr-topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 20px; margin: -1rem -1rem 20px -1rem;
    background: linear-gradient(180deg, var(--panel-2) 0%, var(--panel) 100%);
    border-bottom: 1px solid var(--border);
}
.rr-topbar-title {
    font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 1.05rem;
    letter-spacing: 0.04em; color: var(--text); text-transform: uppercase;
}
.rr-topbar-title span { color: var(--accent); }
.rr-status-badge {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.05em;
    color: var(--suspicious); border: 1px solid var(--suspicious); border-radius: 3px;
    padding: 3px 8px; text-transform: uppercase; background: rgba(242,169,59,0.08);
}

.rr-section-header {
    display: flex; align-items: center; gap: 10px;
    font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.95rem;
    letter-spacing: 0.05em; color: var(--accent); text-transform: uppercase;
    margin: 28px 0 4px 0; padding-bottom: 8px; border-bottom: 1px solid var(--border);
}
.rr-section-step {
    display: inline-block; background: var(--accent); color: #05131F;
    font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 0.75rem;
    width: 20px; height: 20px; border-radius: 50%; text-align: center; line-height: 20px;
}
.rr-section-sub { font-family: 'IBM Plex Sans', sans-serif; color: var(--muted); font-size: 0.82rem; margin: -2px 0 14px 0; }

.rr-panel-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.08em;
    color: var(--muted); text-transform: uppercase; margin-bottom: 6px; margin-top: 4px;
}

.rr-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    padding: 14px 16px; margin-bottom: 12px;
}
.rr-card-benign { border-left: 3px solid var(--benign); }
.rr-card-suspicious { border-left: 3px solid var(--suspicious); }
.rr-card-malignant { border-left: 3px solid var(--malignant); }

.rr-badge {
    display: inline-block; font-family: 'IBM Plex Mono', monospace; font-weight: 600;
    font-size: 0.8rem; letter-spacing: 0.04em; padding: 2px 10px; border-radius: 12px;
    text-transform: uppercase;
}
.rr-badge-benign { background: rgba(62,207,142,0.15); color: var(--benign); }
.rr-badge-suspicious { background: rgba(242,169,59,0.15); color: var(--suspicious); }
.rr-badge-malignant { background: rgba(242,84,91,0.15); color: var(--malignant); }

.rr-confbar-track { background: var(--border); border-radius: 3px; height: 6px; margin: 8px 0 4px 0; overflow: hidden; }
.rr-confbar-fill { height: 100%; border-radius: 3px; }

.rr-mono { font-family: 'IBM Plex Mono', monospace; }
.rr-muted { color: var(--muted); }
.rr-narrative-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.05em; margin-top: 8px; margin-bottom: 4px;
}
.rr-narrative-text { font-size: 0.9rem; line-height: 1.5; color: var(--text); }

.rr-cancer-flag {
    font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.78rem;
    color: var(--malignant); background: rgba(242,84,91,0.12); border: 1px solid var(--malignant);
    border-radius: 4px; padding: 4px 8px; display: inline-block; margin-top: 6px;
}

.rr-status-ok {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: var(--benign);
    background: rgba(62,207,142,0.10); border: 1px solid var(--benign); border-radius: 5px;
    padding: 8px 12px; margin: 10px 0;
}
.rr-status-err {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: var(--malignant);
    background: rgba(242,84,91,0.10); border: 1px solid var(--malignant); border-radius: 5px;
    padding: 8px 12px; margin: 10px 0;
}

.stButton > button {
    background-color: var(--panel-2); color: var(--text); border: 1px solid var(--border);
    font-family: 'IBM Plex Sans', sans-serif; font-weight: 500;
}
.stButton > button[kind="primary"] {
    background-color: var(--accent); color: #05131F; border: none; font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached checkpoint check
# --------------------------------------------------------------------------
@st.cache_resource
def _warm_check_checkpoints() -> bool:
    required = ["best_birads_dm.pt", "best_cancer_dm.pt", "best_birads_cesm.pt", "best_cancer_cesm.pt"]
    missing = [f for f in required if not (CHECKPOINT_DIR / f).exists()]
    if missing:
        st.error(
            f"Missing checkpoint file(s) in {CHECKPOINT_DIR}: {missing}. "
            "Update CHECKPOINT_DIR at the top of app.py."
        )
        return False
    return True


# --------------------------------------------------------------------------
# Top status bar
# --------------------------------------------------------------------------
st.markdown(
    """
<div class="rr-topbar">
  <div class="rr-topbar-title">RAD<span>RAG</span> // CESM REPORT ASSISTANT</div>
  <div class="rr-status-badge">\u26a0 Research Prototype -- Not for Clinical Use</div>
</div>
""",
    unsafe_allow_html=True,
)

if not _warm_check_checkpoints():
    st.stop()

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "report" not in st.session_state:
    st.session_state.report = PatientReport(patient_id="uploaded-case")
if "preview" not in st.session_state:
    st.session_state.preview = {}  # {"R": {"le": path, "des": path}, "L": {...}}
if "last_status" not in st.session_state:
    st.session_state.last_status = None  # (ok: bool, message: str) -- persisted across reruns
if "active_example" not in st.session_state:
    st.session_state.active_example = None  # example name if a bundled example is loaded, else None
if "cam_overlays" not in st.session_state:
    st.session_state.cam_overlays = {}  # {"R": {"DM": PIL.Image, "CESM": PIL.Image}, "L": {...}}


def _load_ground_truth(example_name: str | None) -> dict | None:
    """Ground truth (real radiologist finding + BI-RADS) for bundled example
    cases only -- there is no ground truth for a user's own uploaded image.
    Self-contained JSON per example, not re-parsed from a docx at runtime."""
    if example_name is None:
        return None
    gt_path = EXAMPLES_DIR / example_name / "ground_truth.json"
    if not gt_path.exists():
        return None
    return json.loads(gt_path.read_text())


def _save_upload_to_tempfile(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return Path(tmp.name)


@st.cache_data(show_spinner=False)
def _preprocessed_preview(image_path_str: str, laterality: str):
    """Returns the SAME laterality-aware padded, 1024x1024 image the model
    actually receives -- not the raw upload. Verified pixel-perfect (0.00
    mean diff) against real preprocessed training data earlier this session."""
    from PIL import Image
    from classifier import pad_to_square, PAD_SIZE

    img = Image.open(image_path_str).convert("L")
    return pad_to_square(img, laterality, size=PAD_SIZE)


def _confidence_bar(pct: float, color: str) -> str:
    return (
        f'<div class="rr-confbar-track"><div class="rr-confbar-fill" '
        f'style="width:{pct*100:.0f}%;background:{color};"></div></div>'
    )


def _run_modality(modality: str) -> None:
    """Runs classification + narrative generation for ONE modality (LE or
    DES) across BOTH breasts at once -- mirrors the real report structure
    (one DM section covering both breasts, one CESM section covering both
    breasts), not per-side generation."""
    ran_any = False
    errors = []
    for side in ("R", "L"):
        side_preview = st.session_state.preview.get(side, {})
        path = side_preview.get("le") if modality == "DM" else side_preview.get("des")
        if path is None:
            continue
        ran_any = True
        le_arg = path if modality == "DM" else None
        des_arg = path if modality == "CESM" else None
        result = generate_side_report(le_arg, des_arg, side, CHECKPOINT_DIR, GUIDELINE_CONTEXT)
        st.session_state.report = merge_into_report(st.session_state.report, result)
        errors.extend(result.errors)

        # CAM overlay: pure visual attention map, no location TEXT is derived
        # or claimed from it here -- see cam.py docstring for why (verified
        # unreliable on the medial/lateral axis).
        try:
            from cam import compute_cam_overlay
            overlay = compute_cam_overlay(path, modality, CHECKPOINT_DIR, laterality=side)
            st.session_state.cam_overlays.setdefault(side, {})[modality] = overlay
        except Exception as e:
            errors.append(f"CAM ({modality} {side}): {e}")

    if not ran_any:
        st.session_state.last_status = (False, f"No {modality} images loaded for either breast -- nothing to generate.")
    elif errors:
        st.session_state.last_status = (False, f"Generated with errors: {'; '.join(errors)}")
    else:
        label = "Low Energy (LE)" if modality == "DM" else "Contrast-Enhanced (DES)"
        st.session_state.last_status = (True, f"{label} report generated successfully.")


CATEGORY_CSS_CLASS = {"Benign": "benign", "Suspicious": "suspicious", "Malignant": "malignant"}

# --------------------------------------------------------------------------
# Sidebar: case selection
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="rr-panel-label">Case Selection</div>', unsafe_allow_html=True)

    example_names = sorted(p.name for p in EXAMPLES_DIR.iterdir() if p.is_dir()) if EXAMPLES_DIR.exists() else []
    example_choice = st.selectbox("Bundled example case", ["-- none --"] + example_names, label_visibility="collapsed")

    st.divider()
    st.markdown('<div class="rr-panel-label">Upload Study Images</div>', unsafe_allow_html=True)
    laterality_label = st.radio("Breast side", ["Right (R)", "Left (L)"], horizontal=True)
    side_code = "R" if laterality_label.startswith("Right") else "L"

    le_upload = st.file_uploader("Low Energy (LE / DM)", type=["jpg", "jpeg", "png"], key="le_upload")
    des_upload = st.file_uploader("Dual-Energy Subtracted (DES / CESM)", type=["jpg", "jpeg", "png"], key="des_upload")

    st.caption("Uploaded images appear in the sections below automatically -- no separate upload button needed.")

    st.divider()
    if st.button("Clear / start new case", width='stretch'):
        st.session_state.report = PatientReport(patient_id="uploaded-case")
        st.session_state.preview = {}
        st.session_state.last_status = None
        st.session_state.active_example = None
        st.session_state.cam_overlays = {}
        st.rerun()

# --------------------------------------------------------------------------
# Load example previews / uploads into session state
# --------------------------------------------------------------------------
if example_choice != "-- none --":
    st.session_state.active_example = example_choice
    example_dir = EXAMPLES_DIR / example_choice
    for side in ("R", "L"):
        le_path = example_dir / f"{side}_LE.jpg"
        des_path = example_dir / f"{side}_DES.jpg"
        st.session_state.preview.setdefault(side, {})
        st.session_state.preview[side]["le"] = le_path if le_path.exists() else None
        st.session_state.preview[side]["des"] = des_path if des_path.exists() else None

if le_upload is not None or des_upload is not None:
    st.session_state.active_example = None  # ground truth no longer cleanly applies once the user uploads their own image
    st.session_state.preview.setdefault(side_code, {})
    if le_upload is not None:
        st.session_state.preview[side_code]["le"] = _save_upload_to_tempfile(le_upload)
    if des_upload is not None:
        st.session_state.preview[side_code]["des"] = _save_upload_to_tempfile(des_upload)

# --------------------------------------------------------------------------
# Persistent status message -- survives the rerun that follows generation,
# unlike a bare st.warning() issued right before st.rerun()
# --------------------------------------------------------------------------
if st.session_state.last_status is not None:
    ok, message = st.session_state.last_status
    css = "rr-status-ok" if ok else "rr-status-err"
    icon = "\u2713" if ok else "\u2715"
    st.markdown(f'<div class="{css}">{icon} {message}</div>', unsafe_allow_html=True)

report: PatientReport = st.session_state.report


def _render_modality_section(step: str, title: str, subtitle: str, modality: str, right_section, left_section) -> None:
    st.markdown(
        f'<div class="rr-section-header"><span class="rr-section-step">{step}</span> {title}</div>'
        f'<div class="rr-section-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )

    has_r = st.session_state.preview.get("R", {}).get("le" if modality == "DM" else "des") is not None
    has_l = st.session_state.preview.get("L", {}).get("le" if modality == "DM" else "des") is not None

    if not has_r and not has_l:
        st.markdown(
            '<div class="rr-card rr-muted">No images loaded for this modality yet. '
            "Select an example or upload images in the sidebar.</div>",
            unsafe_allow_html=True,
        )
        return

    img_cols = st.columns(2)
    for col, side, has in ((img_cols[0], "R", has_r), (img_cols[1], "L", has_l)):
        with col:
            st.markdown(f'<div class="rr-mono" style="color:var(--muted);font-size:0.75rem;">{"RIGHT" if side=="R" else "LEFT"} BREAST</div>', unsafe_allow_html=True)
            if has:
                path = st.session_state.preview[side]["le" if modality == "DM" else "des"]
                preview = _preprocessed_preview(str(path), side)
                st.image(preview, caption="1024\u00d71024 model input", width='stretch')

                cam_overlay = st.session_state.cam_overlays.get(side, {}).get(modality)
                if cam_overlay is not None:
                    st.image(cam_overlay, caption="Model attention (Grad-CAM)", width='stretch')
                    st.caption(
                        "\u2139\ufe0f Vertical position (upper/lower) verified reliable on known cases. "
                        "Horizontal position (chest-wall side vs. periphery) showed a systematic bias in "
                        "testing and should NOT be read as a location claim -- view as an approximate "
                        "attention map, not a lesion pointer."
                    )
            else:
                st.caption("Not provided")

    if st.button(f"Generate {title} report (both breasts)", type="primary", width='stretch', key=f"generate_{modality}"):
        with st.spinner(f"Running classifier + narrative generation for {title} ({'~15-40s' if has_r and has_l else '~10-20s'})..."):
            _run_modality(modality)
        st.rerun()

    if right_section is not None or left_section is not None:
        ground_truth = _load_ground_truth(st.session_state.active_example)

        st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)
        rcols = st.columns(2)
        for col, side_label, side_code, section in (
            (rcols[0], "RIGHT", "R", right_section),
            (rcols[1], "LEFT", "L", left_section),
        ):
            with col:
                if section is None:
                    st.markdown(
                        f'<div class="rr-card rr-muted rr-mono" style="font-size:0.8rem;">{side_label} -- not generated</div>',
                        unsafe_allow_html=True,
                    )
                    continue

                gt = None
                if ground_truth is not None:
                    gt = ground_truth.get(modality, {}).get(side_code)

                if gt is not None:
                    gt_group = gt["group"]
                    gt_css = CATEGORY_CSS_CLASS.get(gt_group, "suspicious")
                    match = gt_group == section.classifier_result.birads_group
                    match_html = (
                        f'<span class="rr-mono" style="color:var(--benign);font-size:0.78rem;">\u2713 MATCH</span>'
                        if match
                        else f'<span class="rr-mono" style="color:var(--malignant);font-size:0.78rem;">\u2715 MISMATCH</span>'
                    )
                    st.markdown(
                        f"""
<div class="rr-card rr-card-{gt_css}" style="margin-bottom:6px;">
  <div class="rr-mono" style="font-size:0.68rem;color:var(--muted);letter-spacing:0.05em;">REAL FINDING (radiologist report, BI-RADS {gt['birads']})</div>
  <span class="rr-badge rr-badge-{gt_css}">{category_emoji(gt_group)} {gt_group}</span>
  {match_html}
  <div class="rr-narrative-text" style="margin-top:6px;font-size:0.85rem;">{gt['finding']}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                group = section.classifier_result.birads_group
                conf = section.classifier_result.birads_group_probs[group]
                css_class = CATEGORY_CSS_CLASS.get(group, "suspicious")
                color = category_color(group)
                cancer_html = ""
                if section.classifier_result.cancer:
                    cancer_html = (
                        f'<div class="rr-cancer-flag">\u26a0 CANCER FLAG POSITIVE '
                        f'(p={section.classifier_result.cancer_prob:.2f})</div>'
                    )
                gt_label = "GENERATED (model output)" if gt is not None else "MODEL OUTPUT"
                st.markdown(
                    f"""
<div class="rr-card rr-card-{css_class}">
  <div class="rr-mono" style="font-size:0.68rem;color:var(--muted);letter-spacing:0.05em;">{gt_label}</div>
  <span class="rr-badge rr-badge-{css_class}">{category_emoji(group)} {group}</span>
  <span class="rr-mono" style="color:var(--muted);font-size:0.78rem;margin-left:8px;">{conf:.0%} confidence</span>
  {_confidence_bar(conf, color)}
  {cancer_html}
  <div class="rr-narrative-label">Typical / illustrative language for this category -- not a confirmed finding for this image</div>
  <div class="rr-narrative-text">{section.narrative}</div>
</div>
""",
                    unsafe_allow_html=True,
                )


# --------------------------------------------------------------------------
# Sequential sections: LE first, then DES -- matches real report structure
# --------------------------------------------------------------------------
_render_modality_section(
    "1", "Low Energy (LE / DM)", "Pre-contrast digital mammography -- density & architecture",
    "DM", report.dm_right, report.dm_left,
)
_render_modality_section(
    "2", "Contrast-Enhanced (DES / CESM)", "Dual-energy subtracted -- enhancement pattern",
    "CESM", report.cesm_right, report.cesm_left,
)

if not report_is_empty(report):
    st.markdown('<div class="rr-section-header"><span class="rr-section-step">3</span> Full Report</div>', unsafe_allow_html=True)
    with st.expander("View combined plain-text report (source dataset structure)", expanded=False):
        st.code(report.format(), language=None)

# --------------------------------------------------------------------------
# Methodology
# --------------------------------------------------------------------------
st.divider()
with st.expander("\U0001f4ca Methodology & Validation -- what's actually verified here"):
    st.markdown(
        """
**Category prediction** (trained ConvNeXtV2-tiny classifier, per modality):
- Cross-validated end-to-end against 10 held-out patients (36 breast/modality cases):
  **32/36 (89%)** category-level agreement with independently-verified ground truth.
- Excluding one documented hard case (unusually dense/heterogeneous tissue, confirmed via
  a full elimination process -- ruled out EXIF/file corruption and preprocessing pixel-fidelity
  before concluding it's a genuine model limitation): **31/32 (97%)**.
- BI-RADS 1-6 digits are grouped into 3 coarse categories (Benign / Suspicious / Malignant)
  during training -- the model does **not** predict the individual digit (e.g. it cannot tell
  BI-RADS 3 from 4, or subcategorize 4A/4B/4C).

**Narrative description** (vision-LLM, conditioned on the classifier's category):
- A controlled ablation test showed this system's free-text narrative does **not** reliably
  read fine-grained visual details (location, exact margin type) from the image -- two
  genuinely different real images (one benign, one malignant) produced near-identical
  narrative text when given the same category label (74% word overlap).
- Because of this, narrative text is explicitly generated as **general, category-typical
  language**, not a claim about a specific visual feature in the uploaded image, and is
  labeled as such throughout this app.

**Not covered by this pipeline:**
- ACR breast density category (no trained classifier for this in the project).
- The individual BI-RADS digit/subcategory (only the coarse 3-way group).
"""
    )

st.caption("Built on the public CDD-CESM dataset (Khaled et al., Scientific Data 2022).")