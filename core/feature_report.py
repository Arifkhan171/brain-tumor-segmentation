from core.patient import render_patient_bar
import numpy as np
import pandas as pd
import nibabel as nib
import streamlit as st
import os
from scipy import stats
from scipy.ndimage import binary_dilation
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import sobel

from utils.file_ops import load_nifti
from utils.plotting import plot_feature_radar, normalize   # ← import normalize, no local copy

# ─────────────────────────────────────────────────────────────────────────────
# Clinical Feature Definitions & Metadata
# ─────────────────────────────────────────────────────────────────────────────

CLINICAL_METADATA = {
    "Volumetric Analysis": {
        "wt_vol": {
            "label": "Whole Tumor Volume (WT)",
            "definition": "The total volume of the entire tumor mass, including the enhancing core, non-enhancing core, and peritumoral edema (Labels 1, 2, and 3)."
        },
        "et_vol": {
            "label": "Enhancing Tumor Volume (ET)",
            "definition": "The volume of the tumor component that shows hyperintensity on T1-weighted contrast-enhanced images, representing blood-brain barrier breakdown."
        },
        "ed_vol": {
            "label": "Peritumoral Edema Volume (ED)",
            "definition": "The volume of hyperintense signal on FLAIR/T2 images surrounding the tumor, representing vasogenic edema or infiltrative tumor cells."
        },
        "core_vol": {
            "label": "Tumor Core Volume",
            "definition": "The combined volume of the enhancing tumor and the necrotic/non-enhancing core (Labels 1 and 3). Usually targeted for surgical resection."
        },
        "enhancement_ratio": {
            "label": "Enhancement Ratio",
            "definition": "The proportion of the whole tumor that is enhancing (ET/WT). Higher ratios are often associated with higher-grade malignancies."
        }
    },
    "Size & Morphology": {
        "max_diameter_3d": {
            "label": "Maximum 3D Diameter",
            "definition": "The largest linear dimension of the whole tumor mass measured across any of the three primary planes (Axial, Sagittal, or Coronal)."
        },
        "axial_diameter": {
            "label": "Major Axial Diameter",
            "definition": "The longest diameter of the tumor measured on the single axial slice where the tumor area is largest (Standard RANO measurement)."
        },
        "x_size": {"label": "Lateral Dimension (X)", "definition": "The maximum width of the tumor along the left-to-right axis."},
        "y_size": {"label": "Anteroposterior Dimension (Y)", "definition": "The maximum depth of the tumor along the front-to-back axis."},
        "z_size": {"label": "Craniocaudal Dimension (Z)", "definition": "The maximum height of the tumor along the top-to-bottom axis."}
    },
    "Tissue Intensity & Texture": {
        "mean": {
            "label": "Mean FLAIR Intensity",
            "definition": "The average signal intensity of the tumor pixels on the FLAIR sequence. Reflects the overall water content and cellularity."
        },
        "std": {
            "label": "Intensity Heterogeneity (Std)",
            "definition": "The standard deviation of pixel intensities. Higher values indicate a more 'mixed' or heterogeneous tumor tissue composition."
        },
        "p90": {
            "label": "90th Percentile Intensity",
            "definition": "The intensity value below which 90% of tumor pixels fall. High values often correlate with dense solid tumor components."
        },
        "contrast": {
            "label": "Textural Contrast (GLCM)",
            "definition": "A measure of the local variations in pixel intensity. High contrast indicates a rugged, complex tissue texture."
        },
        "t1ce_mean": {
            "label": "Mean T1CE Intensity",
            "definition": "Average signal on contrast-enhanced T1. High values directly correlate with aggressive blood-brain barrier breakdown (Enhancing Tumor)."
        },
        "t2_mean": {
            "label": "Mean T2 Intensity",
            "definition": "Average signal on T2 sequence. Highly sensitive to water content and cysts."
        }
    },
    "Peritumoral Environment": {
        "peri_mean": {
            "label": "Peritumoral Mean FLAIR",
            "definition": "The average signal intensity in the 5mm zone immediately surrounding the tumor. Used to assess the severity of edema."
        },
        "peri_std": {
            "label": "Peritumoral Heterogeneity",
            "definition": "The variation in intensity around the tumor boundary. May help distinguish between pure edema and infiltrative tumor."
        },
        "t1ce_peri_mean": {
            "label": "Peritumoral Mean T1CE",
            "definition": "Average T1CE signal surrounding the tumor. Abnormal enhancement here may indicate microscopic infiltrative disease beyond the margin."
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _best_tumor_slice(mask_3d):
    """Return the z-index of the axial slice with the most tumor voxels."""
    counts = (mask_3d > 0).sum(axis=(0, 1))   # vectorised — no Python loop
    return int(np.argmax(counts)) if counts.max() > 0 else mask_3d.shape[2] // 2


def _get_voxel_volume_mm3(nifti_path: str) -> float:
    """Return the volume of a single voxel in mm³ from the NIfTI header."""
    try:
        zooms = nib.load(nifti_path).header.get_zooms()[:3]
        return float(np.prod(zooms))
    except Exception:
        return 1.0   # safe fallback — 1mm isotropic


def _intensity_features(voxels, prefix=""):
    """Intensity statistics from raw MRI voxels (3D)."""
    if voxels.size == 0:
        keys = ["mean", "std", "p10", "p25", "p75", "p90", "skew", "kurtosis"]
        return {f"{prefix}{k}": 0.0 for k in keys}
    return {
        f"{prefix}mean":     float(np.mean(voxels)),
        f"{prefix}std":      float(np.std(voxels)),
        f"{prefix}p10":      float(np.percentile(voxels, 10)),
        f"{prefix}p25":      float(np.percentile(voxels, 25)),
        f"{prefix}p75":      float(np.percentile(voxels, 75)),
        f"{prefix}p90":      float(np.percentile(voxels, 90)),
        f"{prefix}skew":     float(stats.skew(voxels)),
        f"{prefix}kurtosis": float(stats.kurtosis(voxels)),
    }


def _glcm_features_display(intensity_slice, mask_slice):
    """GLCM for display table only — NOT used for ML prediction."""
    if intensity_slice.max() == 0:
        return dict(contrast=0, correlation=0, energy=0, homogeneity=0)
    img = (normalize(intensity_slice) * 63).astype(np.uint8)
    img[~mask_slice.astype(bool)] = 0
    glcm = graycomatrix(
        img, distances=[1], angles=[0, np.pi / 4, np.pi / 2],
        levels=64, symmetric=True, normed=True,
    )
    return dict(
        contrast    = float(graycoprops(glcm, "contrast").mean()),
        correlation = float(graycoprops(glcm, "correlation").mean()),
        energy      = float(graycoprops(glcm, "energy").mean()),
        homogeneity = float(graycoprops(glcm, "homogeneity").mean()),
    )


def _peritumoral_features(mri, wt_mask, radius=5, prefix="peri_"):
    expanded  = binary_dilation(wt_mask, iterations=radius)
    peri_zone = expanded & ~wt_mask
    return _intensity_features(mri[peri_zone], prefix=prefix)


def _ml_ready_features(mri_3d, mask_3d):
    """
    Extract the 13 features the XGBoost model was trained on.
    Kept separate from clinical view.
    """
    best_z     = _best_tumor_slice(mask_3d)
    slice_img  = mri_3d[:, :, best_z]
    slice_mask = mask_3d[:, :, best_z]

    mn, mx   = slice_img.min(), slice_img.max()
    slice_01 = (slice_img - mn) / (mx - mn + 1e-8)

    tumor_px = slice_01[slice_mask > 0]
    std    = float(np.std(tumor_px))    if tumor_px.size > 0 else 0.0
    median = float(np.median(tumor_px)) if tumor_px.size > 0 else 0.0

    slice_u8  = (slice_01 * 255).astype(np.uint8)
    masked_u8 = slice_u8.copy()
    masked_u8[slice_mask == 0] = 0

    glcm     = graycomatrix(
        masked_u8, distances=[1], angles=[0],
        levels=256, symmetric=True, normed=False,
    ).astype(np.float64)

    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0
    if np.sum(glcm) > 0:
        glcm /= np.sum(glcm)

    contrast = float(graycoprops(glcm, "contrast")[0, 0])
    energy   = float(graycoprops(glcm, "energy")[0, 0])

    fft        = np.fft.fft2(slice_01)
    fft_energy = float(np.sum(np.abs(fft) ** 2))

    edges       = sobel(slice_01)
    edge_masked = edges[slice_mask > 0]
    edge_density = (
        float(np.sum(edge_masked > np.mean(edge_masked)) / edge_masked.size)
        if edge_masked.size > 0 else 0.0
    )

    lbp      = local_binary_pattern(slice_u8, P=8, R=1, method="uniform")
    lbp_vals = lbp[slice_mask > 0]
    if lbp_vals.size > 0:
        hist, _ = np.histogram(lbp_vals, bins=10, range=(0, 10))
        hist    = hist / (hist.sum() + 1e-8)
    else:
        hist = np.zeros(10)

    result = {
        "std":          std,
        "median":       median,
        "contrast":     contrast,
        "energy":       energy,
        "fft_energy":   fft_energy,
        "edge_density": edge_density,
    }
    for i in range(10):
        result[f"lbp_{i}"] = float(hist[i])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PDF builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_pdf(df_clinical, report_text, ml_prediction=None, patient_info=None):
    """
    Build a professional neuro-oncology PDF report.
    Follows standard radiology report layout used in clinical practice.
    """
    try:
        from fpdf import FPDF
        from datetime import date
    except ImportError:
        return None

    if patient_info is None:
        patient_info = {}

    pdf = FPDF()
    pdf.add_font("DejaVu", "", r"C:\Windows\Fonts\arial.ttf")
    pdf.set_font("Arial", "", 13)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(18, 18, 18)

    # ── Header ────────────────────────────────────────────────────────────────
    pdf.set_font("DejaVu", "", 13)
    pdf.set_text_color(15, 30, 60)
    pdf.cell(0, 8, "NEURO-ONCOLOGY MRI ANALYSIS REPORT", align="C",
             new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("DejaVu", "", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5,
             f"NeuroAI v2.0  ·  DynUNet BraTS 2020  ·  "
             f"Generated: {date.today().strftime('%d %B %Y')}",
             align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_draw_color(15, 30, 60)
    pdf.set_line_width(0.5)
    pdf.line(18, pdf.get_y() + 2, 192, pdf.get_y() + 2)
    pdf.ln(5)

    # ── Patient Demographics ───────────────────────────────────────────────────
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(15, 30, 60)
    pdf.cell(0, 6, "PATIENT INFORMATION", new_x="LMARGIN", new_y="NEXT")
    pdf.set_line_width(0.2)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(18, pdf.get_y(), 192, pdf.get_y())
    pdf.ln(2)

    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(40, 40, 40)

    demo_left = [
        ("Patient Name",  patient_info.get("name",     "Not registered")),
        ("Patient ID",    patient_info.get("id",       "—")),
        ("Age / Sex",     f"{patient_info.get('age','—')} y/o  {patient_info.get('sex','—')}"),
    ]
    demo_right = [
        ("Scan Date",     str(patient_info.get("scan_date", date.today()))),
        ("Referring Physician", patient_info.get("referring_doctor", "—")),
        ("Report System", "NeuroAI — AI-Assisted Analysis"),
    ]

    col_w = 87
    for (lbl_l, val_l), (lbl_r, val_r) in zip(demo_left, demo_right):
        pdf.set_font("DejaVu", "", 8)
        pdf.cell(30, 6, lbl_l + ":", border=0)
        pdf.set_font("DejaVu", "", 8)
        pdf.cell(col_w - 30, 6, val_l, border=0)
        pdf.set_font("DejaVu", "", 8)
        pdf.cell(30, 6, lbl_r + ":", border=0)
        pdf.set_font("DejaVu", "", 8)
        pdf.cell(0, 6, val_r, border=0, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(3)

    # ── AI Classification Banner ───────────────────────────────────────────────
    if ml_prediction:
        cls  = ml_prediction.get("class", "")
        conf = ml_prediction.get("confidence", 0)
        is_hgg = cls == "HGG"

        if is_hgg:
            pdf.set_fill_color(255, 235, 235)
            pdf.set_text_color(160, 20, 20)
        else:
            pdf.set_fill_color(232, 242, 255)
            pdf.set_text_color(20, 60, 140)

        pdf.set_font("DejaVu", "", 10)
        label = "High Grade Glioma (HGG)" if is_hgg else "Low Grade Glioma (LGG)"
        pdf.cell(
            0, 10,
            f"AI Classification:  {label}  —  {conf:.1%} confidence  "
            f"(XGBoost · BraTS 2020 · 87.8% balanced accuracy)",
            fill=True, align="C", new_x="LMARGIN", new_y="NEXT",
        )
        pdf.set_text_color(40, 40, 40)
        pdf.ln(4)

    # ── Clinical Measurements Table ────────────────────────────────────────────
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(15, 30, 60)
    pdf.cell(0, 6, "QUANTITATIVE MEASUREMENTS", new_x="LMARGIN", new_y="NEXT")
    pdf.set_line_width(0.2)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(18, pdf.get_y(), 192, pdf.get_y())
    pdf.ln(2)

    # Table header
    pdf.set_fill_color(240, 242, 245)
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("DejaVu", "", 8)
    pdf.cell(42, 6, "Category",  border=1, fill=True)
    pdf.cell(68, 6, "Parameter", border=1, fill=True)
    pdf.cell(32, 6, "Value",     border=1, fill=True)
    pdf.cell(32, 6, "Unit",      border=1, fill=True,
             new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("DejaVu", "", 8)
    pdf.set_text_color(40, 40, 40)

    alt = False
    for _, row in df_clinical.iterrows():
        if alt:
            pdf.set_fill_color(248, 249, 250)
        else:
            pdf.set_fill_color(255, 255, 255)
        alt = not alt

        val_str = str(row["Value"])
        # Separate value from unit if present
        if " " in val_str and "(" not in val_str:
            parts = val_str.rsplit(" ", 1)
            val_only = parts[0]
            unit_only = parts[1]
        elif "cm³" in val_str:
            val_only = val_str.replace(" cm³", "")
            unit_only = "cm³"
        elif "mm" in val_str:
            val_only = val_str.split(" mm")[0]
            unit_only = "mm"
        elif "(a.u.)" in val_str:
            val_only = val_str.replace(" (a.u.)", "")
            unit_only = "a.u."
        elif "(unitless)" in val_str:
            val_only = val_str.replace(" (unitless)", "")
            unit_only = "—"
        else:
            val_only = val_str
            unit_only = "—"

        pdf.cell(42, 6, str(row["Category"])[:20], border=1, fill=True)
        pdf.cell(68, 6, str(row["Feature"])[:38],  border=1, fill=True)
        pdf.cell(32, 6, val_only[:14],             border=1, fill=True)
        pdf.cell(32, 6, unit_only[:14],            border=1, fill=True,
                 new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)

    # ── Radiology Report Text ──────────────────────────────────────────────────
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(15, 30, 60)
    pdf.cell(0, 6, "RADIOLOGY REPORT", new_x="LMARGIN", new_y="NEXT")
    pdf.set_line_width(0.2)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(18, pdf.get_y(), 192, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 5, report_text.strip())
    pdf.ln(4)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    pdf.set_draw_color(180, 180, 180)
    pdf.line(18, pdf.get_y(), 192, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("DejaVu", "", 7)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(
        0, 4,
        "DISCLAIMER: This report is generated by NeuroAI, an AI-assisted "
        "research tool. All findings must be reviewed and confirmed by a "
        "qualified neuroradiologist or neuropathologist before clinical use. "
        "This report does not constitute a clinical diagnosis.",
    )

    # ── Signature line ─────────────────────────────────────────────────────────
    pdf.ln(6)
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(90, 5, "Reviewing Radiologist: ___________________________")
    pdf.cell(0,  5, f"Date: {date.today().strftime('%d / %m / %Y')}")

    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────────────────────
# LLM report generator
# ─────────────────────────────────────────────────────────────────────────────

def _generate_llm_report(feature_dict, ml_prediction=None):
    try:
        import openai, os
        api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No API key")
        openai.api_key = api_key

        context = "\n".join(
            f"- {k.replace('_', ' ').title()}: {v}"
            for k, v in feature_dict.items()
        )
        if ml_prediction:
            context += (
                f"\n- ML Classification: {ml_prediction.get('class', 'N/A')} "
                f"({ml_prediction.get('confidence', 0):.1%} confidence)"
            )

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                f"You are a consultant neuroradiologist. Write a professional, structured MRI brain "
                f"tumour report. Use standard clinical terminology. Sections: FINDINGS, IMPRESSION, RECOMMENDATION. "
                f"Focus on the volumetric data and diameters provided.\n\n{context}"
            }],
            max_tokens=400,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        pass

    # Fallback template — structured radiology report format
    from datetime import date as _date
    import streamlit as _st

    wt  = feature_dict.get("wt_vol", 0)
    et  = feature_dict.get("et_vol", 0)
    ed  = feature_dict.get("ed_vol", 0)
    er  = feature_dict.get("enhancement_ratio", 0)
    d   = feature_dict.get("max_diameter_3d", 0)
    ax  = feature_dict.get("axial_diameter", 0)

    grade_line = ""
    if ml_prediction:
        cls  = ml_prediction.get("class", "N/A")
        conf = ml_prediction.get("confidence", 0)
        grade_line = f"AI-Assisted Classification: {cls} ({conf:.1%} confidence)"

    enhancement_interp = (
        "Substantial contrast enhancement is present, constituting "
        f"{er*100:.1f}% of the whole tumour volume. "
        "This degree of enhancement is characteristic of high-grade glioma."
        if er > 0.3 else
        f"Enhancement ratio is {er:.3f}, representing minimal contrast uptake. "
        "This pattern is more consistent with low-grade glioma."
    )

    return f"""NEURORADIOLOGY REPORT
Date of Report : {_date.today().strftime('%d %B %Y')}
System         : NeuroAI v2.0 — AI-Assisted MRI Analysis

{'─' * 52}
CLINICAL INDICATION
{'─' * 52}
Brain MRI with automated tumour segmentation and quantitative analysis
using DynUNet (BraTS 2020). Radiomic features extracted for grading support.

{'─' * 52}
FINDINGS
{'─' * 52}
Volumetric Analysis:
  Whole Tumour Volume (WT)  : {wt:.2f} cm³
  Enhancing Component (ET)  : {et:.2f} cm³
  Peritumoral Oedema (ED)   : {ed:.2f} cm³
  Enhancement Ratio (ET/WT) : {er:.3f}

Morphology:
  Maximum 3D Diameter       : {d:.1f} mm
  Largest Axial Diameter    : {ax:.1f} mm  (RANO measurement)

{'─' * 52}
IMPRESSION
{'─' * 52}
A space-occupying lesion is identified with a total volume of {wt:.2f} cm³.
{enhancement_interp}
Peritumoral oedema measures {ed:.2f} cm³, indicating
{"extensive" if ed > 30 else "moderate" if ed > 10 else "mild"} surrounding vasogenic oedema.

{grade_line}

{'─' * 52}
RECOMMENDATION
{'─' * 52}
1. Correlation with clinical presentation and neurological examination.
2. Neurosurgical consultation for consideration of biopsy or resection.
3. Multidisciplinary neuro-oncology team review.
4. Follow-up MRI at interval as per institutional protocol.

{'─' * 52}
DISCLAIMER
{'─' * 52}
This report is AI-assisted and must be reviewed by a qualified
neuroradiologist before clinical use. It does not constitute a diagnosis.""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# SHAP panel — uses the correct grade model (models/xgboost_grade_model.pkl)
# ─────────────────────────────────────────────────────────────────────────────

def _render_shap_panel(ml_feats):
    """
    Show SHAP waterfall for the grade classifier (xgboost_grade_model.pkl).
    ml_feats must be the grade-feature dict (14 volumetric features), not
    the old texture-based pipeline.
    """
    import os
    grade_model_path = st.session_state.get("grade_model_path", "models/xgboost_grade_model.pkl")

    if not os.path.exists(grade_model_path):
        st.info(
            "SHAP explanation unavailable: trained grade model not found at "
            f"`{grade_model_path}`. Run Grade Prediction first to generate SHAP."
        )
        return

    try:
        import shap
        import joblib
        import matplotlib.pyplot as plt

        saved     = joblib.load(grade_model_path)
        xgb_model = saved["model"]
        scaler    = saved["scaler"]
        feat_cols = saved.get("feature_cols", [
            "wt_vol", "tc_vol", "et_vol", "ncr_vol", "ed_vol",
            "et_wt_ratio", "et_tc_ratio", "tc_wt_ratio", "ed_wt_ratio",
            "ncr_tc_ratio", "sphericity", "surface_area", "n_components", "et_present",
        ])

        X        = pd.DataFrame([{k: ml_feats.get(k, 0) for k in feat_cols}], columns=feat_cols)
        X_scaled = scaler.transform(X)

        explainer  = shap.TreeExplainer(xgb_model)
        shap_vals  = explainer.shap_values(X_scaled)
        pred_class = int(np.argmax(xgb_model.predict_proba(X_scaled)[0]))

        if isinstance(shap_vals, list):
            sv = shap_vals[pred_class][0]
            bv = explainer.expected_value[pred_class]
        else:
            if len(shap_vals.shape) == 3:
                sv = shap_vals[0, :, pred_class]
            else:
                sv = shap_vals[0]
            bv = (
                explainer.expected_value[pred_class]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )

        fig, _ = plt.subplots(figsize=(7, 4))
        shap.waterfall_plot(
            shap.Explanation(values=sv, base_values=bv, feature_names=feat_cols),
            show=False,
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    except ImportError:
        st.info("💡 Run `pip install shap` to enable Explainable AI (SHAP values).")
    except Exception as e:
        st.warning(f"Technical Explanation (SHAP) failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def doctor_feature_report():
    render_patient_bar()
    st.markdown("### 📋 Clinical Feature Report & Diagnostic Summary")
    st.info("This module extracts standardized radiomic metrics and generates a professional neuro-oncology report.")

    if "flair_path" not in st.session_state or "seg_path" not in st.session_state:
        st.warning("Please complete Step 1 (Upload & Segment) to view this report.")
        return

    flair_path = st.session_state["flair_path"]
    seg_path   = st.session_state["seg_path"]

    mri  = load_nifti(flair_path)
    mask = load_nifti(seg_path)

    if mri is None or mask is None:
        st.error("Could not load MRI or segmentation file. Please re-upload.")
        return

    # ── Read voxel spacing for physically accurate measurements ───────────────
    voxel_vol_mm3 = _get_voxel_volume_mm3(flair_path)
    voxel_dims    = nib.load(flair_path).header.get_zooms()[:3]   # (dx, dy, dz) mm

    WT   = mask > 0
    ET   = mask == 3
    ED   = mask == 2
    CORE = (mask == 1) | (mask == 3)

    # Volumes in cm³ (using actual voxel spacing, not assumed 1mm)
    wt_vol            = float(WT.sum())   * voxel_vol_mm3 / 1000
    et_vol            = float(ET.sum())   * voxel_vol_mm3 / 1000
    ed_vol            = float(ED.sum())   * voxel_vol_mm3 / 1000
    core_vol          = float(CORE.sum()) * voxel_vol_mm3 / 1000
    enhancement_ratio = et_vol / wt_vol if wt_vol > 0 else 0.0

    # Diameters in mm (multiply voxel count by spacing per axis)
    coords = np.argwhere(WT)
    if coords.shape[0] > 0:
        x_size = float((coords[:, 0].max() - coords[:, 0].min()) * voxel_dims[0])
        y_size = float((coords[:, 1].max() - coords[:, 1].min()) * voxel_dims[1])
        z_size = float((coords[:, 2].max() - coords[:, 2].min()) * voxel_dims[2])
    else:
        x_size = y_size = z_size = 0.0

    max_diameter_3d = max(x_size, y_size, z_size)
    best_z   = _best_tumor_slice(mask)
    slice_wt = WT[:, :, best_z]
    try:
        props      = regionprops(slice_wt.astype(np.int32))
        # major_axis_length is in pixels; convert to mm using in-plane spacing
        axial_diam = float(props[0].major_axis_length * voxel_dims[0]) if props else 0.0
    except Exception:
        axial_diam = 0.0

    tumor_vox = mri[WT]
    int_feats = _intensity_features(tumor_vox)
    glcm      = _glcm_features_display(mri[:, :, best_z], slice_wt)
    peri      = _peritumoral_features(mri, WT, radius=5)

    # Store for global access
    all_raw_data = {
        "wt_vol": wt_vol, "et_vol": et_vol, "ed_vol": ed_vol,
        "core_vol": core_vol, "enhancement_ratio": enhancement_ratio,
        "max_diameter_3d": max_diameter_3d, "axial_diameter": axial_diam,
        "x_size": x_size, "y_size": y_size, "z_size": z_size,
        **int_feats, **glcm, **peri
    }

    # ── Peritumoral features are always computed from FLAIR ──────────────────
    # peri_mean and peri_std come from _peritumoral_features above via **peri
    # They are already in all_raw_data — no need to re-add them

    # ── T1CE features ─────────────────────────────────────────────────────────
    # Remove the path comparison check — on Windows paths may differ in slash style
    # Just check the file exists and loads correctly
    # T1CE features
    t1ce_path = st.session_state.get("t1ce_path")
    if t1ce_path and os.path.exists(t1ce_path):
        t1ce = load_nifti(t1ce_path)
        if t1ce is not None:
            wt_bool = WT.astype(bool)
            t1ce_vox = t1ce[wt_bool]
            all_raw_data.update(_intensity_features(t1ce_vox, prefix="t1ce_"))
            peri_t1ce = _peritumoral_features(t1ce, wt_bool, radius=5, prefix="t1ce_peri_")
            all_raw_data.update(peri_t1ce)

    # T2 features
    t2_path = st.session_state.get("t2_path")
    if t2_path and os.path.exists(t2_path):
        t2 = load_nifti(t2_path)
        if t2 is not None:
            wt_bool = WT.astype(bool)
            t2_vox = t2[wt_bool]
            all_raw_data.update(_intensity_features(t2_vox, prefix="t2_"))

    # Fallback N/A only if still missing after loading attempts
    if "t1ce_mean" not in all_raw_data:
        all_raw_data["t1ce_mean"] = None
    if "t2_mean" not in all_raw_data:
        all_raw_data["t2_mean"] = None
    if "t1ce_peri_mean" not in all_raw_data:
        all_raw_data["t1ce_peri_mean"] = None

    st.session_state["all_features"]  = all_raw_data
    st.session_state["ml_features"]   = _ml_ready_features(mri, mask)

    # ── Grade features (for SHAP consistency with predict.py) ────────────────
    from core.predict import extract_grade_features
    grade_feats = extract_grade_features(seg_path)
    st.session_state["grade_features"] = grade_feats

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📊 Structured Clinical Table",
        "🕸️ Diagnostic Radar",
        "✍️ Automated Radiology Report",
        
    ])

    with tab1:
        st.markdown("### Clinical Metrics by Category")
        st.markdown("Below are the extracted features organized logically. Each group highlights a different aspect of the tumor's presentation.")

        table_data = []

        for category, features in CLINICAL_METADATA.items():
            st.markdown(f"#### 🔹 {category}")
            cat_data = []
            for key, meta in features.items():
                val = all_raw_data.get(key)
                if val is None:
                    val_str = "N/A (upload T1CE/T2 for this value)"
                else:
                    if "vol" in key:
                        val_str = f"{val:.2f} cm³"
                    elif "diameter" in key or "_size" in key:
                        val_str = f"{val:.1f} mm"
                    elif "ratio" in key:
                        val_str = f"{val:.3f} (unitless)"
                    elif key in ["contrast", "correlation", "energy", "homogeneity"]:
                        val_str = f"{val:.3f} (unitless)"
                    elif "mean" in key or "std" in key or key.startswith("p") or                          key in ["skew", "kurtosis", "peri_std"]:
                        val_str = f"{val:.2f} (MRI intensity)"
                    elif key == "sphericity":
                        val_str = f"{val:.3f} (0–1 scale)"
                    elif key == "surface_area":
                        val_str = f"{val:.1f} voxels²"
                    elif key == "n_components":
                        val_str = f"{int(val)} region(s)"
                    elif key == "et_present":
                        val_str = "Yes" if val == 1.0 else "No"
                    else:
                        val_str = f"{val:.2f}"

                row = {
                    "Feature":     meta["label"],
                    "Value":       val_str,
                    "Description": meta["definition"]
                }
                cat_data.append(row)
                table_data.append({"Category": category, **row})

            df_cat = pd.DataFrame(cat_data)
            st.dataframe(
                df_cat,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Feature":     st.column_config.TextColumn("Clinical Metric", width="medium"),
                    "Value":       st.column_config.TextColumn("Measured Value", width="small"),
                    "Description": st.column_config.TextColumn("Definition / Clinical Significance", width="large"),
                }
            )
            st.divider()

        df_clinical = pd.DataFrame(table_data)
        st.download_button(
            "Download Clinical Metrics (CSV)",
            df_clinical.to_csv(index=False).encode("utf-8"),
            "NeuroAI_Clinical_Report.csv",
            "text/csv",
        )

    with tab2:
        st.write("Visual comparison of key tumor characteristics:")
        fig_radar = plot_feature_radar(
            wt_vol, et_vol, ed_vol,
            enhancement_ratio, max_diameter_3d, int_feats["mean"],
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab3:
        ml_pred = st.session_state.get("ml_prediction")

        # Cache report so it is not regenerated every time tab is opened
        if "generated_report" not in st.session_state:
            with st.spinner("Compiling clinical findings..."):
                st.session_state["generated_report"] = _generate_llm_report(
                    all_raw_data, ml_pred
                )
        report_text = st.session_state["generated_report"]

        # Patient info for PDF
        from core.patient import get_info, is_registered
        patient_info = get_info() if is_registered() else {}

        # Report display
        pid   = patient_info.get("id",   "patient")
        pname = patient_info.get("name", "report")
        safe_name = pname.replace(" ", "_")

        st.caption("Review the report below. Download as PDF or plain text.")
        st.text_area("", report_text, height=380)
        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download as Text",
                report_text,
                f"NeuroAI_{safe_name}_{pid}.txt",
                "text/plain",
            )
        with c2:
            pdf_bytes = _build_pdf(df_clinical, report_text, ml_pred, patient_info)
            if pdf_bytes:
                st.download_button(
                    "Download PDF Report",
                    pdf_bytes,
                    f"NeuroAI_{safe_name}_{pid}.pdf",
                    "application/pdf",
                )
            else:
                st.caption("Install `fpdf2` to enable PDF export.")

   


# ─────────────────────────────────────────────────────────────────────────────
# Headless version — called by the background pipeline (no Streamlit UI)
# ─────────────────────────────────────────────────────────────────────────────

def doctor_feature_report_headless(flair_path: str, seg_path: str):
    """
    Compute all clinical features and save to session_state without
    rendering any Streamlit UI. Used by the background auto-pipeline.
    """
    from utils.file_ops import load_nifti, get_voxel_volume_mm3, get_voxel_dims, get_best_slice
    import numpy as np
    import nibabel as nib
    from skimage.measure import regionprops

    mri  = load_nifti(flair_path)
    mask = load_nifti(seg_path)
    if mri is None or mask is None:
        return

    voxel_vol_mm3 = get_voxel_volume_mm3(flair_path)
    voxel_dims    = get_voxel_dims(flair_path)

    WT   = mask > 0
    ET   = mask == 3
    ED   = mask == 2
    CORE = (mask == 1) | (mask == 3)

    wt_vol            = float(WT.sum())   * voxel_vol_mm3 / 1000
    et_vol            = float(ET.sum())   * voxel_vol_mm3 / 1000
    ed_vol            = float(ED.sum())   * voxel_vol_mm3 / 1000
    core_vol          = float(CORE.sum()) * voxel_vol_mm3 / 1000
    enhancement_ratio = et_vol / wt_vol if wt_vol > 0 else 0.0

    coords = np.argwhere(WT)
    if coords.shape[0] > 0:
        x_size = float((coords[:, 0].max() - coords[:, 0].min()) * voxel_dims[0])
        y_size = float((coords[:, 1].max() - coords[:, 1].min()) * voxel_dims[1])
        z_size = float((coords[:, 2].max() - coords[:, 2].min()) * voxel_dims[2])
    else:
        x_size = y_size = z_size = 0.0

    max_diameter_3d = max(x_size, y_size, z_size)

    best_z   = get_best_slice(mask, axis=(0, 1))
    slice_wt = WT[:, :, best_z]
    try:
        props      = regionprops(slice_wt.astype(np.int32))
        axial_diam = float(props[0].major_axis_length * voxel_dims[0]) if props else 0.0
    except Exception:
        axial_diam = 0.0

    tumor_vox = mri[WT.astype(bool)]
    int_feats = _intensity_features(tumor_vox)
    glcm      = _glcm_features_display(mri[:, :, best_z], slice_wt)
    peri      = _peritumoral_features(mri, WT.astype(bool), radius=5)

    all_raw_data = {
        "wt_vol":           wt_vol,
        "et_vol":           et_vol,
        "ed_vol":           ed_vol,
        "core_vol":         core_vol,
        "enhancement_ratio":enhancement_ratio,
        "max_diameter_3d":  max_diameter_3d,
        "axial_diameter":   axial_diam,
        "x_size":           x_size,
        "y_size":           y_size,
        "z_size":           z_size,
        **int_feats,
        **glcm,
        **peri,
    }

    import streamlit as st
    st.session_state["all_features"] = all_raw_data
