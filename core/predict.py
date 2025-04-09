from core.patient import render_patient_bar
import streamlit as st
import numpy as np
import pandas as pd
import nibabel as nib
import joblib
import os
import shap
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import sobel


FEATURE_COLS = [
    "wt_vol", "tc_vol", "et_vol", "ncr_vol", "ed_vol",
    "et_wt_ratio", "et_tc_ratio", "tc_wt_ratio", "ed_wt_ratio",
    "ncr_tc_ratio", "sphericity", "surface_area", "n_components", "et_present",
]


def extract_grade_features(seg_path: str) -> dict:
    seg = nib.load(seg_path).get_fdata()
    seg[seg == 4] = 3

    wt  = (seg > 0).astype(float)
    tc  = ((seg == 1) | (seg == 3)).astype(float)
    et  = (seg == 3).astype(float)
    ncr = (seg == 1).astype(float)
    ed  = (seg == 2).astype(float)

    wt_vol  = float(wt.sum())
    tc_vol  = float(tc.sum())
    et_vol  = float(et.sum())
    ncr_vol = float(ncr.sum())
    ed_vol  = float(ed.sum())

    eps          = 1e-6
    et_wt_ratio  = et_vol  / (wt_vol  + eps)
    et_tc_ratio  = et_vol  / (tc_vol  + eps)
    tc_wt_ratio  = tc_vol  / (wt_vol  + eps)
    ed_wt_ratio  = ed_vol  / (wt_vol  + eps)
    ncr_tc_ratio = ncr_vol / (tc_vol  + eps)

    sx = sobel(wt, axis=0)
    sy = sobel(wt, axis=1)
    sz = sobel(wt, axis=2)
    surface_area = float(np.sqrt(sx**2 + sy**2 + sz**2).sum())

    sphericity = 0.0
    if surface_area > 0 and wt_vol > 0:
        sphericity = float(
            (np.pi ** (1/3)) * ((6 * wt_vol) ** (2/3)) / surface_area
        )

    _, n_components = ndimage.label(wt > 0)

    return {
        "wt_vol":       wt_vol,
        "tc_vol":       tc_vol,
        "et_vol":       et_vol,
        "ncr_vol":      ncr_vol,
        "ed_vol":       ed_vol,
        "et_wt_ratio":  et_wt_ratio,
        "et_tc_ratio":  et_tc_ratio,
        "tc_wt_ratio":  tc_wt_ratio,
        "ed_wt_ratio":  ed_wt_ratio,
        "ncr_tc_ratio": ncr_tc_ratio,
        "sphericity":   sphericity,
        "surface_area": surface_area,
        "n_components": float(n_components),
        "et_present":   float(1 if et_vol > 50 else 0),
    }


@st.cache_resource(show_spinner=False)
def load_grade_model(model_path: str):
    return joblib.load(model_path)


def rule_based_grade(features: dict) -> tuple:
    et_vol      = features["et_vol"]
    et_wt_ratio = features["et_wt_ratio"]

    if et_vol > 500 or et_wt_ratio > 0.15:
        return (
            "HGG",
            min(0.70 + et_wt_ratio * 0.5, 0.95),
            f"ET volume {et_vol:.0f} vx ({et_wt_ratio*100:.1f}% of WT). "
            "Strong enhancement indicates high-grade biology.",
        )
    elif et_vol > 50:
        return (
            "HGG",
            0.65,
            f"Small enhancing region detected ({et_vol:.0f} vx). "
            "Moderate suspicion — clinical correlation advised.",
        )
    return (
        "LGG",
        0.72,
        f"No significant enhancement detected (ET = {et_vol:.0f} vx). "
        "Low-grade gliomas typically lack contrast enhancement.",
    )


def _render_shap(model, x_scaled, feat_cols, grade):
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(x_scaled)

        if isinstance(shap_vals, list):
            idx = 1 if grade == "HGG" else 0
            sv  = shap_vals[idx][0]
            bv  = explainer.expected_value[idx]
        else:
            sv = shap_vals[0]
            bv = explainer.expected_value
            if isinstance(bv, (list, np.ndarray)):
                bv = bv[0]
            if grade == "LGG":
                sv = -sv
                bv = -bv

        fig, _ = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(
            shap.Explanation(values=sv, base_values=bv, feature_names=feat_cols),
            show=False,
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    except ImportError:
        st.info("Install SHAP to enable model explanations: pip install shap")
    except Exception as e:
        st.warning(f"SHAP plot unavailable: {e}")


def predict_tumor():
    render_patient_bar()
    st.subheader("Grade Prediction — HGG vs LGG")

    if "seg_path" not in st.session_state:
        st.warning("Run segmentation first (Step 1b).")
        return

    with st.spinner("Extracting features from segmentation mask..."):
        features = extract_grade_features(st.session_state["seg_path"])
    st.session_state["grade_features"] = features

    model_path  = st.text_input(
        "Model path (xgboost_grade_model.pkl)",
        value=st.session_state.get("grade_model_path", "models/xgboost_grade_model.pkl"),
    )
    st.session_state["grade_model_path"] = model_path
    use_trained = os.path.exists(model_path)

    if not use_trained:
        st.info(
            "Trained model not found — using evidence-based rule classifier. "
        )

    if not st.button("Predict Grade", type="primary"):
        with st.expander("Extracted segmentation features"):
            st.dataframe(
                pd.DataFrame([
                    {"Feature": k, "Value": f"{v:.4f}"}
                    for k, v in features.items()
                ]),
                use_container_width=True,
                hide_index=True,
            )
        return

    model_loaded = False
    xgb_model    = None
    feat_cols    = FEATURE_COLS
    x_scaled     = None

    if use_trained:
        try:
            saved        = load_grade_model(model_path)
            xgb_model    = saved["model"]
            scaler       = saved["scaler"]
            feat_cols    = saved.get("feature_cols", FEATURE_COLS)
            x_raw        = np.array([[features.get(c, 0) for c in feat_cols]])
            x_scaled     = scaler.transform(x_raw)
            prob         = xgb_model.predict_proba(x_scaled)[0]
            hgg_prob     = float(prob[1])
            lgg_prob     = float(prob[0])
            grade        = "HGG" if hgg_prob >= 0.5 else "LGG"
            confidence   = max(hgg_prob, lgg_prob)
            explanation  = (
                f"Confidence {confidence*100:.1f}% — "
                f"ET {features['et_vol']:.0f} vx, "
                f"ET/WT {features['et_wt_ratio']*100:.1f}%."
            )
            method       = ""
            model_loaded = True
        except Exception as e:
            st.warning(f"Model load error: {e}. Falling back to rule classifier.")

    if not model_loaded:
        grade, confidence, explanation = rule_based_grade(features)
        hgg_prob = confidence if grade == "HGG" else 1 - confidence
        lgg_prob = 1 - hgg_prob
        method   = "Rule-based classifier"

    color = "#E24B4A" if grade == "HGG" else "#378ADD"
    icon  = "" if grade == "HGG" else "✅"
    label = "High Grade Glioma" if grade == "HGG" else "Low Grade Glioma"

    st.markdown(
        f"""<div style="background:{color}14;border:1.5px solid {color};
            border-radius:8px;padding:18px 22px;margin:14px 0;">
            <div style="font-size:24px;font-weight:600;color:{color}">
                {icon} {grade} — {label}
            </div>
            <div style="font-size:13px;color:#aaa;margin-top:6px">{explanation}</div>
            <div style="font-size:11px;color:#666;margin-top:4px">Method: {method}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c1.metric("HGG Probability", f"{hgg_prob*100:.1f}%")
    c2.metric("LGG Probability", f"{lgg_prob*100:.1f}%")
    st.progress(hgg_prob)

    if grade == "HGG":
        st.error(
            "**High Grade Glioma (Grade IV)** — Aggressive, fast-growing. "
            "Median survival 12–18 months. Standard treatment: resection + "
            "radiotherapy + temozolomide chemotherapy (Stupp protocol)."
        )
    else:
        st.success(
            "**Low Grade Glioma (Grade II/III)** — Slower growing. "
            "Median survival 5–15 years. Management: watchful waiting or "
            "surgery/radiation based on location and neurological risk."
        )

    st.caption(
        "⚠️ For research and educational use only. "
        "Clinical diagnosis requires histopathological confirmation."
    )

    st.session_state["ml_prediction"] = {
        "class":         grade,
        "confidence":    confidence,
        "probabilities": {"HGG": hgg_prob, "LGG": lgg_prob},
        "classes":       ["LGG", "HGG"],
    }

    if model_loaded:
        st.markdown("---")
        st.subheader("Model Explanation (SHAP)")
        st.caption(
            "Each bar shows how much one feature pushed the prediction "
            "toward HGG (red) or LGG (blue)."
        )
        with st.spinner("Computing SHAP values..."):
            _render_shap(xgb_model, x_scaled, feat_cols, grade)

    with st.expander("Extracted segmentation features"):
        st.dataframe(
            pd.DataFrame([
                {"Feature": k, "Value": f"{v:.4f}"}
                for k, v in features.items()
            ]),
            use_container_width=True,
            hide_index=True,
        )
