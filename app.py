# app.py
# ─────────────────────────────────────────────────────────────────────────────
# NeuroAI — Brain Tumour MRI Analysis System
# DynUNet | SwinUNETR | SegResNet | BraTS 2020
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import logging

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(module)s: %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)

st.set_page_config(
    page_title         = "NeuroAI Intellegence System",
    page_icon          = "🧠",
    layout             = "wide",
    initial_sidebar_state = "expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    
}

/* Step indicator in sidebar */
.step-row {
    display: flex;
    align-items: center;
    padding: 5px 0;
    font-size: 13px;
}
.step-dot-done   { color: #3FB950; font-size: 10px; margin-right: 8px; }
.step-dot-active { color: #58A6FF; font-size: 10px; margin-right: 8px; }
.step-dot-idle   { color: #484F58; font-size: 10px; margin-right: 8px; }
.step-label-done   { color: #8B949E; }
.step-label-active { color: #E6EDF3; font-weight: 600; }
.step-label-idle   { color: #484F58; }

/* Nav buttons — clean left-aligned */
[data-testid="stSidebar"] button {
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 9px 12px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    margin-bottom: 3px !important;
}

[data-testid="stSidebar"] {
    background-color: #0D1117 !important;
    border-right: 1px solid #21262D !important;
}

/* Page content */
p, li, span, label { font-size: 14px !important; line-height: 1.6 !important; }
h1 { font-size: 22px !important; font-weight: 700 !important; }
h2 { font-size: 18px !important; font-weight: 600 !important; }
h3 { font-size: 15px !important; font-weight: 600 !important; color: #58A6FF !important; }

/* Metric cards */
[data-testid="stMetricValue"]  { font-size: 22px !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"]  { font-size: 12px !important; color: #8B949E !important; }

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #58A6FF !important;
    font-weight: 600 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th { font-size: 13px !important; }

/* Alerts */
[data-testid="stAlert"] p { font-size: 13px !important; }

/* Captions */
[data-testid="stCaptionContainer"] p { font-size: 12px !important; color: #8B949E !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed #30363D !important;
    border-radius: 8px !important;
    padding: 10px !important;
}

/* Divider */
hr { border-color: #21262D !important; }
</style>
""", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from core.uploader             import upload_mri_files
from core.auto_segment         import auto_segment_ui
from core.viewer               import interactive_viewer
from core.feature_report       import doctor_feature_report
from core.followup             import patient_followup
from core.tumor_habitate       import tumor_habitat_analysis
from core.predict              import predict_tumor
from core.dashboard            import clinical_dashboard
from core.longitudinal_compare import timestamp_comparison
from core.pipeline             import render_pipeline_status, get_status
from core.patient              import (
    is_registered, render_registration,
    render_patient_bar, render_sidebar_patient,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline step definitions
# ─────────────────────────────────────────────────────────────────────────────

STEPS = [
    ("upload",    "Upload MRI"),
    ("segment",   "Segmentation"),
    ("viewer",    "MRI Viewer"),
    ("features",  "Feature Report"),
    ("habitat",   "Habitat Analysis"),
    ("ml",        "Grade Prediction"),
    ("dashboard", "Dashboard"),
    ("followup",  "Follow-Up"),
    ("compare",   "Compare Visits"),
]

def _step_done(key: str) -> bool:
    checks = {
        "upload":    "flair_path"    in st.session_state,
        "segment":   st.session_state.get("seg_auto", False),
        "viewer":    st.session_state.get("seg_auto", False),
        "features":  "all_features"  in st.session_state,
        "habitat":   "habitat_map"   in st.session_state,
        "ml":        "ml_prediction" in st.session_state,
        "dashboard": "all_features"  in st.session_state,
        "followup":  False,
        "compare":   False,
    }
    return checks.get(key, False)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        # Logo + title
        st.markdown(
            "<div style='padding:12px 4px 8px 4px;'>"
            "<span style='font-size:22px;font-weight:700;color:#E6EDF3;'>🧠 NeuroAI Intellegence System</span><br>"
            "<span style='font-size:12px;color:#8B949E;'>Brain Tumour Analysis System</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Patient info block
        render_sidebar_patient()
        

        st.divider()

        # Navigation buttons with step indicator dots
        current = st.session_state.get("page", "upload")

        for key, label in STEPS:
            done   = _step_done(key)
            active = current == key

            if done:
                dot_cls   = "step-dot-done"
                label_cls = "step-label-done"
                dot       = "●"
            elif active:
                dot_cls   = "step-dot-active"
                label_cls = "step-label-active"
                dot       = "●"
            else:
                dot_cls   = "step-dot-idle"
                label_cls = "step-label-idle"
                dot       = "○"

            # Button — full width, styled by type
            btn_type = "primary" if active else "secondary"
            if st.button(
                label,
                key        = f"nav_{key}",
                use_container_width = True,
                type       = btn_type,
            ):
                st.session_state["page"] = key
                st.rerun()

        # Pipeline status (background AI progress)
        render_pipeline_status()

        st.divider()

    

        # Reset session
        if st.button(
            "New Patient Session",
            use_container_width = True,
            help = "Clear all data and start fresh for a new patient.",
        ):
            st.cache_resource.clear()
            st.cache_data.clear()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    return st.session_state.get("page", "upload")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if "page" not in st.session_state:
        st.session_state["page"] = "upload"

    page = _sidebar()

    # Registration gate
    if not is_registered():
        render_registration()
        return

   

    # Route to correct page
    routes = {
        "upload":    upload_mri_files,
        "segment":   auto_segment_ui,
        "viewer":    interactive_viewer,
        "features":  doctor_feature_report,
        "habitat":   tumor_habitat_analysis,
        "ml":        predict_tumor,
        "dashboard": clinical_dashboard,
        "followup":  patient_followup,
        "compare":   timestamp_comparison,
    }

    fn = routes.get(page)
    if fn:
        fn()


if __name__ == "__main__":
    main()
