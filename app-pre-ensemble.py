# app.py
# ============================================================
# NeuroAI — Brain Tumour Analysis System
# Full pipeline with DynUNet automatic segmentation
# ============================================================

import streamlit as st
import plotly.graph_objects as go
import logging

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="NeuroAI - Brain Tumour Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global Font Size Fix ───────────────────────────────────────────────────────
st.markdown("""
    <style>
    /* Base font size — increase this number to make everything bigger */
    html, body, [class*="css"] {
        font-size: 16px !important;
    }

    /* Main content area text */
    .main p, .main li, .main span, .main label {
        font-size: 15px !important;
        line-height: 1.7 !important;
    }

    /* Headings */
    .main h1 { font-size: 2rem !important; }
    .main h2 { font-size: 1.6rem !important; }
    .main h3 { font-size: 1.3rem !important; }

    /* Sidebar text */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        font-size: 14px !important;
    }

    /* Sidebar buttons — bigger and easier to click */
    [data-testid="stSidebar"] button {
        font-size: 14px !important;
        padding: 10px 8px !important;
        margin-bottom: 4px !important;
    }

    /* Input fields, selectbox, text area */
    input, textarea, select {
        font-size: 15px !important;
    }

    /* Dataframe table text */
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th {
        font-size: 14px !important;
    }

    /* Metric value (big numbers) */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }

    /* Metric label */
    [data-testid="stMetricLabel"] {
        font-size: 13px !important;
    }

    /* Caption text */
    .main small, [data-testid="stCaptionContainer"] {
        font-size: 13px !important;
    }

    /* Info, warning, error, success boxes */
    [data-testid="stAlert"] p {
        font-size: 14px !important;
    }

    /* Tabs */
    [data-testid="stTabs"] button {
        font-size: 14px !important;
    }

    /* Expander header */
    [data-testid="stExpander"] summary {
        font-size: 14px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ── Module imports ────────────────────────────────────────────────────────────
from core.uploader         import upload_mri_files
from core.auto_segment     import auto_segment_ui          # NEW — DynUNet
from core.viewer           import interactive_viewer
from core.feature_report   import doctor_feature_report
from core.followup         import patient_followup
from core.tumor_habitate   import tumor_habitat_analysis
from core.predict          import predict_tumor            # UPDATED — HGG/LGG
from core.dashboard        import clinical_dashboard
from core.longitudinal_compare import timestamp_comparison
from core.pipeline             import render_pipeline_status
from core.patient              import (
    is_registered, render_registration,
    render_patient_bar, render_sidebar_patient,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
def _sidebar():
    with st.sidebar:
        st.markdown("## 🧠 NeuroAI")
        st.caption("Brain Tumour MRI Analysis System")
        st.divider()
        render_sidebar_patient()

        pages = {
            "📤  Upload MRI Files":        "upload",
            "🧠  Auto Segmentation":       "segment",    # NEW
            "🔬  MRI Viewer":              "viewer",
            "📋  Feature Report":          "features",
            "🗺️  Habitat Analysis":        "habitat",
            "🤖  Grade Prediction":        "ml",         # UPDATED
            "📊  Clinical Dashboard":      "dashboard",
            "📈  Follow-Up Monitor":       "followup",
            "⏱️  Timestamp Compare":       "compare",
        }

        if "page" not in st.session_state:
            st.session_state["page"] = "upload"

        for label, key in pages.items():
            active = st.session_state["page"] == key
            if st.button(
                label,
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state["page"] = key

        # Pipeline status — auto-updates as background pipeline runs
        render_pipeline_status()

        st.divider()
        st.caption("DynUNet Scores (BraTS 2020)")
        st.caption("WT 91.14% | TC 89.38% | ET 83.06%")

        st.divider()
        if st.button("🔄 Reset Session / Clear Memory", use_container_width=True, help="Clear all uploaded files and free memory for the next patient."):
            st.cache_resource.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    return st.session_state["page"]



# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    page = _sidebar()

    # Show registration form before anything else
    if not is_registered():
        render_registration()
        return

    # Patient context bar at top of every page
    render_patient_bar()

    if page == "upload":
        upload_mri_files()
    elif page == "segment":
        auto_segment_ui()          # NEW — DynUNet automatic segmentation
    elif page == "viewer":
        interactive_viewer()
    elif page == "features":
        doctor_feature_report()
    elif page == "habitat":
        tumor_habitat_analysis()
    elif page == "ml":
        predict_tumor()            # UPDATED — HGG/LGG grade prediction
    elif page == "dashboard":
        clinical_dashboard()
    elif page == "followup":
        patient_followup()   # patient_id auto-filled from session
    elif page == "compare":
        timestamp_comparison()


if __name__ == "__main__":
    main()
