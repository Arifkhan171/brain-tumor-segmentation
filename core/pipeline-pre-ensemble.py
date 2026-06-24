# core/pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# Background Pipeline Manager
#
# Runs segmentation and feature extraction automatically in a background
# thread the moment MRI files are uploaded. The doctor sees the MRI previews
# while the AI processes. No manual clicking of Run Segmentation required.
# ─────────────────────────────────────────────────────────────────────────────

import threading
import time
import os
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline status keys stored in st.session_state
# ─────────────────────────────────────────────────────────────────────────────
#   pipeline_status : "idle" | "running" | "done" | "error"
#   pipeline_stage  : human-readable current stage string
#   pipeline_error  : error message if status == "error"


def get_status() -> str:
    return st.session_state.get("pipeline_status", "idle")


def get_stage() -> str:
    return st.session_state.get("pipeline_stage", "")


def is_running() -> bool:
    return get_status() == "running"


def is_done() -> bool:
    return get_status() == "done"


def _set(status: str, stage: str = ""):
    st.session_state["pipeline_status"] = status
    st.session_state["pipeline_stage"]  = stage


# ─────────────────────────────────────────────────────────────────────────────
# Background worker
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(flair_path, t1_path, t1ce_path, t2_path,
                  model_path, seg_out_path):
    """
    Runs in a background thread. Executes:
      1. DynUNet segmentation
      2. Feature extraction
      3. Grade prediction
    Updates session_state at each stage so the sidebar reflects progress.
    """
    try:
        # ── Stage 1: Segmentation ─────────────────────────────────────────────
        _set("running", "Running DynUNet segmentation...")

        from core.auto_segment import run_segmentation
        pred_np, seg_path = run_segmentation(
            flair_path, t1_path, t1ce_path, t2_path,
            model_path, seg_out_path,
        )
        st.session_state["seg_path"] = seg_path
        st.session_state["seg_auto"] = True

        # ── Stage 2: Feature Extraction ───────────────────────────────────────
        _set("running", "Extracting clinical features...")

        from core.feature_report import doctor_feature_report_headless
        doctor_feature_report_headless(flair_path, seg_path)

        # ── Stage 3: Grade Prediction ─────────────────────────────────────────
        _set("running", "Running grade prediction...")

        from core.predict import extract_grade_features, load_grade_model, FEATURE_COLS
        import numpy as np

        features  = extract_grade_features(seg_path)
        st.session_state["grade_features"] = features

        if os.path.exists(model_path.replace("best_overall.pt", "xgboost_grade_model.pkl")):
            grade_model_path = model_path.replace("best_overall.pt", "xgboost_grade_model.pkl")
        else:
            grade_model_path = st.session_state.get(
                "grade_model_path", "models/xgboost_grade_model.pkl"
            )

        if os.path.exists(grade_model_path):
            saved     = load_grade_model(grade_model_path)
            xgb_model = saved["model"]
            scaler    = saved["scaler"]
            feat_cols = saved.get("feature_cols", FEATURE_COLS)
            x_raw     = np.array([[features.get(c, 0) for c in feat_cols]])
            x_scaled  = scaler.transform(x_raw)
            prob      = xgb_model.predict_proba(x_scaled)[0]
            hgg_prob  = float(prob[1])
            lgg_prob  = float(prob[0])
            grade     = "HGG" if hgg_prob >= 0.5 else "LGG"
            confidence = max(hgg_prob, lgg_prob)

            st.session_state["ml_prediction"] = {
                "class":         grade,
                "confidence":    confidence,
                "probabilities": {"HGG": hgg_prob, "LGG": lgg_prob},
                "classes":       ["LGG", "HGG"],
            }

        _set("done", "Analysis complete.")

    except Exception as e:
        _set("error", f"Pipeline error: {e}")
        st.session_state["pipeline_error"] = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Public launcher — call this right after files are uploaded
# ─────────────────────────────────────────────────────────────────────────────

def launch_pipeline():
    """
    Start the background pipeline if files are uploaded and pipeline
    is not already running or done.
    Only launches if model file exists.
    """
    if get_status() in ("running", "done"):
        return

    flair_path = st.session_state.get("flair_path")
    if not flair_path:
        return

    model_path = st.session_state.get("model_path", "models/best_overall.pt")
    if not os.path.exists(model_path):
        return   # model not found — user must set path manually

    import tempfile
    temp_dir     = st.session_state.get("temp_dir", tempfile.mkdtemp())
    seg_out_path = os.path.join(temp_dir, "auto_seg.nii.gz")

    t1_path   = st.session_state.get("t1_path",   flair_path)
    t1ce_path = st.session_state.get("t1ce_path", flair_path)
    t2_path   = st.session_state.get("t2_path",   flair_path)

    _set("running", "Starting AI pipeline...")

    thread = threading.Thread(
        target=_run_pipeline,
        args=(flair_path, t1_path, t1ce_path, t2_path, model_path, seg_out_path),
        daemon=True,
    )
    thread.start()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar status widget — call from _sidebar() in app.py
# ─────────────────────────────────────────────────────────────────────────────

def render_pipeline_status():
    """
    Shows a compact pipeline status indicator in the sidebar.
    Replaces the manual pipeline checklist.
    """
    status = get_status()
    stage  = get_stage()

    if status == "idle":
        return   # nothing to show before upload

    st.sidebar.divider()

    if status == "running":
        st.sidebar.caption(f"⚙️ {stage}")

    elif status == "done":
        pred = st.session_state.get("ml_prediction", {})
        grade = pred.get("class", "")
        conf  = pred.get("confidence", 0)

        if grade == "HGG":
            st.sidebar.markdown(
                f"<div style='background:#E24B4A18;border-left:3px solid #E24B4A;"
                f"padding:8px 10px;border-radius:0 4px 4px 0;font-size:13px;'>"
                f"<b style='color:#E24B4A'>HGG</b> "
                f"<span style='color:#888'>{conf:.0%} confidence</span></div>",
                unsafe_allow_html=True,
            )
        elif grade == "LGG":
            st.sidebar.markdown(
                f"<div style='background:#378ADD18;border-left:3px solid #378ADD;"
                f"padding:8px 10px;border-radius:0 4px 4px 0;font-size:13px;'>"
                f"<b style='color:#378ADD'>LGG</b> "
                f"<span style='color:#888'>{conf:.0%} confidence</span></div>",
                unsafe_allow_html=True,
            )
        st.sidebar.caption("✅ Pipeline complete")

    elif status == "error":
        err = st.session_state.get("pipeline_error", "Unknown error")
        st.sidebar.caption(f"❌ {err[:60]}")
