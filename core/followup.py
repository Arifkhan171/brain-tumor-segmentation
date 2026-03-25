from core.patient import render_patient_bar
import streamlit as st
import re
from pathlib import Path
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import os
import shutil
from datetime import datetime
from utils.plotting import plot_followup_trends

# ─────────────────────────────────────────────────────────────────────────────
# RANO criteria classifier
# ─────────────────────────────────────────────────────────────────────────────

def _classify_rano(current: dict, previous: dict) -> tuple[str, str, str]:
    """
    Classify tumour response per RANO criteria using ET volume change.
    Returns (status, colour_hex, description).
    """
    prev_et = previous.get("et", 0)
    curr_et = current.get("et", 0)

    if prev_et <= 0:
        return "Indeterminate", "#888780", "No prior enhancing tumour to compare."

    et_change_pct = (curr_et - prev_et) / prev_et * 100

    if curr_et < 0.1:
        return (
            "Complete Response (CR)", "#1D9E75",
            "No measurable enhancing tumour detected.",
        )
    elif et_change_pct <= -50:
        return (
            "Partial Response (PR)", "#378ADD",
            f"ET reduced by {abs(et_change_pct):.1f}% from baseline.",
        )
    elif et_change_pct >= 25:
        return (
            "Progressive Disease (PD)", "#E24B4A",
            f"ET increased by {et_change_pct:.1f}% — progression confirmed.",
        )
    else:
        return (
            "Stable Disease (SD)", "#EF9F27",
            f"ET change {et_change_pct:+.1f}% — within stable range.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Progression risk score
# ─────────────────────────────────────────────────────────────────────────────

def _progression_risk_score(df: pd.DataFrame) -> float:
    """
    Simple 0–100 risk score based on:
      - Rate of WT growth (slope)
      - Current enhancement ratio
      - ET change direction
    """
    if len(df) < 2:
        return 0.0

    wt_slope    = (df["wt"].iloc[-1] - df["wt"].iloc[0]) / max(len(df) - 1, 1)
    growth_score = min(max(wt_slope / 2.0 * 50, 0), 50)

    er       = df["enhancement_ratio"].iloc[-1] if "enhancement_ratio" in df.columns else 0
    er_score = min(er * 100, 30)

    et_delta = df["et"].iloc[-1] - df["et"].iloc[-2]
    et_score = 20 if et_delta > 0 else 0

    return round(min(growth_score + er_score + et_score, 100), 1)


import pathlib

DB_PATH = pathlib.Path(__file__).resolve().parent.parent / "radiology_followup.db"

# ─────────────────────────────────────────────────────────────────────────────
# DB connection helper
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patient_visits (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id        TEXT,
            visit_date        TEXT,
            wt                REAL,
            et                REAL,
            ed                REAL,
            core              REAL,
            axial_diameter    REAL,
            enhancement_ratio REAL,
            ml_grade          TEXT,
            mean_intensity    REAL
        )
    """)
    # Migration: add columns if they don't exist
    try:
        conn.execute("ALTER TABLE patient_visits ADD COLUMN ml_grade TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE patient_visits ADD COLUMN mean_intensity REAL")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE patient_visits ADD COLUMN archive_flair_path TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE patient_visits ADD COLUMN archive_seg_path TEXT")
    except sqlite3.OperationalError:
        pass
        
    conn.commit()
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def patient_followup():
    render_patient_bar()
    st.subheader(" Patient Follow-Up ")

    conn   = _get_connection()
    cursor = conn.cursor()

    # ── Patient identity ──────────────────────────────────────────────────────
    # Auto-fill from patient registration if available
    registered_id = st.session_state.get("patient_id", "")
    registered_date = st.session_state.get("scan_date", datetime.today())

    col_id, col_date = st.columns(2)
    with col_id:
        patient_id = st.text_input(
            "Patient ID (CNIC / Hospital MRN)",
            value=registered_id,
        )
    with col_date:
        visit_date = st.date_input("Visit Date", registered_date)

    if not patient_id:
        st.info("Enter a Patient ID to load or save visit data.")
        return

    # ── Save current visit ────────────────────────────────────────────────────
    has_metrics = "all_features" in st.session_state
    if has_metrics:
        if st.button("💾 Save This MRI Visit to Record"):
            # Validate patient ID — only allow safe characters
            if not re.match(r"^[\w\-]+$", patient_id):
                st.error("Patient ID must contain only letters, numbers, dashes, and underscores.")
                return

            # Check for duplicate visit
            existing = cursor.execute(
                "SELECT id FROM patient_visits WHERE patient_id=? AND visit_date=?",
                (patient_id, str(visit_date))
            ).fetchone()
            if existing:
                st.warning(
                    f"A visit for patient {patient_id} on {visit_date} already exists. "
                    "Use a different date or reset the session to record a new scan."
                )
                return

            feats = st.session_state["all_features"]
            ml_pred = st.session_state.get("ml_prediction", {})
            ml_grade = ml_pred.get("class", "Pending")

            # Archive NIfTI files — use absolute path so it works regardless of launch directory
            archive_dir = Path(__file__).resolve().parent.parent / "patient_archives" / patient_id / str(visit_date)
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            flair_src = st.session_state.get("flair_path")
            seg_src = st.session_state.get("seg_path")
            
            arch_flair = ""
            arch_seg = ""
            
            if flair_src and os.path.exists(flair_src):
                ext = ".nii.gz" if flair_src.endswith(".nii.gz") else ".nii"
                arch_flair = str(archive_dir / f"flair{ext}")
                shutil.copy2(flair_src, arch_flair)
            
            if seg_src and os.path.exists(seg_src):
                ext = ".nii.gz" if seg_src.endswith(".nii.gz") else ".nii"
                arch_seg = str(archive_dir / f"seg{ext}")
                shutil.copy2(seg_src, arch_seg)

            cursor.execute(
                """
                INSERT INTO patient_visits
                    (patient_id, visit_date, wt, et, ed, core,
                     axial_diameter, enhancement_ratio, ml_grade, mean_intensity,
                     archive_flair_path, archive_seg_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_id,
                    str(visit_date),
                    feats.get("wt_vol", 0),
                    feats.get("et_vol", 0),
                    feats.get("ed_vol", 0),
                    feats.get("core_vol", 0),
                    feats.get("axial_diameter", 0),
                    feats.get("enhancement_ratio", 0),
                    ml_grade,
                    feats.get("mean", 0),
                    arch_flair,
                    arch_seg
                ),
            )
            conn.commit()
            st.success("✅ Visit and 3D MRI data saved successfully to archive.")
    else:
        st.info("Run Feature Extraction first to enable saving this visit.")

    # ── Load visit history ────────────────────────────────────────────────────
    df_history = pd.read_sql_query(
        """
        SELECT visit_date, wt, et, ed, core, axial_diameter, enhancement_ratio, ml_grade, mean_intensity
        FROM patient_visits
        WHERE patient_id = ?
        ORDER BY visit_date ASC
        """,
        conn,
        params=(patient_id,),
    )

    if df_history.empty:
        st.warning("No visits found for this patient ID.")
        return

    # ── Visit history table ───────────────────────────────────────────────────
    with st.expander("  Full Visit History", expanded=False):
        st.dataframe(
            df_history.style.format({
                "wt": "{:.2f}", "et": "{:.2f}", "ed": "{:.2f}",
                "core": "{:.2f}", "axial_diameter": "{:.1f}",
                "enhancement_ratio": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "⬇ Download History CSV",
            df_history.to_csv(index=False).encode("utf-8"),
            f"Patient_{patient_id}_History.csv",
            "text/csv",
        )

    # ── Longitudinal trend chart ──────────────────────────────────────────────
    st.subheader("Tumour Volume Trends Over Time")
    st.plotly_chart(plot_followup_trends(df_history), use_container_width=True)

    if len(df_history) < 2:
        st.info("At least 2 visits required for progression analysis.")
        return

    prev    = df_history.iloc[-2].to_dict()
    current = df_history.iloc[-1].to_dict()

    # ── Metric change cards ───────────────────────────────────────────────────
    st.subheader("Progression Summary — Last Two Visits")
    wt_chg = current["wt"] - prev["wt"]
    et_chg = current["et"] - prev["et"]
    ed_chg = current["ed"] - prev["ed"]
    wt_pct = (wt_chg / prev["wt"]) * 100 if prev["wt"] > 0 else 0
    et_pct = (et_chg / prev["et"]) * 100 if prev["et"] > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("WT Change", f"{wt_chg:+.2f} cm³", f"{wt_pct:+.1f}%")
    c2.metric("ET Change", f"{et_chg:+.2f} cm³", f"{et_pct:+.1f}%")
    c3.metric("ED Change", f"{ed_chg:+.2f} cm³")
    c4.metric(
        "Axial Diam",
        f"{current['axial_diameter']:.1f} mm",
        f"{current['axial_diameter'] - prev['axial_diameter']:+.1f} mm",
    )

    # ── RANO classification ───────────────────────────────────────────────────
    st.write("")
    st.write("")
    st.subheader("RANO Response Classification")
    rano_status, rano_color, rano_desc = _classify_rano(current, prev)
    st.markdown(
        f"""
        <div style="
            background:{rano_color}18;
            border-left:4px solid {rano_color};
            border-radius:0 8px 8px 0;
            padding:12px 16px;
            margin-bottom:8px;
        ">
            <div style="font-weight:500;font-size:15px;color:{rano_color}">
                {rano_status}
            </div>
            <div style="font-size:13px;color:#555;margin-top:4px">
                {rano_desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # st.caption(
    #     "RANO = Response Assessment in Neuro-Oncology criteria. "
    #     "Based on enhancing tumour volume change. "
    #     "CR < 0.1 cm³ · PR ≤ −50% · SD between −50% and +25% · PD ≥ +25%."
    # )

    # ── Progression risk score ────────────────────────────────────────────────
    risk_score = _progression_risk_score(df_history)
    st.subheader("Progression Risk Score")

    col_gauge, col_text = st.columns([1, 2])
    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={"suffix": "/100", "font": {"size": 22}},
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color="#378ADD"),
                steps=[
                    dict(range=[0,  33], color="#E1F5EE"),
                    dict(range=[33, 66], color="#FAEEDA"),
                    dict(range=[66,100], color="#FCEBEB"),
                ],
                threshold=dict(line=dict(color="red", width=2), thickness=0.75, value=75),
            ),
        ))
        fig_gauge.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_text:
        if risk_score < 33:
            level, msg = "Low Risk", "Tumour appears stable. Routine follow-up recommended."
        elif risk_score < 66:
            level, msg = "Moderate Risk", "Some progression indicators present. Close monitoring advised."
        else:
            level, msg = "High Risk", "Multiple progression markers active. Urgent clinical review recommended."

        st.markdown(f"**Risk Level: {level}**")
        st.markdown(msg)
        # st.caption(
        #     "Score based on: whole-tumour growth rate, enhancement ratio, and ET trajectory."
        # )
