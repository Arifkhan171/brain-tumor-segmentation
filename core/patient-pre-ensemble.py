# core/patient.py
# ─────────────────────────────────────────────────────────────────────────────
# Patient Management System
#
# Handles patient registration, session context, and the patient info
# header shown on every page. Stores patient demographics in session state
# and persists to SQLite alongside scan results.
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
from datetime import date


# ─────────────────────────────────────────────────────────────────────────────
# Keys used in st.session_state
# ─────────────────────────────────────────────────────────────────────────────
#   patient_id       : str   — MRN or CNIC
#   patient_name     : str
#   patient_age      : int
#   patient_sex      : str   — "Male" | "Female" | "Other"
#   referring_doctor : str
#   scan_date        : date
#   patient_registered : bool


def is_registered() -> bool:
    return st.session_state.get("patient_registered", False)


def get_info() -> dict:
    return {
        "id":              st.session_state.get("patient_id",       "—"),
        "name":            st.session_state.get("patient_name",     "—"),
        "age":             st.session_state.get("patient_age",      "—"),
        "sex":             st.session_state.get("patient_sex",      "—"),
        "referring_doctor":st.session_state.get("referring_doctor", "—"),
        "scan_date":       st.session_state.get("scan_date",        date.today()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Registration form — shown before the upload page
# ─────────────────────────────────────────────────────────────────────────────

def render_registration():
    """
    Full-page patient registration form.
    On submit, saves all fields to session_state and marks patient_registered.
    """
    st.subheader("Patient Registration")
    st.caption(
        "Enter patient details before uploading MRI scans. "
        "This information will appear on all reports."
    )
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        pid = st.text_input(
            "Patient ID (MRN / CNIC) *",
            placeholder="e.g. 42301-1234567-8",
            help="Required. Used to link follow-up visits.",
        )
        name = st.text_input(
            "Patient Name *",
            placeholder="Full name",
        )
        age = st.number_input(
            "Age (years) *",
            min_value=1, max_value=120, value=45, step=1,
        )

    with col2:
        sex = st.selectbox(
            "Sex",
            ["Male", "Female", "Other"],
        )
        referring = st.text_input(
            "Referring Physician",
            placeholder="Dr. Name / Department",
        )
        scan_date = st.date_input(
            "Scan Date",
            value=date.today(),
        )

    st.divider()

    if st.button("Confirm and Continue →", type="primary"):
        if not pid.strip():
            st.error("Patient ID is required.")
            return
        if not name.strip():
            st.error("Patient name is required.")
            return

        st.session_state["patient_id"]         = pid.strip()
        st.session_state["patient_name"]        = name.strip()
        st.session_state["patient_age"]         = int(age)
        st.session_state["patient_sex"]         = sex
        st.session_state["referring_doctor"]    = referring.strip()
        st.session_state["scan_date"]           = scan_date
        st.session_state["patient_registered"]  = True
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Patient context bar — shown at top of every analysis page
# ─────────────────────────────────────────────────────────────────────────────

def render_patient_bar():
    """
    Compact one-line patient info bar shown at the top of every page.
    Only renders if a patient has been registered.
    """
    if not is_registered():
        return

    info  = get_info()
    grade = ""
    pred  = st.session_state.get("ml_prediction", {})
    if pred:
        cls   = pred.get("class", "")
        conf  = pred.get("confidence", 0)
        color = "#E24B4A" if cls == "HGG" else "#378ADD"
        grade = (
            f"<span style='color:{color};font-weight:600;margin-left:16px;'>"
            f"{cls} {conf:.0%}</span>"
        )

    st.markdown(
        f"""<div style='
            background:#161B22;
            border:1px solid #21262D;
            border-radius:6px;
            padding:8px 16px;
            margin-bottom:16px;
            font-size:13px;
            color:#8B95A2;
            display:flex;
            align-items:center;
            gap:0;
        '>
            <span style='color:#E6EDF3;font-weight:500'>{info["name"]}</span>
            <span style='margin:0 10px;'>·</span>
            <span>ID: {info["id"]}</span>
            <span style='margin:0 10px;'>·</span>
            <span>{info["age"]} y/o {info["sex"]}</span>
            <span style='margin:0 10px;'>·</span>
            <span>Scan: {info["scan_date"]}</span>
            {"<span style='margin:0 10px;'>·</span>" if info["referring_doctor"] else ""}
            {"<span>Ref: " + info["referring_doctor"] + "</span>" if info["referring_doctor"] else ""}
            {grade}
        </div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar patient summary — shown in sidebar after registration
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar_patient():
    """
    Compact patient summary in sidebar above navigation buttons.
    """
    if not is_registered():
        return

    info = get_info()
    st.sidebar.markdown(
        f"""<div style='
            padding:8px 4px;
            font-size:12px;
            color:#8B95A2;
            border-bottom:1px solid #21262D;
            margin-bottom:8px;
        '>
            <div style='color:#E6EDF3;font-weight:500;font-size:13px;margin-bottom:2px;'>
                {info["name"]}
            </div>
            <div>{info["id"]} · {info["age"]} y/o · {info["sex"]}</div>
            <div>Scan: {info["scan_date"]}</div>
        </div>""",
        unsafe_allow_html=True,
    )
