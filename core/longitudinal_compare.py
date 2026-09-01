from core.patient import render_patient_bar
import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import os
import numpy as np
from core.followup import _classify_rano
from utils.file_ops import load_nifti
from utils.plotting import create_rgb_overlay, plot_interactive_slice, plot_triplanar, plot_3d_tumor_mesh

import pathlib

DB_PATH = pathlib.Path(__file__).resolve().parent.parent / "radiology_followup.db"

def _get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def timestamp_comparison():
    render_patient_bar()
    # Inject CSS for cleaner visual badges
    st.markdown("""
        <style>
        .delta-badge {
            font-size: 20px; font-weight: bold; text-align: center;
            padding: 10px; border-radius: 8px; background-color: #1E1E1E; border: 1px solid #333;
        }
        .delta-pos { color: #FF5252; } /* Growth is red */
        .delta-neg { color: #4DA8DA; } /* Shrinkage is blue */
        .delta-neu { color: #888888; }
        .vs-title { text-align: center; font-size: 24px; color: white; margin-top: 10px;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("** Visual Timestamp Comparison**")
    
    conn = _get_connection()
    try:
        patients_df = pd.read_sql_query("SELECT DISTINCT patient_id FROM patient_visits", conn)
    except Exception:
        st.warning("Database error. Please save a visit first.")
        return
        
    if patients_df.empty:
        st.warning("No patient records found.")
        return
        
    selected_patient = st.selectbox("Select Patient ID", patients_df["patient_id"].tolist())
    
    # Query all columns including new archive paths
    df_history = pd.read_sql_query(
        """
        SELECT *
        FROM patient_visits
        WHERE patient_id = ?
        ORDER BY visit_date ASC
        """,
        conn,
        params=(selected_patient,)
    )
    
    if len(df_history) < 2:
        st.info("At least 2 visits required for comparison.")
        return
        
    dates = df_history["visit_date"].tolist()
    
    col_a, col_b = st.columns(2)
    with col_a: date_a = st.selectbox("Timestamp 1 (Baseline)", dates, index=0)
    with col_b: date_b = st.selectbox("Timestamp 2 (Follow-up)", dates, index=len(dates)-1)
        
    if date_a == date_b:
        st.warning("Select two different timestamps.")
        return
        
    if dates.index(date_a) > dates.index(date_b):
        date_a, date_b = date_b, date_a

    # Extract Data
    vA = df_history[df_history["visit_date"] == date_a].iloc[0].to_dict()
    vB = df_history[df_history["visit_date"] == date_b].iloc[0].to_dict()

    st.divider()

    # ── TOP SECTION: Overall Clinical Status ──────────────────────────────────
    rano_status, rano_color, _ = _classify_rano(vB, vA)
    grade_a = vA.get("ml_grade", "N/A")
    grade_b = vB.get("ml_grade", "N/A")
    grade_color = "#FF5252" if grade_b == "HGG" else "#4DA8DA"
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown(
            f"<div style='text-align:center; padding: 15px; border-radius: 8px; background: {rano_color}22; border: 1px solid {rano_color};'>"
            f"<span style='color:#AAA; font-size:14px; text-transform:uppercase;'>Response (RANO)</span><br>"
            f"<span style='color:{rano_color}; font-size:24px; font-weight:bold;'>{rano_status}</span></div>",
            unsafe_allow_html=True
        )
    with col_stat2:
        st.markdown(
            f"<div style='text-align:center; padding: 15px; border-radius: 8px; background: {grade_color}22; border: 1px solid {grade_color};'>"
            f"<span style='color:#AAA; font-size:14px; text-transform:uppercase;'>Grade Evolution</span><br>"
            f"<span style='color:{grade_color}; font-size:24px; font-weight:bold;'>{grade_a} ➔ {grade_b}</span></div>",
            unsafe_allow_html=True
        )

    st.write("")
    
    # Create Tabs for the different comparison views
    tab_metrics, tab_2d, tab_3p, tab_3d = st.tabs([
        "📊 Clinical Metrics",
        "🔲 2D Slice Viewer",
        "📐 3-Plane Viewer",
        "🌐 3D Tumor Mesh"
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: Clinical Metrics (Donuts & Radars)
    # ─────────────────────────────────────────────────────────────────────────
    with tab_metrics:
        # Helpers for Charts
        def _plot_donut(visit, title):
            labels = ["Enhancing (ET)", "Necrotic Core", "Edema (ED)"]
            core_vol = visit.get("core", 0)
            et_vol = visit.get("et", 0)
            ncr_vol = max(core_vol - et_vol, 0)
            values = [et_vol, ncr_vol, visit.get("ed", 0)]
            colors = ["#E24B4A", "#EF9F27", "#378ADD"]

            fig = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.6,
                marker=dict(colors=colors),
                textinfo="percent", textfont=dict(size=14, color="white"),
                hovertemplate="<b>%{label}</b><br>%{value:.1f} cm³<extra></extra>"
            ))
            fig.update_layout(
                title=dict(text=title, font=dict(size=16, color="white"), x=0.5),
                annotations=[dict(text=f"WT<br>{visit.get('wt',0):.1f}cm³", x=0.5, y=0.5, font_size=16, showarrow=False, font_color="white")],
                showlegend=False,
                height=250, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        def _plot_radar(visit, title, color_line, color_fill):
            labels = ["WT Vol", "ET Vol", "Edema", "Enhance Ratio", "Axial Diam", "Intensity"]
            keys = ["wt", "et", "ed", "enhancement_ratio", "axial_diameter"]
            ref = dict(wt=50.0, et=12.0, ed=30.0, enhancement_ratio=0.25, axial_diameter=50.0)
            
            intensity = visit.get("mean_intensity") or 1.0
            vals = [min((visit[k] / ref[k]) * 100, 200) for k in keys] + [min(intensity * 100, 200)]
            
            fig = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]], theta=labels + [labels[0]],
                fill="toself",
                line=dict(color=color_line, width=2), fillcolor=color_fill,
                hovertemplate="%{theta}<extra></extra>"
            ))
            fig.update_layout(
                title=dict(text=title, font=dict(size=16, color="white"), x=0.5),
                polar=dict(
                    radialaxis=dict(visible=False, range=[0, 200]),
                    angularaxis=dict(tickfont=dict(size=10, color="#AAA")),
                    bgcolor="#111"
                ),
                showlegend=False,
                height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        c_left, c_mid, c_right = st.columns([1.5, 1, 1.5])
        
        with c_left:
            st.plotly_chart(_plot_donut(vA, f"Composition ({date_a})"), use_container_width=True)
            st.plotly_chart(_plot_radar(vA, f"Signature ({date_a})", "#4DA8DA", "rgba(77,168,218,0.2)"), use_container_width=True)

        with c_right:
            st.plotly_chart(_plot_donut(vB, f"Composition ({date_b})"), use_container_width=True)
            st.plotly_chart(_plot_radar(vB, f"Signature ({date_b})", "#FF5252", "rgba(255,82,82,0.2)"), use_container_width=True)

        with c_mid:
            st.markdown("<div class='vs-title'>VS</div><br>", unsafe_allow_html=True)
            
            def _render_delta(label, val_a, val_b, unit=""):
                d = val_b - val_a
                p = (d / val_a * 100) if val_a > 0 else 0
                cls = "delta-neu"
                if d > 0.1: cls = "delta-pos"
                elif d < -0.1: cls = "delta-neg"
                
                icon = "▲" if d > 0 else "▼" if d < 0 else "➖"
                if d == 0: icon = ""
                
                st.markdown(f"""
                    <div class="delta-badge" style="margin-bottom: 15px;">
                        <div style="font-size:12px; color:#888;">{label}</div>
                        <div class="{cls}">{icon} {abs(d):.1f} {unit}</div>
                        <div style="font-size:12px; color:#555;">({p:+.1f}%)</div>
                    </div>
                """, unsafe_allow_html=True)

            _render_delta("WT Volume", vA.get("wt",0), vB.get("wt",0), "cm³")
            _render_delta("ET Volume", vA.get("et",0), vB.get("et",0), "cm³")
            _render_delta("Edema", vA.get("ed",0), vB.get("ed",0), "cm³")
            _render_delta("Max Axial", vA.get("axial_diameter",0), vB.get("axial_diameter",0), "mm")

    # ── Check MRI Availability for Visualizers ───────────────────────────────
    def _get_volumes(visit):
        flair_p = visit.get("archive_flair_path")
        seg_p   = visit.get("archive_seg_path")
        if flair_p and seg_p and os.path.exists(str(flair_p)) and os.path.exists(str(seg_p)):
            return load_nifti(flair_p), load_nifti(seg_p)
        session_flair = st.session_state.get("flair_path")
        session_seg   = st.session_state.get("seg_path")
        all_dates     = df_history["visit_date"].tolist()
        is_latest     = visit.get("visit_date") == all_dates[-1]
        if is_latest and session_flair and session_seg:
            if os.path.exists(session_flair) and os.path.exists(session_seg):
                return load_nifti(session_flair), load_nifti(session_seg)
        return None, None

    flairA, maskA = _get_volumes(vA)
    flairB, maskB = _get_volumes(vB)

    def get_best_slices(mask):
        if mask is None: return 0, 0, 0
        tumor_counts_z = (mask > 0).sum(axis=(0, 1))
        best_z = int(np.argmax(tumor_counts_z)) if tumor_counts_z.max() > 0 else mask.shape[2] // 2
        tumor_counts_y = (mask > 0).sum(axis=(0, 2))
        best_y = int(np.argmax(tumor_counts_y)) if tumor_counts_y.max() > 0 else mask.shape[1] // 2
        tumor_counts_x = (mask > 0).sum(axis=(1, 2))
        best_x = int(np.argmax(tumor_counts_x)) if tumor_counts_x.max() > 0 else mask.shape[0] // 2
        return best_x, best_y, best_z

    best_xA, best_yA, best_zA = get_best_slices(maskA)
    best_xB, best_yB, best_zB = get_best_slices(maskB)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2: 2D Slice Viewer
    # ─────────────────────────────────────────────────────────────────────────
    with tab_2d:
        if flairA is None or flairB is None:
            st.warning("⚠️ MRI scan files not found in archive for one or both selected dates. This happens when visits were saved before the archive feature was added. To fix this: save a new visit on the Follow-Up page with your current MRI loaded — that visit will have full visual comparison available.")
        else:
            st.markdown("### Side-by-Side Slice Viewer")
            st.caption("Scroll through the Z-axis to find equivalent slices.")
            
            c1, c2 = st.columns(2)
            with c1:
                z_A = st.slider(f"Z-Axis ({date_a})", 0, flairA.shape[2]-1, best_zA, key="z_A")
                rgbA = create_rgb_overlay(flairA[:,:,z_A], maskA[:,:,z_A])
                st.plotly_chart(plot_interactive_slice(rgbA, z_A, flairA.shape[2], vA.get("wt",0), vA.get("et",0), vA.get("ed",0)), use_container_width=True)
            
            with c2:
                z_B = st.slider(f"Z-Axis ({date_b})", 0, flairB.shape[2]-1, best_zB, key="z_B")
                rgbB = create_rgb_overlay(flairB[:,:,z_B], maskB[:,:,z_B])
                st.plotly_chart(plot_interactive_slice(rgbB, z_B, flairB.shape[2], vB.get("wt",0), vB.get("et",0), vB.get("ed",0)), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3: 3-Plane Viewer
    # ─────────────────────────────────────────────────────────────────────────
    with tab_3p:
        if flairA is None or flairB is None:
            st.warning("⚠️ MRI scan files not found in archive for one or both selected dates. This happens when visits were saved before the archive feature was added. To fix this: save a new visit on the Follow-Up page with your current MRI loaded — that visit will have full visual comparison available.")
        else:
            st.markdown("### 3-Plane Viewer Comparison")
            
            st.markdown(f"#### {date_a} (Baseline)")
            x_A = st.slider(f"X-Axis ({date_a})", 0, flairA.shape[0]-1, best_xA, key="3p_xA")
            st.plotly_chart(plot_triplanar(flairA, maskA, x_A, best_yA, best_zA), use_container_width=True)
            
            st.divider()
            
            st.markdown(f"#### {date_b} (Follow-up)")
            x_B = st.slider(f"X-Axis ({date_b})", 0, flairB.shape[0]-1, best_xB, key="3p_xB")
            st.plotly_chart(plot_triplanar(flairB, maskB, x_B, best_yB, best_zB), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4: 3D Tumor Mesh
    # ─────────────────────────────────────────────────────────────────────────
    with tab_3d:
        if maskA is None or maskB is None:
            st.warning("⚠️ MRI scan files not found in archive for one or both selected dates. This happens when visits were saved before the archive feature was added. To fix this: save a new visit on the Follow-Up page with your current MRI loaded — that visit will have full visual comparison available.")
        else:
            st.markdown("### Side-by-Side 3D Tumor Morphology")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"#### {date_a}")
                with st.spinner(f"Rendering 3D mesh for {date_a}..."):
                    fig3dA = plot_3d_tumor_mesh(maskA, wt_opacity=0.2, tc_opacity=0.5, et_opacity=0.9)
                    if fig3dA: st.plotly_chart(fig3dA, use_container_width=True)
                    
            with c2:
                st.markdown(f"#### {date_b}")
                with st.spinner(f"Rendering 3D mesh for {date_b}..."):
                    fig3dB = plot_3d_tumor_mesh(maskB, wt_opacity=0.2, tc_opacity=0.5, et_opacity=0.9)
                    if fig3dB: st.plotly_chart(fig3dB, use_container_width=True)
