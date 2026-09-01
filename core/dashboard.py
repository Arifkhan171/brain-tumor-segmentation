from core.patient import render_patient_bar
import streamlit as st
import plotly.graph_objects as go
from utils.plotting import plot_feature_radar

def clinical_dashboard():
    render_patient_bar()
    # Inject some custom CSS for a premium feel
    st.markdown("""
        <style>
        .metric-card {
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            text-align: center;
            border: 1px solid #333;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #4DA8DA;
        }
        .metric-label {
            font-size: 14px;
            color: #AAAAAA;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.subheader("Clinical Dashboard")

    # Check dependencies
    has_features = "all_features" in st.session_state
    has_ml       = "ml_prediction" in st.session_state

    if not has_features:
        st.warning("⚠️ **Dashboard Incomplete:** Please complete 'Auto Segmentation' and 'Feature Report' to populate this dashboard.")
        return

    # Extract Data
    feats = st.session_state["all_features"]
    wt = feats.get("wt_vol", 0)
    et = feats.get("et_vol", 0)
    ed = feats.get("ed_vol", 0)
    core = feats.get("core_vol", 0)
    ncr = max(core - et, 0)
    er = feats.get("enhancement_ratio", 0)
    ax = feats.get("axial_diameter", 0)
    max_d = feats.get("max_diameter_3d", 0)
    mean_int = feats.get("mean", 0)

    # ─────────────────────────────────────────────────────────────────────────
    # HERO SECTION: Primary AI Diagnosis
    # ─────────────────────────────────────────────────────────────────────────
    if has_ml:
        pred  = st.session_state["ml_prediction"]
        cls   = pred.get("class", "-")
        conf  = pred.get("confidence", 0)
        color = "#E24B4A" if cls == "HGG" else "#378ADD"
        label = "High Grade Glioma (HGG)" if cls == "HGG" else "Low Grade Glioma (LGG)"
        st.markdown(
            f'''<div style="border-left:4px solid {color};padding:12px 18px;
                background:{color}12;border-radius:0 6px 6px 0;margin-bottom:18px;">
                <span style="font-size:20px;font-weight:600;color:{color}">{label}</span>
                <span style="font-size:16px;color:#888;margin-left:18px;">
                    AI confidence {conf:.1%} 
                </span></div>''',
            unsafe_allow_html=True,
        )
    else:
        st.info("Run Grade Prediction to show classification result here.")

    # ─────────────────────────────────────────────────────────────────────────
    # INTERACTIVE PORTIONS (TABS)
    # ─────────────────────────────────────────────────────────────────────────
    tab1, tab2= st.tabs([
        " Clinical Risk Indicators", 
        " Tumor Composition & Volumes", 
        
    ])

    # --- TAB 1: Clinical Risk Indicators (Gauges) ---
    with tab1:
        
        
        g1, g2, g3 = st.columns(3)

        with g1:
            fig_er = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = er,
                title = {'text': "Enhancement Ratio", 'font': {'size': 16, 'color': 'white'}},
                number = {'font': {'color': 'white'}},
                gauge = {
                    'axis': {'range': [None, max(1.0, round(er + 0.1, 1))], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#4DA8DA"},
                    'bgcolor': "#1E1E1E",
                    'borderwidth': 2,
                    'bordercolor': "#333",
                    'steps': [
                        {'range': [0, 0.15], 'color': "#111C2C"},
                        {'range': [0.15, 0.3], 'color': "#3A2A18"},
                        {'range': [0.3, 1.0], 'color': "#3A1111"}],
                    'threshold': {'line': {'color': "#FF5252", 'width': 4}, 'thickness': 0.75, 'value': 0.3}
                }
            ))
            fig_er.update_layout(height=350, margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_er, use_container_width=True)
            if er > 0.3:
                st.error("High Enhancement: Suggests active BBB breakdown.")

        with g2:
            ed_ratio = (ed / wt) if wt > 0 else 0
            fig_ed = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = ed_ratio,
                title = {'text': "Edema / Tumor Ratio", 'font': {'size': 16, 'color': 'white'}},
                number = {'font': {'color': 'white'}},
                gauge = {
                    'axis': {'range': [None, max(5, ed_ratio + 1)], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#4DA8DA"},
                    'bgcolor': "#1E1E1E",
                    'borderwidth': 2,
                    'bordercolor': "#333",
                    'steps': [
                        {'range': [0, 1.5], 'color': "#111C2C"},
                        {'range': [1.5, 2.5], 'color': "#3A2A18"},
                        {'range': [2.5, max(5, ed_ratio + 1)], 'color': "#3A1111"}],
                    'threshold': {'line': {'color': "#FF5252", 'width': 4}, 'thickness': 0.75, 'value': 2.0}
                }
            ))
            fig_ed.update_layout(height=350, margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_ed, use_container_width=True)
            if ed_ratio > 2.0:
                st.warning("Extensive Edema: High risk of mass effect.")

        with g3:
            fig_sz = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = max_d,
                number = {'suffix': " mm", 'font': {'color': 'white'}},
                title = {'text': "Max 3D Diameter", 'font': {'size': 16, 'color': 'white'}},
                gauge = {
                    'axis': {'range': [None, 120], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#4DA8DA"},
                    'bgcolor': "#1E1E1E",
                    'borderwidth': 2,
                    'bordercolor': "#333",
                    'steps': [
                        {'range': [0, 30], 'color': "#111C2C"},
                        {'range': [30, 60], 'color': "#3A2A18"},
                        {'range': [60, 120], 'color': "#3A1111"}],
                    'threshold': {'line': {'color': "#FF5252", 'width': 4}, 'thickness': 0.75, 'value': 60}
                }
            ))
            fig_sz.update_layout(height=350, margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_sz, use_container_width=True)
            if max_d > 60:
                st.error("Large Mass: High risk of midline shift.")

    # --- TAB 2: Tumor Composition ---
    with tab2:

        
        c1, c2 = st.columns([1, 1.5])
        
        with c1:
            st.write("") # spacing
            st.write("")
            st.markdown(f"""
                <div class="metric-card" style="margin-bottom: 15px;">
                    <div class="metric-label">Whole Tumor Volume</div>
                    <div class="metric-value">{wt:.1f} cm³</div>
                </div>
                <div class="metric-card" style="margin-bottom: 15px; border-left: 4px solid #FF5252;">
                    <div class="metric-label">Enhancing Tumor (ET)</div>
                    <div class="metric-value" style="color: #FF5252;">{et:.1f} cm³</div>
                </div>
                <div class="metric-card" style="margin-bottom: 15px; border-left: 4px solid #4DA8DA;">
                    <div class="metric-label">Peritumoral Edema (ED)</div>
                    <div class="metric-value">{ed:.1f} cm³</div>
                </div>
                <div class="metric-card" style="border-left: 4px solid #EF9F27;">
                    <div class="metric-label">Necrotic Core (NCR)</div>
                    <div class="metric-value" style="color: #EF9F27;">{ncr:.1f} cm³</div>
                </div>
            """, unsafe_allow_html=True)

        with c2:
            if wt > 0:
                # Interactive Sunburst Chart for hierarchical view
                labels = ["Whole Tumor", "Edema", "Tumor Core", "Enhancing", "Necrotic"]
                parents = ["", "Whole Tumor", "Whole Tumor", "Tumor Core", "Tumor Core"]
                values = [wt, ed, core, et, ncr]
                colors = ["#1E1E1E", "#378ADD", "#2A2A2A", "#E24B4A", "#EF9F27"]

                fig_sunburst = go.Figure(go.Sunburst(
                    labels=labels,
                    parents=parents,
                    values=values,
                    branchvalues="total",
                    marker=dict(colors=colors, line=dict(color='#000000', width=1)),
                    hovertemplate='<b>%{label}</b><br>Volume: %{value:.1f} cm³<br>Proportion: %{percentParent:.1%}<extra></extra>',
                    textfont=dict(size=14, color="white")
                ))
                fig_sunburst.update_layout(
                    height=380, 
                    margin=dict(t=10, l=10, r=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_sunburst, use_container_width=True)
            else:
                st.info("No tumor volumes calculated.")

    