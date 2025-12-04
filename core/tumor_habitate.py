from core.patient import render_patient_bar
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.file_ops import load_nifti
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.plotting import normalize, plot_habitat_slice

# ─────────────────────────────────────────────────────────────────────────────
# Habitat labels
# ─────────────────────────────────────────────────────────────────────────────
HABITAT_LABELS = {
    1: ("Hypoxic Core",      "#E24B4A"),
    2: ("Proliferating Rim", "#EF9F27"),
    3: ("Infiltrating Edge", "#378ADD"),
    4: ("Peritumoral Zone",  "#1D9E75"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Habitat computation (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_habitats(
    t1_path: str,
    t1ce_path: str,
    t2_path: str,
    flair_path: str,
    seg_path: str,
    n_clusters: int = 4,
) -> "np.ndarray":
   
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    mask    = load_nifti(seg_path)
    wt_mask = mask > 0

    if wt_mask.sum() < 20:
        return np.zeros_like(mask, dtype=np.int32)

    # Load and normalise all available modalities
    modalities = {}
    for name, path in [
        ("t1", t1_path), ("t1ce", t1ce_path),
        ("t2", t2_path),  ("flair", flair_path),
    ]:
        if path:
            modalities[name] = normalize(load_nifti(path))

    if len(modalities) < 2:
        st.warning("At least 2 modalities are needed for habitat analysis.")
        return np.zeros_like(mask, dtype=np.int32)

    feature_matrix = np.stack([vol[wt_mask] for vol in modalities.values()], axis=1)
    scaler         = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(feature_matrix) + 1   # shift to 1-based

    habitat_map           = np.zeros(mask.shape, dtype=np.int32)
    habitat_map[wt_mask]  = labels
    return habitat_map


# ─────────────────────────────────────────────────────────────────────────────
# Habitat statistics
# ─────────────────────────────────────────────────────────────────────────────

def _habitat_statistics(habitat_map, mri, n_clusters=4):
    """Return a list of per-habitat intensity statistics."""
    rows = []
    for h in range(1, n_clusters + 1):
        vox         = mri[habitat_map == h]
        label, color = HABITAT_LABELS.get(h, (f"Habitat {h}", "#888780"))
        rows.append({
            "Habitat":    label,
            "Voxels":     int((habitat_map == h).sum()),
            "Volume cm³": round(float((habitat_map == h).sum()) / 1000, 2),
            "Mean Int":   round(float(vox.mean()), 2) if vox.size else 0,
            "Std Int":    round(float(vox.std()),  2) if vox.size else 0,
            "_color":     color,
        })
    return rows


def _plot_habitat_volumes(stats_rows):
    """Horizontal bar chart of habitat volumes."""
    labels  = [r["Habitat"]    for r in stats_rows]
    volumes = [r["Volume cm³"] for r in stats_rows]
    colors  = [r["_color"]     for r in stats_rows]

    fig = go.Figure(go.Bar(
        x=volumes, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f} cm³" for v in volumes],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Volume (cm³)",
        margin=dict(l=0, r=60, t=10, b=30),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(showgrid=False),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def tumor_habitat_analysis():
    render_patient_bar()
    st.subheader("**Tumour Habitat Analysis**")
    

    if "flair_path" not in st.session_state or "seg_path" not in st.session_state:
        st.info("Complete MRI upload first.")
        return

    t1_path    = st.session_state.get("t1_path")
    t1ce_path  = st.session_state.get("t1ce_path")
    t2_path    = st.session_state.get("t2_path")
    flair_path = st.session_state["flair_path"]
    seg_path   = st.session_state["seg_path"]

    n_clusters = st.slider("Number of habitat clusters", min_value=2, max_value=6, value=4)

    if st.button("🔬 Run Habitat Analysis"):
        with st.spinner("Clustering tumour voxels across modalities..."):
            habitat_map = compute_habitats(
                t1_path, t1ce_path, t2_path, flair_path, seg_path, n_clusters
            )
        st.session_state["habitat_map"] = habitat_map
        st.success("Habitat map computed.")

    if "habitat_map" not in st.session_state:
        return

    habitat_map = st.session_state["habitat_map"]
    flair       = load_nifti(flair_path)

    best_z = int(np.argmax((habitat_map > 0).sum(axis=(0, 1))))

    tab1, tab2 = st.tabs(["🗺️ Habitat Overlay", "📊 Habitat Statistics"])

    # ── Tab 1: Overlay viewer ─────────────────────────────────────────────────
    with tab1:
        z_idx = st.slider(
            "Axial slice", 0, habitat_map.shape[2] - 1, best_z,
            key="habitat_z",
        )
        fig_hab = plot_habitat_slice(flair[:, :, z_idx], habitat_map[:, :, z_idx], n_clusters)
        st.plotly_chart(fig_hab, use_container_width=True)

        cols = st.columns(len(HABITAT_LABELS))
        for col, (idx, (name, color)) in zip(cols, HABITAT_LABELS.items()):
            if idx <= n_clusters:
                col.markdown(
                    f"<span style='color:{color};font-weight:500'>■</span> "
                    f"<small>{name}</small>",
                    unsafe_allow_html=True,
                )

    # ── Tab 2: Statistics table & chart ──────────────────────────────────────
    with tab2:
        stats_rows = _habitat_statistics(habitat_map, flair, n_clusters)
        df_stats   = pd.DataFrame([
            {k: v for k, v in r.items() if k != "_color"}
            for r in stats_rows
        ])
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
        st.plotly_chart(_plot_habitat_volumes(stats_rows), use_container_width=True)
        