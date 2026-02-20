import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────────────────
# Basic helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize(arr):
    """Min-max normalize array to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


def plot_middle_slice(data, title="Slice Preview"):
    mid      = data.shape[2] // 2
    slice_img = data[:, :, mid]
    fig, ax  = plt.subplots(figsize=(2, 2))
    ax.imshow(slice_img, cmap="gray")
    ax.set_title(title[:12], fontsize=7)
    ax.axis("off")
    return fig


def create_rgb_overlay(flair_slice, mask_slice):
    """Create colour-coded RGB overlay on a FLAIR slice."""
    flair_norm = normalize(flair_slice)
    rgb        = np.stack([flair_norm] * 3, axis=-1)
    rgb[mask_slice == 2] = [0.10, 0.45, 1.00]   # Edema  → blue
    rgb[mask_slice == 3] = [1.00, 0.20, 0.20]   # ET     → red   (label 3 after DynUNet remap)
    rgb[mask_slice == 1] = [1.00, 0.88, 0.10]   # NCR    → yellow
    return (rgb * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Single-plane interactive slice
# ─────────────────────────────────────────────────────────────────────────────

def plot_interactive_slice(rgb, slice_index, total_slices, WT, ET, ED):
    fig = px.imshow(
        rgb,
        title=(
            f"Slice {slice_index + 1}/{total_slices}  |  "
            f"WT={WT:.2f} cm³  ET={ET:.2f} cm³  ED={ED:.2f} cm³"
        ),
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), coloraxis_showscale=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Triplanar synchronized viewer
# ─────────────────────────────────────────────────────────────────────────────

def _overlay_slice(volume, mask, axis, index):
    """Return an RGB uint8 image for a given plane/index."""
    if axis == 0:       # Sagittal
        v_sl = volume[index, :, :]
        m_sl = mask[index, :, :]
    elif axis == 1:     # Coronal
        v_sl = volume[:, index, :]
        m_sl = mask[:, index, :]
    else:               # Axial
        v_sl = volume[:, :, index]
        m_sl = mask[:, :, index]
    return create_rgb_overlay(v_sl, m_sl)


def plot_triplanar(flair, mask, x, y, z):
    """Three-plane MRI viewer: Axial | Coronal | Sagittal with crosshair lines."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Axial (Z)", "Coronal (Y)", "Sagittal (X)"],
        horizontal_spacing=0.002,
    )

    planes = [
        (_overlay_slice(flair, mask, 2, z), z, flair.shape[2], y, x),  # Axial
        (_overlay_slice(flair, mask, 1, y), y, flair.shape[1], x, z),  # Coronal
        (_overlay_slice(flair, mask, 0, x), x, flair.shape[0], y, z),  # Sagittal
    ]

    for col, (rgb, idx, total, ch, cv) in enumerate(planes, start=1):
        fig.add_trace(go.Image(z=rgb), row=1, col=col)
        H, W = rgb.shape[:2]
        fig.add_shape(type="line", x0=0, x1=W, y0=ch, y1=ch,
                      line=dict(color="cyan", width=1, dash="dot"), row=1, col=col)
        fig.add_shape(type="line", x0=cv, x1=cv, y0=0, y1=H,
                      line=dict(color="cyan", width=1, dash="dot"), row=1, col=col)

    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=25, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, scaleanchor=None)
    fig.update_yaxes(showticklabels=False, showgrid=False, scaleanchor=None)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3D Tumor Mesh
# ─────────────────────────────────────────────────────────────────────────────

def plot_3d_tumor_mesh(mask, wt_opacity=0.20, tc_opacity=0.55, et_opacity=0.90):
    """Render WT / Core / Enhancing as stacked 3D Plotly meshes using marching cubes with dynamic opacity."""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        return None

    regions = {
        "Whole Tumor":     (mask > 0,                          "rgba(100,180,255,1.0)", wt_opacity),
        "Tumor Core":      ((mask == 1) | (mask == 3),         "rgba(255,220,60,1.0)",  tc_opacity),  # NCR + ET
        "Enhancing Tumor": (mask == 3,                         "rgba(255,70,70,1.0)",   et_opacity),  # label 3
    }

    fig = go.Figure()
    for name, (vol, color, opacity) in regions.items():
        if vol.sum() < 20:
            continue
        try:
            verts, faces, _, _ = marching_cubes(vol.astype(float), level=0.5, step_size=2)
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=color, opacity=opacity,
                name=name, showlegend=True, flatshading=False,
            ))
        except Exception:
            continue

    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            bgcolor="rgba(15,15,20,1)",
            aspectmode="data",
        ),
        legend=dict(x=0.01, y=0.99, font=dict(size=11)),
        margin=dict(l=0, r=0, t=10, b=0),
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Radar / spider chart for feature profile
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_radar(wt_vol, et_vol, ed_vol, enhancement_ratio,
                       max_diameter, mean_intensity):
    """
    Normalised radar chart comparing extracted features against
    approximate BraTS population reference ranges.
    """
    ref = dict(wt=50.0, et=12.0, ed=30.0, ratio=0.25, diam=50.0, intensity=1.0)
    raw = dict(wt=wt_vol, et=et_vol, ed=ed_vol,
               ratio=enhancement_ratio, diam=max_diameter, intensity=mean_intensity)

    labels = ["WT Volume", "ET Volume", "Edema Volume",
              "Enhancement Ratio", "Max Diameter", "Mean Intensity"]
    keys   = ["wt", "et", "ed", "ratio", "diam", "intensity"]

    patient_vals = [min((raw[k] / ref[k]) * 100, 200) for k in keys]
    ref_vals     = [100.0] * len(keys)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=ref_vals + [ref_vals[0]], theta=labels + [labels[0]],
        fill="toself", name="Population median",
        line=dict(color="rgba(100,180,255,0.7)", width=1.5),
        fillcolor="rgba(100,180,255,0.08)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=patient_vals + [patient_vals[0]], theta=labels + [labels[0]],
        fill="toself", name="This patient",
        line=dict(color="rgba(255,100,80,0.9)", width=2),
        fillcolor="rgba(255,100,80,0.15)",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 200], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.12),
        margin=dict(l=40, r=40, t=20, b=40),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Follow-up longitudinal trend chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_followup_trends(df_history):
    """
    Multi-line chart of WT / ET / ED / Core volumes over visit dates.
    df_history must have columns: visit_date, wt, et, ed, core.
    """
    fig    = go.Figure()
    traces = [
        ("wt",   "Whole Tumor", "#378ADD"),
        ("et",   "Enhancing",   "#E24B4A"),
        ("ed",   "Edema",       "#5DCAA5"),
        ("core", "Core",        "#EF9F27"),
    ]
    for col, label, color in traces:
        if col not in df_history.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df_history["visit_date"], y=df_history[col],
            mode="lines+markers", name=label,
            line=dict(color=color, width=2), marker=dict(size=7),
        ))

    fig.update_layout(
        xaxis_title="Visit Date", yaxis_title="Volume (cm³)",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=0, r=0, t=10, b=60),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Tumor habitat overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_habitat_slice(flair_slice, habitat_slice, n_clusters=4):
    """Overlay habitat cluster labels on a FLAIR slice."""
    palette = [
        [0,   0,   0  ],   # background
        [255, 70,  70 ],   # habitat 1 — hypoxic core
        [255, 200, 60 ],   # habitat 2 — proliferating rim
        [80,  200, 120],   # habitat 3 — infiltrating edge
        [100, 160, 255],   # habitat 4 — peritumoral
    ]

    flair_norm = normalize(flair_slice)
    rgb        = np.stack([flair_norm] * 3, axis=-1)

    for i in range(1, n_clusters + 1):
        mask_i = habitat_slice == i
        if mask_i.any() and i < len(palette):
            c     = np.array(palette[i]) / 255.0
            alpha = 0.55
            rgb[mask_i] = (1 - alpha) * rgb[mask_i] + alpha * c

    fig = px.imshow((rgb * 255).astype(np.uint8), title="Tumor Habitat Map")
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=35, b=0),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITY — Tumor color legend
# Replaces duplicated legend code in uploader.py, auto_segment.py, viewer.py
# ─────────────────────────────────────────────────────────────────────────────

def render_tumor_legend():
    """
    Render the standard blue/red/yellow tumor region color legend.
    Call this once after any brain image display instead of duplicating
    the three markdown lines across every file.
    """
    import streamlit as st
    c1, c2, c3 = st.columns(3)
    c1.markdown("🔵 **Edema** (label 2)")
    c2.markdown("🔴 **Enhancing Tumor** (label 3)")
    c3.markdown("🟡 **Necrotic Core** (label 1)")
