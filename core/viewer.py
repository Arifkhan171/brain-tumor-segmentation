from core.patient import render_patient_bar
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from utils.file_ops import load_nifti, get_best_slice, get_voxel_volume_mm3
from utils.plotting import (
    create_rgb_overlay,
    plot_interactive_slice,
    plot_triplanar,
    plot_3d_tumor_mesh,
    render_tumor_legend,
)


def interactive_viewer():
    render_patient_bar()
    st.subheader("Step 2 — MRI Viewer")

    if "flair_path" not in st.session_state or "seg_path" not in st.session_state:
        st.info("Complete Steps 1a and 1b before opening the viewer.")
        return

    flair = load_nifti(st.session_state["flair_path"])
    mask  = load_nifti(st.session_state["seg_path"])

    if flair is None or mask is None:
        st.error("Could not load MRI or mask. Please re-upload in Step 1a.")
        return

    voxel_vol = get_voxel_volume_mm3(st.session_state["flair_path"])

    wt_cm3 = float((mask > 0).sum())                      * voxel_vol / 1000
    et_cm3 = float((mask == 3).sum())                     * voxel_vol / 1000
    ed_cm3 = float((mask == 2).sum())                     * voxel_vol / 1000

    best_z = get_best_slice(mask, axis=(0, 1))
    best_y = get_best_slice(mask, axis=(0, 2))
    best_x = get_best_slice(mask, axis=(1, 2))

    tab1, tab2, tab3 = st.tabs(["2D Slice Viewer", "3-Plane Viewer", "3D Tumor Mesh"])

    with tab1:
        slice_index = st.slider(
            "Axial slice (Z)", 0, flair.shape[2] - 1, best_z, key="z_slider_2d"
        )
        rgb = create_rgb_overlay(flair[:, :, slice_index], mask[:, :, slice_index])
        fig = plot_interactive_slice(rgb, slice_index, flair.shape[2], wt_cm3, et_cm3, ed_cm3)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"scrollZoom": True, "displayModeBar": True},
        )
        render_tumor_legend()

    with tab2:
        st.caption("Axial · Coronal · Sagittal — move sliders to reposition the crosshair.")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            z_idx = st.slider("Axial (Z)",    0, flair.shape[2] - 1, best_z, key="z_tri")
        with col_s2:
            y_idx = st.slider("Coronal (Y)",  0, flair.shape[1] - 1, best_y, key="y_tri")
        with col_s3:
            x_idx = st.slider("Sagittal (X)", 0, flair.shape[0] - 1, best_x, key="x_tri")
        st.plotly_chart(
            plot_triplanar(flair, mask, x_idx, y_idx, z_idx),
            use_container_width=True,
            )

    with tab3:
        if (mask > 0).sum() < 100:
            st.warning("Insufficient tumor voxels for 3D reconstruction.")
        else:
            st.caption("Drag to rotate · scroll to zoom · double-click to reset.")
            op1, op2, op3 = st.columns(3)
            wt_opc = op1.slider("Edema opacity",     0.0, 1.0, 0.20, 0.05, key="wt_opc")
            tc_opc = op2.slider("Core opacity",      0.0, 1.0, 0.55, 0.05, key="tc_opc")
            et_opc = op3.slider("Enhancing opacity", 0.0, 1.0, 0.90, 0.05, key="et_opc")

            @st.cache_resource(show_spinner=False)
            def _build_mesh(mask_arr):
                return plot_3d_tumor_mesh(
                    mask_arr, wt_opacity=1.0, tc_opacity=1.0, et_opacity=1.0
                )

            with st.spinner("Building 3D mesh..."):
                fig_3d = _build_mesh(mask)

            if fig_3d is None:
                st.error("scikit-image required. Run: pip install scikit-image")
            else:
                fig_out = go.Figure(fig_3d)
                for trace in fig_out.data:
                    if trace.name == "Whole Tumor":
                        trace.opacity = wt_opc
                    elif trace.name == "Tumor Core":
                        trace.opacity = tc_opc
                    elif trace.name == "Enhancing Tumor":
                        trace.opacity = et_opc
                st.plotly_chart(fig_out, use_container_width=True)
                render_tumor_legend()
