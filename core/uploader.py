# core/uploader.py
# ============================================================
# MRI Uploader — 4 modalities only (no manual seg required)
# DynUNet generates the segmentation mask automatically.
# ============================================================

import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from utils.file_ops import create_temp_dir, save_uploaded_file, detect_modality
from utils.file_ops import load_nifti, get_best_slice
from utils.plotting import render_tumor_legend
from core.pipeline  import launch_pipeline



def upload_mri_files():
    st.subheader(" Upload MRI Files")
    st.markdown(
        "Upload **4 NIfTI files** "
        "`t1`, `t1ce`, `t2`, `flair`. "
        "The segmentation mask will be **generated automatically** by Artifcial intellegence "
    )

    temp_dir = create_temp_dir()

    # ── File uploader ─────────────────────────────────────────────────────────
    if "uploaded_files" not in st.session_state:
        uploaded_files = st.file_uploader(
            "",
            type=["nii", "nii.gz"],
            accept_multiple_files=True,
            key="step1_uploader",
        )

        if uploaded_files:
            if len(uploaded_files) < 1:
                st.info("Upload at least 1 NIfTI file (FLAIR is required).")
                return

            if len(uploaded_files) > 4:
                st.error(
                    f"❌ {len(uploaded_files)} files uploaded. "
                    "Upload at most 4 files (FLAIR, T1, T1CE, T2). "
                    "Do NOT upload the segmentation mask — DynUNet generates it."
                )
                return

            with st.spinner("Saving files and detecting modalities..."):
                assigned     = {}
                unrecognised = []
                for f in uploaded_files:
                    path     = save_uploaded_file(f, temp_dir)
                    modality = detect_modality(f.name)

                    # Reject seg file if user accidentally uploads it
                    if modality == "seg":
                        st.warning(
                            f"⚠️ Segmentation file `{f.name}` ignored. "
                            "Artificial intellegence will generate the mask automatically."
                        )
                        continue

                    if modality == "unknown":
                        unrecognised.append(f.name)
                    else:
                        assigned[modality] = path
                        st.session_state[f"{modality}_path"] = path

            if unrecognised:
                st.warning(
                    f"⚠️ Could not detect modality for: {', '.join(unrecognised)}. "
                    "Rename files to include: t1, t1ce, t2, flair in the filename."
                )

            REQUIRED = {"flair", "t1", "t1ce", "t2"}
            missing  = REQUIRED - set(assigned.keys())

            if missing:
                st.error(
                    f"❌ Missing required modalities: {', '.join(sorted(missing)).upper()}. "
                    "All four modalities (FLAIR, T1, T1CE, T2) are required — "
                    "segmentation cannot proceed with incomplete data."
                )
                return

            # Reject duplicate files reused across multiple modality slots
            import hashlib
            def _file_hash(path):
                return hashlib.md5(open(path, "rb").read()).hexdigest()

            hashes = {mod: _file_hash(assigned[mod]) for mod in REQUIRED}
            if len(set(hashes.values())) < 4:
                st.error(
                    "❌ Duplicate file detected across modalities — "
                    "each of the 4 modalities must be a distinct real MRI sequence."
                )
                return

            st.session_state["uploaded_files"]      = uploaded_files
            st.session_state["assigned_modalities"] = assigned

            n = len(assigned)
           
            launch_pipeline()

           

    # ── Preview uploaded modalities ───────────────────────────────────────────
    if "uploaded_files" in st.session_state:
        st.subheader("MRI Modality Previews")

        modality_order = ["t1", "t1ce", "t2", "flair"]
        files_to_show  = [
            (mod, st.session_state[f"{mod}_path"])
            for mod in modality_order
            if f"{mod}_path" in st.session_state
            and os.path.exists(st.session_state[f"{mod}_path"])
        ]

        if not files_to_show:
            return

        # Check if segmentation exists to show as 5th image
        show_seg = "seg_path" in st.session_state and st.session_state.get("seg_auto", False)
        num_cols = len(files_to_show) + (1 if show_seg else 0)

        cols = st.columns(num_cols)
        for i, (mod, path) in enumerate(files_to_show):
            data = load_nifti(path)
            if data is None:
                continue
            best_z = get_best_slice(data, axis=(0, 1))
            fig, ax = plt.subplots(figsize=(2.3,2.3))
            ax.imshow(data[:, :, best_z], cmap="gray", origin="lower")
            ax.set_title(mod.upper(), fontsize=9, pad=4)
            ax.axis("off")
            plt.tight_layout(pad=0.2)
            cols[i].pyplot(fig, use_container_width=True)
            plt.close(fig)

        if show_seg:
            seg_data = load_nifti(st.session_state["seg_path"])
            if seg_data is not None:
                best_z = get_best_slice(seg_data, axis=(0, 1))

                seg_sl = seg_data[:, :, best_z]

                # Create RGB overlay (black background)
                rgb = np.zeros((*seg_sl.shape, 3), dtype=np.float32)
                rgb[seg_sl == 2] = [0.10, 0.45, 1.00]   # edema  → blue
                rgb[seg_sl == 3] = [1.00, 0.20, 0.20]   # ET     → red
                rgb[seg_sl == 1] = [1.00, 0.88, 0.10]   # core   → yellow

                fig, ax = plt.subplots(figsize=(2.2, 2.2))
                ax.imshow(rgb, origin="lower")
                ax.set_title("MASK", fontsize=9, pad=4)
                ax.axis("off")
                plt.tight_layout(pad=0.2)
                cols[-1].pyplot(fig, use_container_width=True)
                plt.close(fig)

                render_tumor_legend()

        with st.expander(""):
            for mod, path in files_to_show:
                st.markdown(f"*{mod.upper()}* → {os.path.basename(path)}")
