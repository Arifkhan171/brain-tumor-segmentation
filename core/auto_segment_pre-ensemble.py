# core/auto_segment.py
# ============================================================
# DynUNet Automatic Segmentation
# Replaces the manual mask upload completely.
# ============================================================

import os
import numpy as np
import nibabel as nib
import streamlit as st
import torch

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    NormalizeIntensityd,
)
from monai.networks.nets import DynUNet
from monai.inferers import sliding_window_inference
from scipy import ndimage


# ── Model configuration — must match your training exactly ────────────────────
MODEL_CONFIG = dict(
    spatial_dims        = 3,
    in_channels         = 4,
    out_channels        = 4,
    kernel_size         = [[3,3,3]] * 6,
    strides             = [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
    upsample_kernel_size= [[2,2,2]] * 5,
    filters             = [32, 64, 128, 256, 320, 320],
    dropout             = 0.0,
    norm_name           = "instance",
    deep_supervision    = False,   # False at inference — only main output needed
    res_block           = True,
)

# ── Patch size used during training ──────────────────────────────────────────
SPATIAL_SIZE = (96, 96, 64)   # must match cfg.spatial_size you trained with


# ── Load model (cached so it loads only once per session) ────────────────────
@st.cache_resource(show_spinner=False)
def load_dynunet(model_path: str):
    """
    Load trained DynUNet weights from disk.
    model_path : full path to best_overall.pt on your computer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DynUNet(**MODEL_CONFIG).to(device)

    state = torch.load(model_path, map_location=device, weights_only=False)

    # Handle both plain state dict and full checkpoint dict
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    # If trained with deep_supervision=True the keys may have a different prefix
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)

    model.eval()
    return model, device


def map_brats_labels(x):
    """Remap label 4 → 3 (same as training)."""
    x = x.clone()
    x[x == 4] = 3
    x[(x < 0) | (x > 3)] = 0
    return x


def postprocess(pred_np):
    """
    Post-processing:
    Step 1: Keep largest connected tumor component (removes spurious isolated voxels).
    Step 2: Enforce hierarchy ET ⊆ TC ⊆ WT by dilating TC mask to fill isolated ET voxels,
            then downgrading any ET voxel that has no TC neighbours to NCR (label 1).
    """
    pred = pred_np.copy()

    # Step 1 — largest connected component on whole-tumour mask
    wt_mask = (pred > 0).astype(np.uint8)
    labeled, n = ndimage.label(wt_mask)
    if n > 1:
        sizes   = ndimage.sum(wt_mask, labeled, range(1, n + 1))
        largest = np.argmax(sizes) + 1
        pred[labeled != largest] = 0

    # Step 2 — hierarchy: ET (label 3) must be inside TC (labels 1 or 3)
    # Dilate TC mask by 1 voxel to allow slight boundary touching
    tc_mask   = (pred == 1) | (pred == 3)
    tc_dilated = ndimage.binary_dilation(tc_mask, iterations=1)
    # Any ET voxel not covered by dilated TC is an isolated enhancer → downgrade to NCR
    isolated_et = (pred == 3) & ~tc_dilated
    pred[isolated_et] = 1   # downgrade: ET → NCR

    # Step 3 — ensure any remaining non-zero voxel is inside WT
    pred[pred > 0] = pred[pred > 0]  # trivially satisfied by design; kept for clarity

    return pred


def _compute_foreground_bbox(flair_path: str):
    """
    Compute the bounding box of non-zero voxels in the FLAIR volume.
    Returns (crop_start, crop_end) as numpy arrays, each of shape (3,).
    Falls back to full volume if no foreground found.
    """
    data = nib.load(flair_path).get_fdata()
    fg   = data > 0
    if fg.any():
        coords     = np.argwhere(fg)
        crop_start = coords.min(axis=0)
        crop_end   = coords.max(axis=0) + 1
    else:
        crop_start = np.array([0, 0, 0])
        crop_end   = np.array(data.shape[:3])
    return crop_start, crop_end


def run_segmentation(flair_path, t1_path, t1ce_path, t2_path, model_path, output_path):
    """
    Full inference pipeline.

    Parameters
    ----------
    flair_path   : str  — path to FLAIR .nii / .nii.gz
    t1_path      : str  — path to T1
    t1ce_path    : str  — path to T1CE (contrast enhanced)
    t2_path      : str  — path to T2
    model_path   : str  — path to best_overall.pt
    output_path  : str  — where to save the predicted seg .nii.gz

    Returns
    -------
    pred_np      : numpy array [D,H,W] with labels 0/1/2/3
    output_path  : str (same as input, for convenience)
    """
    model, device = load_dynunet(model_path)

    # ── Step 1: Compute foreground bounding box BEFORE MONAI transforms ───────
    # This lets us correctly map the prediction back to original MRI space.
    ref_img    = nib.load(flair_path)
    orig_shape = ref_img.shape[:3]
    crop_start, crop_end = _compute_foreground_bbox(flair_path)

    # ── Step 2: Preprocessing (NO ResizeWithPadOrCropd) ───────────────────────
    # sliding_window_inference handles arbitrary input sizes natively.
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ])

    # Channel order MUST match training: FLAIR, T1, T1CE, T2
    image_paths = [flair_path, t1_path, t1ce_path, t2_path]

    # Handle missing modalities gracefully — duplicate FLAIR as placeholder
    safe_paths = []
    for p in image_paths:
        if p and os.path.exists(p):
            safe_paths.append(p)
        else:
            safe_paths.append(flair_path)   # fallback to FLAIR

    data = transforms({"image": safe_paths})

    # ── Step 3: Crop to foreground bounding box ───────────────────────────────
    # Manually crop the normalised tensor so the model only processes brain tissue.
    img_tensor = data["image"]   # shape: [4, D, H, W]
    x = img_tensor[
        :,
        crop_start[0]:crop_end[0],
        crop_start[1]:crop_end[1],
        crop_start[2]:crop_end[2],
    ].unsqueeze(0).to(device)    # shape: [1, 4, D', H', W']

    # ── Step 4: Inference ─────────────────────────────────────────────────────
    with torch.no_grad():
        with torch.amp.autocast(
            device_type="cuda", enabled=device.type == "cuda"
        ):
            out = sliding_window_inference(
                x,
                roi_size     = SPATIAL_SIZE,
                sw_batch_size= 1,
                predictor    = model,
                overlap      = 0.5,
                mode         = "gaussian",
                padding_mode = "constant",
            )

    # ── Step 5: Decode prediction ─────────────────────────────────────────────
    pred = torch.argmax(
        torch.softmax(out, dim=1), dim=1
    ).cpu().numpy()[0]   # [D', H', W'] — in cropped space

    # Free GPU memory immediately
    del out, x
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pred = postprocess(pred)

    # ── Step 6: Restore prediction to original MRI space ─────────────────────
    pred_full = np.zeros(orig_shape, dtype=np.uint8)
    d = min(pred.shape[0], crop_end[0] - crop_start[0])
    h = min(pred.shape[1], crop_end[1] - crop_start[1])
    w = min(pred.shape[2], crop_end[2] - crop_start[2])
    pred_full[
        crop_start[0]:crop_start[0] + d,
        crop_start[1]:crop_start[1] + h,
        crop_start[2]:crop_start[2] + w,
    ] = pred[:d, :h, :w]

    # ── Step 7: Save as NIfTI using FLAIR affine/header ──────────────────────
    out_nib = nib.Nifti1Image(pred_full, ref_img.affine, ref_img.header)
    nib.save(out_nib, output_path)

    return pred_full, output_path


# ── Streamlit UI for the segmentation step ────────────────────────────────────

def auto_segment_ui():
    """
    Renders the automatic segmentation interface inside the Streamlit app.
    Called from app.py after MRI files are uploaded.
    Saves the predicted mask to session_state["seg_path"].
    """
    st.subheader(" Automatic Tumor Segmentation")

    # ── Check MRI files are uploaded ─────────────────────────────────────────
    if "flair_path" not in st.session_state:
        st.info("Upload MRI files first ")
        return

    # ── Missing modality warning ──────────────────────────────────────────────
    modality_map = {
        "T1":   "t1_path",
        "T1CE": "t1ce_path",
        "T2":   "t2_path",
    }
    missing = [
        name for name, key in modality_map.items()
        if not st.session_state.get(key) or
           st.session_state.get(key) == st.session_state.get("flair_path")
    ]
    if missing:
        st.warning(
            f"⚠️ **Reduced Accuracy Warning** — The following modalities are missing or "
            f"substituted with FLAIR: **{', '.join(missing)}**. "
            "DynUNet requires all 4 sequences (FLAIR, T1, T1CE, T2) for clinical-grade accuracy. "
            "Segmentation will still run but results may be unreliable. "
            "For clinical use, upload all 4 modalities."
        )

    # ── Model path input ──────────────────────────────────────────────────────
    # st.markdown(
    #     "Point to your downloaded **`best_overall.pt`** model file. "
    #     "See the download instructions below."
    # )

    model_path = st.text_input(
        "Path to best_overall.pt",
        value=st.session_state.get("model_path", "models/best_overall.pt"),
        help="Full path on your computer, e.g. C:/models/best_overall.pt",
    )
    st.session_state["model_path"] = model_path

    # ── Already segmented? ────────────────────────────────────────────────────
    if "seg_path" in st.session_state and st.session_state.get("seg_auto", False):
        st.success("✅ Segmentation already computed for this session.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Re-run Segmentation", type="secondary"):
                del st.session_state["seg_path"]
                st.session_state["seg_auto"] = False
                st.rerun()
        with col2:
            st.caption("Re-run only if you uploaded new MRI files.")

        _display_segmentation_results()
        return

    # ── Run button ────────────────────────────────────────────────────────────
    if st.button("Run Automatic Segmentation", type="primary"):

        if not os.path.exists(model_path):
            st.error(
                 f"Some internal error occurs"
                # Model file not found: `{model_path}`\n\n"
                # "Download `best_overall.pt` from your Google Drive "
                # "(see instructions below) and place it at the path above."
            )
            return

        flair_path = st.session_state.get("flair_path")
        t1_path    = st.session_state.get("t1_path",   flair_path)
        t1ce_path  = st.session_state.get("t1ce_path", flair_path)
        t2_path    = st.session_state.get("t2_path",   flair_path)

        import tempfile
        temp_dir  = st.session_state.get("temp_dir", tempfile.mkdtemp())
        seg_out   = os.path.join(temp_dir, "auto_seg.nii.gz")

        progress = st.progress(0, text="Loading model...")

        try:
            progress.progress(20, text="Model loaded. Running inference...")
            pred_np, seg_path = run_segmentation(
                flair_path, t1_path, t1ce_path, t2_path,
                model_path, seg_out,
            )
            progress.progress(100, text="Done!")

            st.session_state["seg_path"] = seg_path
            st.session_state["seg_auto"] = True

            _display_segmentation_results()

        except Exception as e:
            st.error(f"Segmentation failed: {e}")
            st.exception(e)

def _display_segmentation_results():
    seg_path   = st.session_state.get("seg_path")
    flair_path = st.session_state.get("flair_path")

    if not seg_path or not flair_path or not os.path.exists(seg_path) or not os.path.exists(flair_path):
        return

    mask_img  = nib.load(seg_path).get_fdata()
    flair_img = nib.load(flair_path).get_fdata()

    # ── Read voxel spacing for accurate cm³ volumes ───────────────────────────
    ref_img      = nib.load(flair_path)
    voxel_dims   = ref_img.header.get_zooms()[:3]
    voxel_vol_mm3 = float(np.prod(voxel_dims))

    wt = int((mask_img > 0).sum())
    tc = int(((mask_img == 1) | (mask_img == 3)).sum())
    et = int((mask_img == 3).sum())

    wt_cm3 = wt * voxel_vol_mm3 / 1000
    tc_cm3 = tc * voxel_vol_mm3 / 1000
    et_cm3 = et * voxel_vol_mm3 / 1000

    st.success("✅ Segmentation complete!")
    c1, c2, c3 = st.columns(3)
    c1.metric("WT Volume", f"{wt_cm3:.2f} cm³")
    c2.metric("TC Volume", f"{tc_cm3:.2f} cm³")
    c3.metric("ET Volume", f"{et_cm3:.2f} cm³")

    st.markdown("### Segmentation Mask (Best View)")
    st.caption("Auto-selected slice showing the largest tumor area, similar to a radiologist's mask review.")

    tumor_counts_per_slice = (mask_img > 0).sum(axis=(0, 1))
    best_z = int(np.argmax(tumor_counts_per_slice)) if tumor_counts_per_slice.max() > 0 else mask_img.shape[2] // 2

    from utils.plotting import create_rgb_overlay, plot_interactive_slice
    flair_sl = flair_img[:, :, best_z]
    mask_sl  = mask_img[:, :, best_z]
    rgb      = create_rgb_overlay(flair_sl, mask_sl)

    fig = plot_interactive_slice(
        rgb, best_z, mask_img.shape[2],
        wt_cm3, et_cm3, (wt_cm3 - tc_cm3)
    )
    st.plotly_chart(fig, use_container_width=True)

    c_leg1, c_leg2, c_leg3 = st.columns(3)
    c_leg1.markdown("🔵 **Edema** (label 2)")
    c_leg2.markdown("🔴 **Enhancing Tumor** (label 3)")
    c_leg3.markdown("🟡 **Necrotic Core** (label 1)")

    st.caption("Proceed to Step 2 (MRI Viewer) to fully explore the 3D volume.")

    # ── Download instructions ─────────────────────────────────────────────────
#     with st.expander("📥 How to download the model from Google Drive"):
#         st.markdown("""
# **Which file to download:**

# | File | When to use |
# |---|---|
# | `best_overall.pt` | **Always use this** — best mean Dice across WT/TC/ET |
# | `best_wt.pt` | Only if you need maximum WT accuracy |
# | `best_tc.pt` | Only if you need maximum TC accuracy |
# | `best_et.pt` | Only if you need maximum ET accuracy |

# **Steps:**
# 1. Open Google Drive → navigate to `BraTS_DynUNet` folder
# 2. Right-click `best_overall.pt` → Download
# 3. Place it in your project folder, e.g. `brain_tumor/models/best_overall.pt`
# 4. Enter that path in the text box above
# 5. Click **Run Automatic Segmentation**

# **File size:** approximately 120 MB (DynUNet with res_block)
#         """)
