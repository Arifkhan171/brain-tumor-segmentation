# core/auto_segment.py
# ─────────────────────────────────────────────────────────────────────────────
# Automatic tumour segmentation — DynUNet / SwinUNETR / SegResNet ensemble
#
# Three models vote per-voxel. Majority wins. Falls back gracefully to
# whichever models are available — the system works with just one model.
# ─────────────────────────────────────────────────────────────────────────────

import os
import tempfile
import numpy as np
import nibabel as nib
import streamlit as st
import torch

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
)
from monai.inferers import sliding_window_inference
from scipy import ndimage

from utils.file_ops import get_best_slice
from utils.plotting import create_rgb_overlay, plot_interactive_slice, render_tumor_legend

# ─────────────────────────────────────────────────────────────────────────────
# Model architecture configs
# ─────────────────────────────────────────────────────────────────────────────

DYNUNET_CONFIG = dict(
    spatial_dims         = 3,
    in_channels          = 4,
    out_channels         = 4,
    kernel_size          = [[3,3,3]] * 6,
    strides              = [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
    upsample_kernel_size = [[2,2,2]] * 5,
    filters              = [32, 64, 128, 256, 320, 320],
    dropout              = 0.0,
    norm_name            = "instance",
    deep_supervision     = False,
    res_block            = True,
)

SWINUNETR_CONFIG = dict(
    
    in_channels     = 4,
    out_channels    = 4,
    feature_size    = 48,
    use_checkpoint  = True,
)

SEGRESNET_CONFIG = dict(
    spatial_dims    = 3,
    in_channels     = 4,
    out_channels    = 4,
    init_filters    = 32,
)

SPATIAL_SIZE = (96, 96, 64)


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders — each cached independently
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_dynunet(model_path: str):
    from monai.networks.nets import DynUNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DynUNet(**DYNUNET_CONFIG).to(device)
    state  = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model, device


@st.cache_resource(show_spinner=False)
def _load_swinunetr(model_path: str):
    try:
        from monai.networks.nets import SwinUNETR
    except ImportError:
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SwinUNETR(**SWINUNETR_CONFIG).to(device)
    state  = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model, device


@st.cache_resource(show_spinner=False)
def _load_segresnet(model_path: str):
    try:
        from monai.networks.nets import SegResNet
    except ImportError:
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SegResNet(**SEGRESNET_CONFIG).to(device)
    state  = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# Shared inference — same for all three models
# ─────────────────────────────────────────────────────────────────────────────

def _run_inference(model, x, device) -> np.ndarray:
    """
    Run sliding window inference and return a soft probability array.
    Shape: (4, H, W, D) — 4 class probability channels.
    """
    with torch.no_grad():
        with torch.amp.autocast(
            device_type="cuda", enabled=device.type == "cuda"
        ):
            out = sliding_window_inference(
                x,
                roi_size      = SPATIAL_SIZE,
                sw_batch_size = 1,
                predictor     = model,
                overlap       = 0.5,
                mode          = "gaussian",
                padding_mode  = "constant",
            )
    probs = torch.softmax(out, dim=1)[0].cpu().numpy()  # (4, H, W, D)
    del out
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def postprocess(pred_np: np.ndarray) -> np.ndarray:
    """
    1. Keep only the largest connected component (removes isolated noise).
    2. Downgrade isolated ET voxels to NCR (enforces ET ⊆ TC).
    """
    pred = pred_np.copy()

    # Largest connected component
    wt_mask    = (pred > 0).astype(np.uint8)
    labeled, n = ndimage.label(wt_mask)
    if n > 1:
        sizes   = ndimage.sum(wt_mask, labeled, range(1, n + 1))
        largest = np.argmax(sizes) + 1
        pred[labeled != largest] = 0

    # ET must be adjacent to TC
    tc_mask    = (pred == 1) | (pred == 3)
    tc_dilated = ndimage.binary_dilation(tc_mask, iterations=1)
    isolated_et = (pred == 3) & ~tc_dilated
    pred[isolated_et] = 1

    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Foreground bounding box
# ─────────────────────────────────────────────────────────────────────────────

def _compute_foreground_bbox(flair_path: str):
    data = nib.load(flair_path).get_fdata()
    fg   = data > 0
    if fg.any():
        coords = np.argwhere(fg)
        return coords.min(axis=0), coords.max(axis=0) + 1
    return np.array([0, 0, 0]), np.array(data.shape[:3])


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — shared by all models
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(flair_path, t1_path, t1ce_path, t2_path, crop_start, crop_end, device):
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ])

    safe_paths = [
        p if (p and os.path.exists(p)) else flair_path
        for p in [flair_path, t1_path, t1ce_path, t2_path]
    ]

    data       = transforms({"image": safe_paths})
    img_tensor = data["image"]

    x = img_tensor[
        :,
        crop_start[0]:crop_end[0],
        crop_start[1]:crop_end[1],
        crop_start[2]:crop_end[2],
    ].unsqueeze(0).to(device)

    return x


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble segmentation — core function
# ─────────────────────────────────────────────────────────────────────────────

def run_segmentation(
    flair_path, t1_path, t1ce_path, t2_path,
    model_path, output_path,
    swin_path: str = "",
    segresnet_path: str = "",
):
    """
    Run ensemble segmentation using all available models.
    Soft probability averaging then argmax gives the final label.

    model_path      — path to best_overall.pt   (DynUNet)
    swin_path       — path to swinunetr.pt       (optional)
    segresnet_path  — path to segresnet.pt       (optional)
    """
    ref_img              = nib.load(flair_path)
    orig_shape           = ref_img.shape[:3]
    crop_start, crop_end = _compute_foreground_bbox(flair_path)

    # Determine device from DynUNet (always required)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess once — reused by all models
    x = _preprocess(
        flair_path, t1_path, t1ce_path, t2_path,
        crop_start, crop_end, device,
    )

    accumulated_probs = None
    n_models          = 0
    models_used       = []

    model_stats = {}

    # ── DynUNet ───────────────────────────────────────────────────────────────
    try:
        model, device = _load_dynunet(model_path)
        probs = _run_inference(model, x, device)
        accumulated_probs = probs
        n_models += 1
        models_used.append("DynUNet")
        # Record individual model prediction volumes
        _pred_tmp = np.argmax(probs, axis=0)
        vvol = float(np.prod(nib.load(flair_path).header.get_zooms()[:3])) / 1000
        model_stats["DynUNet"] = {
            "wt": float((_pred_tmp > 0).sum()) * vvol,
            "tc": float(((_pred_tmp==1)|(_pred_tmp==3)).sum()) * vvol,
            "et": float((_pred_tmp == 3).sum()) * vvol,
        }
    except Exception as e:
        st.warning(f"DynUNet failed: {e}")

    # ── SwinUNETR ─────────────────────────────────────────────────────────────
    if swin_path and os.path.exists(swin_path):
        try:
            model, dev = _load_swinunetr(swin_path)
            if model is not None:
                x_dev = x.to(dev)
                probs = _run_inference(model, x_dev, dev)
                accumulated_probs = (
                    probs if accumulated_probs is None
                    else accumulated_probs + probs
                )
                n_models += 1
                models_used.append("SwinUNETR")
                _pred_tmp = np.argmax(probs, axis=0)
                model_stats["SwinUNETR"] = {
                    "wt": float((_pred_tmp > 0).sum()) * vvol,
                    "tc": float(((_pred_tmp==1)|(_pred_tmp==3)).sum()) * vvol,
                    "et": float((_pred_tmp == 3).sum()) * vvol,
                }
        except Exception as e:
            st.warning(f"SwinUNETR failed: {e}")

    # ── SegResNet ─────────────────────────────────────────────────────────────
    if segresnet_path and os.path.exists(segresnet_path):
        try:
            model, dev = _load_segresnet(segresnet_path)
            if model is not None:
                x_dev = x.to(dev)
                probs = _run_inference(model, x_dev, dev)
                accumulated_probs = (
                    probs if accumulated_probs is None
                    else accumulated_probs + probs
                )
                n_models += 1
                models_used.append("SegResNet")
                _pred_tmp = np.argmax(probs, axis=0)
                model_stats["SegResNet"] = {
                    "wt": float((_pred_tmp > 0).sum()) * vvol,
                    "tc": float(((_pred_tmp==1)|(_pred_tmp==3)).sum()) * vvol,
                    "et": float((_pred_tmp == 3).sum()) * vvol,
                }
        except Exception as e:
            st.warning(f"SegResNet failed: {e}")

    if accumulated_probs is None:
        raise RuntimeError(
            "No models ran successfully. "
            "Check model paths and file integrity."
        )

    # Average probabilities and take argmax
    avg_probs = accumulated_probs / n_models
    pred      = np.argmax(avg_probs, axis=0).astype(np.uint8)

    # Post-process and restore to original space
    pred      = postprocess(pred)
    pred_full = np.zeros(orig_shape, dtype=np.uint8)
    d = min(pred.shape[0], crop_end[0] - crop_start[0])
    h = min(pred.shape[1], crop_end[1] - crop_start[1])
    w = min(pred.shape[2], crop_end[2] - crop_start[2])
    pred_full[
        crop_start[0]:crop_start[0] + d,
        crop_start[1]:crop_start[1] + h,
        crop_start[2]:crop_start[2] + w,
    ] = pred[:d, :h, :w]

    nib.save(
        nib.Nifti1Image(pred_full, ref_img.affine, ref_img.header),
        output_path,
    )

    # Store which models contributed and their individual stats
    st.session_state["ensemble_models_used"]  = models_used
    st.session_state["ensemble_model_stats"]  = model_stats

    return pred_full, output_path


# ─────────────────────────────────────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────────────────────────────────────

def _display_segmentation_results():
    seg_path   = st.session_state.get("seg_path")
    flair_path = st.session_state.get("flair_path")

    if not seg_path or not flair_path:
        return
    if not os.path.exists(seg_path) or not os.path.exists(flair_path):
        return

    ref_img       = nib.load(flair_path)
    flair_img     = ref_img.get_fdata()
    voxel_vol_mm3 = float(np.prod(ref_img.header.get_zooms()[:3]))
    mask_img      = nib.load(seg_path).get_fdata()

    wt_cm3 = float((mask_img > 0).sum())                      * voxel_vol_mm3 / 1000
    tc_cm3 = float(((mask_img==1)|(mask_img==3)).sum())       * voxel_vol_mm3 / 1000
    et_cm3 = float((mask_img == 3).sum())                     * voxel_vol_mm3 / 1000
    ed_cm3 = wt_cm3 - tc_cm3

    # Show which models were used in the ensemble
    models_used = st.session_state.get("ensemble_models_used", ["DynUNet"])
    st.success(
        f"Segmentation complete — "
        f"{'Ensemble: ' if len(models_used) > 1 else ''}"
        f"{' + '.join(models_used)}"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Whole Tumor", f"{wt_cm3:.2f} cm³")
    c2.metric("Tumor Core",  f"{tc_cm3:.2f} cm³")
    c3.metric("Enhancing",   f"{et_cm3:.2f} cm³")

    best_z = get_best_slice(mask_img, axis=(0, 1))
    rgb    = create_rgb_overlay(flair_img[:, :, best_z], mask_img[:, :, best_z])
    fig    = plot_interactive_slice(rgb, best_z, mask_img.shape[2], wt_cm3, et_cm3, ed_cm3)
    st.plotly_chart(fig, use_container_width=True)
    render_tumor_legend()

    # Show ensemble composition if multiple models were used
    models_used = st.session_state.get("ensemble_models_used", [])
    if len(models_used) > 1:
        st.divider()
        st.caption(
            f"Ensemble result from {len(models_used)} models: "
            f"{' + '.join(models_used)}. "
            "Soft probability averaging was used — each model's class "
            "probabilities were averaged before taking the final label."
        )

        # Per-model volume contribution table
        ensemble_stats = st.session_state.get("ensemble_model_stats", {})
        if ensemble_stats:
            import pandas as pd
            rows = []
            for mname, stats in ensemble_stats.items():
                rows.append({
                    "Model":      mname,
                    "WT (cm³)":   f"{stats.get('wt', 0):.2f}",
                    "TC (cm³)":   f"{stats.get('tc', 0):.2f}",
                    "ET (cm³)":   f"{stats.get('et', 0):.2f}",
                })
            rows.append({
                "Model":    "Ensemble (Final)",
                "WT (cm³)": f"{wt_cm3:.2f}",
                "TC (cm³)": f"{tc_cm3:.2f}",
                "ET (cm³)": f"{et_cm3:.2f}",
            })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation page UI
# ─────────────────────────────────────────────────────────────────────────────

def auto_segment_ui():
    from core.patient import render_patient_bar
    render_patient_bar()

    st.subheader("Automatic Tumour Segmentation")

    if "flair_path" not in st.session_state:
        st.info("Complete Step 1 (Upload MRI Files) first.")
        return

    # Missing modality warning
    missing = [
        name for name, key in {
            "T1": "t1_path", "T1CE": "t1ce_path", "T2": "t2_path"
        }.items()
        if not st.session_state.get(key)
        or st.session_state.get(key) == st.session_state.get("flair_path")
    ]
    if missing:
        st.warning(
            f"Missing modalities — {', '.join(missing)} substituted with FLAIR. "
            "Upload all 4 files for clinical-grade accuracy."
        )

    device_label = "GPU" if torch.cuda.is_available() else "CPU"
    est_time     = "~30 sec" if torch.cuda.is_available() else "~3–5 min"

    # ── Model path configuration ───────────────────────────────────────────────
    with st.expander("Model Configuration", expanded=False):
        # st.caption(
        #     "DynUNet is required. SwinUNETR and SegResNet are optional — "
        #     "if their model files exist, they join the ensemble automatically."
        # )

        model_path = st.text_input(
            "DynUNet — best_overall.pt",
            value=st.session_state.get("model_path", "models/best_overall.pt"),
        )
        swin_path = st.text_input(
            "SwinUNETR — swinunetr.pt  (optional)",
            value=st.session_state.get("swin_path", "models/swinunetr.pt"),
        )
        segresnet_path = st.text_input(
            "SegResNet — segresnet.pt  (optional)",
            value=st.session_state.get("segresnet_path", "models/segresnet.pt"),
        )

        st.session_state["model_path"]      = model_path
        st.session_state["swin_path"]       = swin_path
        st.session_state["segresnet_path"]  = segresnet_path

        # Show which models are available
        available = []
        if os.path.exists(model_path):      available.append("DynUNet ✓")
        if os.path.exists(swin_path):       available.append("SwinUNETR ✓")
        if os.path.exists(segresnet_path):  available.append("SegResNet ✓")
        missing_models = []
        if not os.path.exists(swin_path):       missing_models.append("SwinUNETR")
        if not os.path.exists(segresnet_path):  missing_models.append("SegResNet")

        # if available:
        #     st.caption(f"Available: {', '.join(available)}")
        # if missing_models:
        #     st.caption(
        #         f"Not found (will run DynUNet only): {', '.join(missing_models)}"
        #     )

    # st.caption(f'it can take 2-5 minute on cpu')

    # ── Already segmented ─────────────────────────────────────────────────────
    if "seg_path" in st.session_state and st.session_state.get("seg_auto", False):
        st.success("Segmentation already computed for this session.")
        col1, _ = st.columns([1, 4])
        with col1:
            if st.button("Re-run", type="secondary"):
                del st.session_state["seg_path"]
                st.session_state["seg_auto"] = False
                st.session_state.pop("ensemble_models_used", None)
                st.rerun()
        _display_segmentation_results()
        return

    # ── Run button ────────────────────────────────────────────────────────────
    if st.button("Run Segmentation", type="primary"):
        if not os.path.exists(model_path):
            st.error(
                f"DynUNet model not found at `{model_path}`. "
                "Download best_overall.pt and set the path above."
            )
            return

        flair_path = st.session_state.get("flair_path")
        t1_path    = st.session_state.get("t1_path",   flair_path)
        t1ce_path  = st.session_state.get("t1ce_path", flair_path)
        t2_path    = st.session_state.get("t2_path",   flair_path)

        temp_dir = st.session_state.get("temp_dir", tempfile.mkdtemp())
        seg_out  = os.path.join(temp_dir, "auto_seg.nii.gz")

        n_active = sum([
            1,
            int(os.path.exists(swin_path)),
            int(os.path.exists(segresnet_path)),
        ])
        mode_label = (
            f"{n_active}-model ensemble"
            if n_active > 1 else "DynUNet"
        )

        progress = st.progress(0, text=f"Starting {mode_label}...")

        try:
            progress.progress(10, text="Loading model weights...")
            pred_np, seg_path = run_segmentation(
                flair_path, t1_path, t1ce_path, t2_path,
                model_path, seg_out,
                swin_path       = swin_path,
                segresnet_path  = segresnet_path,
            )
            progress.progress(90, text="Post-processing...")
            progress.progress(100, text="Done.")

            st.session_state["seg_path"] = seg_path
            st.session_state["seg_auto"] = True
            _display_segmentation_results()

        except Exception as e:
            st.error(f"Segmentation failed: {e}")
            st.exception(e)
