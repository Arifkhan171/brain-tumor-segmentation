import os
import tempfile
import streamlit as st
import numpy as np
import nibabel as nib


# ─────────────────────────────────────────────────────────────────────────────
# Temporary directory management
# ─────────────────────────────────────────────────────────────────────────────

def create_temp_dir():
    """Create a temporary directory stored in session_state (not globals)."""
    if "temp_dir" not in st.session_state:
        st.session_state["temp_dir"] = tempfile.mkdtemp()
    return st.session_state["temp_dir"]


# ─────────────────────────────────────────────────────────────────────────────
# File saving
# ─────────────────────────────────────────────────────────────────────────────

def save_uploaded_file(uploaded_file, temp_dir):
    """Save an uploaded Streamlit file object to disk. Returns the file path."""
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# ─────────────────────────────────────────────────────────────────────────────
# Modality detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_modality(filename: str) -> str:
    """
    Detect MRI modality from filename.
    Returns one of: 't1', 't1ce', 't2', 'flair', 'seg', 'unknown'.
    Order matters — check t1ce before t1.
    """
    name = filename.lower()
    if "t1ce" in name or "t1c" in name or "t1_ce" in name or "ce" in name:
        return "t1ce"
    elif "flair" in name:
        return "flair"
    elif "t2" in name:
        return "t2"
    elif "t1" in name:
        return "t1"
    elif "seg" in name or "mask" in name or "label" in name or "gt" in name:
        return "seg"
    return "unknown"


def assign_modality_paths(uploaded_files, temp_dir):
    """
    Save files and assign session_state paths for each modality.
    Returns: (assigned: dict, unrecognised: list)
    """
    assigned     = {}
    unrecognised = []
    for f in uploaded_files:
        path     = save_uploaded_file(f, temp_dir)
        modality = detect_modality(f.name)
        if modality == "unknown":
            unrecognised.append(f.name)
        else:
            assigned[modality] = path
            st.session_state[f"{modality}_path"] = path
    return assigned, unrecognised


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITY 1 — NIfTI loader (moved here from uploader.py)
# Used by: viewer.py, feature_report.py, auto_segment.py,
#          tumor_habitate.py, longitudinal_compare.py
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_nifti(path: str) -> np.ndarray:
    """
    Load a NIfTI file and return its 3D data array. Cached per path.
    Returns None on failure instead of crashing.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        data = nib.load(path).get_fdata()

        if len(data.shape) not in [3, 4]:
            st.error(
                f"File `{os.path.basename(path)}` is {len(data.shape)}D. "
                "Expected a 3D MRI volume."
            )
            return None

        if len(data.shape) == 4:
            if data.shape[3] == 1:
                data = data[:, :, :, 0]
            else:
                st.error(
                    f"File `{os.path.basename(path)}` has 4 dimensions {data.shape}. "
                    "Expected a static 3D volume, not a time-series."
                )
                return None

        if any(d < 10 for d in data.shape[:3]):
            st.warning(
                f"File `{os.path.basename(path)}` has a very small dimension "
                f"{data.shape}. Check file integrity."
            )

        return data

    except Exception as e:
        st.warning(
            f"Could not load NIfTI file `{os.path.basename(path)}`: {e}"
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITY 2 — Best tumor slice finder
# Replaces the duplicated (mask > 0).sum(axis=(0,1)) pattern
# used in 5 different files
# ─────────────────────────────────────────────────────────────────────────────

def get_best_slice(mask_3d: np.ndarray, axis: tuple = (0, 1)) -> int:
    """
    Return the index of the slice with the most tumor voxels.
    axis=(0,1) → best axial (Z) slice  [default]
    axis=(0,2) → best coronal (Y) slice
    axis=(1,2) → best sagittal (X) slice
    Falls back to middle slice if mask is empty.
    """
    counts = (mask_3d > 0).sum(axis=axis)
    dim    = [i for i in range(3) if i not in axis][0]
    return int(np.argmax(counts)) if counts.max() > 0 else mask_3d.shape[dim] // 2


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITY 3 — Physical voxel volume from NIfTI header
# Replaces the duplicated /1000 assumption across files
# ─────────────────────────────────────────────────────────────────────────────

def get_voxel_volume_mm3(nifti_path: str) -> float:
    """
    Return the volume of one voxel in mm³ by reading the NIfTI header.
    Falls back to 1.0 (1mm isotropic) if header cannot be read.
    """
    try:
        zooms = nib.load(nifti_path).header.get_zooms()[:3]
        return float(np.prod(zooms))
    except Exception:
        return 1.0


def get_voxel_dims(nifti_path: str) -> tuple:
    """
    Return (dx, dy, dz) voxel spacing in mm from the NIfTI header.
    Falls back to (1.0, 1.0, 1.0) if header cannot be read.
    """
    try:
        return tuple(float(z) for z in nib.load(nifti_path).header.get_zooms()[:3])
    except Exception:
        return (1.0, 1.0, 1.0)
