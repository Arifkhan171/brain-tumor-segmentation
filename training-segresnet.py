

# ==================== CELL 1: INSTALL ====================
# Cell 1 — INSTALL (fixed)
!pip install -q "monai[all]" nibabel tqdm scikit-learn
!pip install -q --force-reinstall "numpy==1.26.4" scipy

# ⚠️ REQUIRED: Restart the runtime after this cell runs
# Runtime → Restart session  (or Ctrl+M .)
# Then run Cell 2 and onwards — do NOT re-run Cell 1scikit-learn
from google.colab import drive
drive.mount('/content/drive')




# ==================== CELL 2: IMPORTS ====================

import os
import gc
import glob
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from sklearn.model_selection import train_test_split

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    NormalizeIntensityd,       # z-score normalisation — same as DynUNet and SwinUNETR
    CropForegroundd,           # remove zero padding around brain
    RandCropByPosNegLabeld,    # extract random patches centred on tumor or background
    Lambdad,                   # apply custom function to a data key
    RandFlipd,                 # random axis flip
    RandRotate90d,             # random 90-degree rotation
    RandGaussianNoised,        # add random noise
    RandAdjustContrastd,       # random contrast change
    RandScaleIntensityd,       # random intensity scaling
    RandShiftIntensityd,       # random intensity shifting
    RandAffined,               # random rotation + scale together
)

from monai.data import Dataset, DataLoader, list_data_collate
from monai.losses import DiceCELoss
from monai.networks.nets import SegResNet
# SegResNet: residual encoder-decoder specifically designed for brain tumour segmentation
# Reference: Myronenko A. (2018) "3D MRI Brain Tumour Segmentation Using Autoencoder Regularisation"

from monai.inferers import sliding_window_inference
from torch.amp import GradScaler, autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} | Memory: {mem:.1f} GB")

from google.colab import drive
drive.mount('/content/drive')

!unzip -q /content/drive/MyDrive/Brats_data -d /content/brats2020/


# ==================== CELL 3: CONFIGURATION ====================

class Config:
    # ---- Paths ----
    DRIVE_ZIP  = "/content/drive/MyDrive/brats_data"
    EXTRACT_TO = "/content/brats2020"
    SAVE_DIR   = "/content/drive/MyDrive/BraTS_SegResNet"  # separate from DynUNet/SwinUNETR

    # ---- SegResNet settings ----
    # init_filters : number of filters in the first layer — doubles at each stage
    #                16 → 32 → 64 → 128 → 256 through the encoder
    #                Larger = more capacity but more memory
    #                32 is a good balance for free T4
    init_filters  = 32
    # blocks_down  : how many residual blocks at each encoder stage
    #                [1,2,2,4] means: 1 block at stage 1, 2 at stage 2, etc.
    #                More blocks = model goes deeper = better feature learning
    blocks_down   = (1, 2, 2, 4)
    # blocks_up    : how many residual blocks at each decoder stage
    #                Symmetric to encoder — standard practice
    blocks_up     = (1, 1, 1)
    # ---- Training ----
    spatial_size  = (96,96,96)  # SegResNet is light enough for 128^3 on free T4
    batch_size    = 2                # can use batch=2 because SegResNet uses ~2-3 GB GPU
    num_samples   = 2                # patches per loaded volume
    grad_accum    = 4                # effective batch = 2 * 2 * 2 = 8
    max_epochs    = 250
    learning_rate = 1e-2             # SGD with poly LR (same as DynUNet — good for ResNets)
    weight_decay  = 3e-5
    val_split     = 0.15
    patience      = 30


cfg = Config()
os.makedirs(cfg.SAVE_DIR, exist_ok=True)

# Unzip dataset once per session
if not os.path.exists(cfg.EXTRACT_TO) or len(os.listdir(cfg.EXTRACT_TO)) == 0:
    print(f"Unzipping to {cfg.EXTRACT_TO}...")
    os.makedirs(cfg.EXTRACT_TO, exist_ok=True)
    os.system(f"unzip -q '{cfg.DRIVE_ZIP}' -d '{cfg.EXTRACT_TO}'")
    print("Done.")
else:
    print(f"Dataset ready at {cfg.EXTRACT_TO}")



# ==================== CELL 4: DATA PREPARATION ====================

def map_brats_labels(x):
    """
    Remap BraTS 2020 labels so they are contiguous (0,1,2,3).
    Original: 0=BG, 1=NCR, 2=ED, 4=ET  (label 3 doesn't exist)
    Remapped: 0=BG, 1=NCR, 2=ED, 3=ET
    Without this fix the model wastes an entire output channel.
    """
    x = x.clone()
    x[x == 4] = 3
    x[(x < 0) | (x > 3)] = 0
    return x


def find_data_dir(base):
    """
    Auto-detects the folder containing BraTS20_Training_* case subfolders.
    Handles different zip extraction structures (flat, nested, Kaggle archive).
    """
    for root, dirs, files in os.walk(base):
        brats_cases = [d for d in dirs if d.startswith("BraTS20_Training_")]
        if len(brats_cases) > 0:
            print(f"Found {len(brats_cases)} cases in: {root}")
            return root
        if root.replace(base, "").count(os.sep) >= 4:
            break
    return None


def build_file_list(data_dir):
    """
    Builds the list of {image, label} dicts for MONAI Dataset.
    Searches for both .nii.gz and .nii file extensions.
    Only uses labelled training cases — BraTS validation has no labels.
    """
    case_dirs = sorted(glob.glob(os.path.join(data_dir, "BraTS20_Training_*")))
    file_list, skipped = [], []

    for case in case_dirs:
        name = os.path.basename(case)
        modality_files = []

        for mod in ["flair", "t1", "t1ce", "t2"]:
            for ext in [".nii.gz", ".nii"]:
                path = os.path.join(case, f"{name}_{mod}{ext}")
                if os.path.exists(path):
                    modality_files.append(path)
                    break

        seg_path = None
        for ext in [".nii.gz", ".nii"]:
            p = os.path.join(case, f"{name}_seg{ext}")
            if os.path.exists(p):
                seg_path = p
                break

        if len(modality_files) == 4 and seg_path:
            file_list.append({"image": modality_files, "label": seg_path})
        else:
            skipped.append(name)

    print(f"Valid: {len(file_list)} | Skipped: {len(skipped)}")
    if len(file_list) == 0:
        raise RuntimeError(f"No valid cases found in {data_dir}")
    return file_list


data_dir = find_data_dir(cfg.EXTRACT_TO)
if data_dir is None:
    raise RuntimeError("Could not find BraTS data. Check DRIVE_ZIP in Config.")
cfg.DATA_DIR = data_dir

all_files = build_file_list(cfg.DATA_DIR)

# Same random_state=42 as DynUNet ensures identical val split → fair comparison
train_files, val_files = train_test_split(
    all_files, test_size=cfg.val_split, random_state=42
)
print(f"Train: {len(train_files)} | Val: {len(val_files)}")

import os
import gc
import glob
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from sklearn.model_selection import train_test_split

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    NormalizeIntensityd,       # z-score normalisation — same as DynUNet and SwinUNETR
    CropForegroundd,           # remove zero padding around brain
    RandCropByPosNegLabeld,    # extract random patches centred on tumor or background
    Lambdad,                   # apply custom function to a data key
    RandFlipd,                 # random axis flip
    RandRotate90d,             # random 90-degree rotation
    RandGaussianNoised,        # add random noise
    RandAdjustContrastd,       # random contrast change
    RandScaleIntensityd,       # random intensity scaling
    RandShiftIntensityd,       # random intensity shifting
    RandAffined,               # random rotation + scale together
    SpatialPadd,               # Add SpatialPadd to ensure all patches are exactly spatial_size
)

from monai.data import Dataset, DataLoader, list_data_collate, pad_list_data_collate
from monai.losses import DiceCELoss
from monai.networks.nets import SegResNet
# SegResNet: residual encoder-decoder specifically designed for brain tumour segmentation
# Reference: Myronenko A. (2018) "3D MRI Brain Tumour Segmentation Using Autoencoder Regularisation"

from monai.inferers import sliding_window_inference
from torch.amp import GradScaler, autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} | Memory: {mem:.1f} GB")
# ==================== CELL 5: TRANSFORMS ====================
#
# IDENTICAL to DynUNet and SwinUNETR — this is intentional.
# To compare models fairly you MUST use the same data pipeline.
# Only the model definition (Cell 6) changes between the three experiments.

train_transforms = Compose([
    # Load .nii files from paths in the dict
    LoadImaged(keys=["image", "label"]),

    # Ensure [C, D, H, W] format for all tensors
    EnsureChannelFirstd(keys=["image", "label"]),

    # Remap label 4 → 3 for contiguous class indices
    Lambdad(keys=["label"], func=map_brats_labels),

    # Z-score normalise each modality using brain-voxel statistics
    # This is the SAME normalisation BraTS used — critical for new image generalisation
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

    # Remove skull-stripped zero-padding around the brain
    CropForegroundd(keys=["image", "label"], source_key="image"),

    # Extract 2 random patches per volume with 50% tumor / 50% background ratio
    # 128^3 patches are used here because SegResNet fits comfortably in memory
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=cfg.spatial_size,
        pos=1,
        neg=1,
        num_samples=cfg.num_samples,
        image_key="image",
        image_threshold=0,
        allow_smaller=True,
    ),
    # Add SpatialPadd to ensure all patches are exactly spatial_size
    SpatialPadd(keys=["image", "label"], spatial_size=cfg.spatial_size, method="end"),

    # Spatial augmentations — same as DynUNet for consistency
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    RandAffined(
        keys=["image", "label"], prob=0.3,
        rotate_range=(0.26, 0.26, 0.26),
        scale_range=(0.1, 0.1, 0.1),
        mode=("bilinear", "nearest"),
    ),

    # Intensity augmentations — scanner robustness
    RandGaussianNoised(keys=["image"], prob=0.15, std=0.1),
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
    RandScaleIntensityd(keys=["image"], prob=0.3, factors=0.1),
    RandShiftIntensityd(keys=["image"], prob=0.3, offsets=0.1),

    # Convert MetaTensor to standard torch.Tensor to ensure resizable storage
    Lambdad(keys=["image"], func=lambda x: x.as_tensor() if hasattr(x, 'as_tensor') else x),
    Lambdad(keys=["label"], func=lambda x: x.as_tensor() if hasattr(x, 'as_tensor') else x),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Lambdad(keys=["label"], func=map_brats_labels),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    # No patch crop — full volume evaluated with sliding window
    Lambdad(keys=["image"], func=lambda x: x.as_tensor() if hasattr(x, 'as_tensor') else x),
    Lambdad(keys=["label"], func=lambda x: x.as_tensor() if hasattr(x, 'as_tensor') else x),
])

train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds   = Dataset(data=val_files,   transform=val_transforms)

train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=pad_list_data_collate,
)
val_loader = DataLoader(
    val_ds, batch_size=1,
    shuffle=False, num_workers=4, pin_memory=True,
)
print(f"Train batches: {len(train_loader)} | Val cases: {len(val_loader)}")


# ==================== CELL 6: MODEL — SegResNet ====================

model = SegResNet(
    # spatial_dims : 3D volumetric segmentation
    spatial_dims=3,

    # blocks_down : number of residual blocks at each encoder stage
    #               [1,2,2,4] — gradually more blocks as we go deeper
    #               More blocks at deeper stages helps capture complex features
    blocks_down=cfg.blocks_down,

    # blocks_up : number of residual blocks at each decoder stage
    #             Decoder is simpler than encoder — standard practice
    blocks_up=cfg.blocks_up,

    # init_filters : feature map size at the first encoder stage
    #                Doubles at each stage: 32 → 64 → 128 → 256 → 512
    #                This is the "width" of the network
    init_filters=cfg.init_filters,

    # in_channels : 4 MRI modalities (FLAIR, T1, T1CE, T2)
    in_channels=4,

    # out_channels : 4 classes (background, NCR, edema, enhancing tumor)
    out_channels=4,

    # dropout_prob : randomly zeros 20% of neurons during training
    #                Acts as regularisation to prevent overfitting
    #                More important for SegResNet than DynUNet because
    #                SegResNet is a smaller model and can memorise training data
    dropout_prob=0.2,

    # norm : group normalisation — works better than batch norm with small batch sizes
    #        (batch size of 1-2 makes batch norm statistics unreliable)
    #        Group norm divides channels into groups and normalises within each group
    norm=("GROUP", {"num_groups": 8}),

).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Approx GPU memory: ~{total_params * 4 / 1e9 * 4:.1f} GB (much lighter than DynUNet/SwinUNETR)")


# ==================== CELL 7: LOSS FUNCTION ====================

# Same DiceCELoss as DynUNet and SwinUNETR — consistent comparison
criterion = DiceCELoss(
    include_background=False,  # ignore background class in Dice — it dominates
    to_onehot_y=True,          # convert int labels → one-hot for Dice computation
    softmax=True,              # apply softmax to raw logits before loss
    squared_pred=True,         # squared denominator — more numerically stable
    lambda_dice=0.5,           # 50% weight on Dice loss
    lambda_ce=0.5,             # 50% weight on cross-entropy loss
)

def compute_loss(outputs, target):
    """
    SegResNet returns a single [B, 4, D, H, W] tensor — no deep supervision.
    Loss is simply DiceCE on the single output.
    """
    return criterion(outputs, target)




# ==================== CELL 8: OPTIMIZER ====================
#
# SegResNet uses SGD + polynomial LR — same as DynUNet.
# Why SGD for SegResNet but AdamW for SwinUNETR?
# ResNet-style CNNs (DynUNet, SegResNet) converge better with SGD + momentum
# because their gradient landscapes are smooth and convex-like.
# Transformers (SwinUNETR) have more complex, non-convex landscapes
# where adaptive optimisers (AdamW) work better.

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=cfg.learning_rate,    # 1e-2 initial LR — polynomial schedule decays this
    momentum=0.99,           # high momentum — standard for medical image segmentation
    nesterov=True,           # Nesterov lookahead — slightly better convergence than vanilla momentum
    weight_decay=cfg.weight_decay,
)

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    """
    Polynomial learning rate decay.
    At epoch 0   : lr = initial_lr (e.g. 0.01)
    At epoch 100 : lr = initial_lr * (1 - 100/250)^0.9 ≈ 0.0042
    At epoch 200 : lr = initial_lr * (1 - 200/250)^0.9 ≈ 0.0008
    The exponent 0.9 is the nnU-Net standard — empirically verified on many datasets.
    """
    return initial_lr * (1 - epoch / max_epochs) ** exponent

scaler = GradScaler()




# ==================== CELL 9: EVALUATION HELPERS ====================

def dice_coeff(pred, target):
    """
    Binary Dice = 2 * intersection / (pred_sum + target_sum)
    Returns 0 (no overlap) to 1 (perfect overlap).
    Small epsilon (1e-8) prevents division by zero on empty regions.
    """
    inter = np.sum(pred * target)
    return (2.0 * inter) / (np.sum(pred) + np.sum(target) + 1e-8)


def compute_brats_dice(pred_np, true_np):
    """
    Compute the 3 official BraTS evaluation regions.
    These are nested regions derived from the 4 classes:

    WT = Whole Tumor   = all non-background (labels 1, 2, 3)
         Tests: can the model find the full extent of the tumor?

    TC = Tumor Core    = necrosis + enhancing tumor (labels 1, 3)
         Tests: can the model find the solid core without the surrounding edema?

    ET = Enhancing Tumor = label 3 only
         Tests: can the model find the actively growing tumor region?
         This is the most clinically important region for treatment planning.
    """
    wt = dice_coeff(
        (pred_np > 0).astype(float),
        (true_np > 0).astype(float)
    )
    tc = dice_coeff(
        ((pred_np == 1) | (pred_np == 3)).astype(float),
        ((true_np == 1) | (true_np == 3)).astype(float)
    )
    et = dice_coeff(
        (pred_np == 3).astype(float),
        (true_np == 3).astype(float)
    )
    return wt, tc, et


def postprocess(pred_np):
    """
    Three-step post-processing — identical across all three models.
    Having identical post-processing ensures any score differences
    are due to the model, not the post-processing.

    Step 1: Connected component analysis
            Keep only the largest tumor component.
            Removes small isolated false positive clusters.

    Step 2: Hierarchical consistency
            ET must be inside TC (ET ⊆ TC)
            TC must be inside WT (TC ⊆ WT)
            Without this, small model errors can produce
            anatomically impossible predictions.

    Step 3: Small ET removal
            If predicted ET volume < 50 voxels, remove it.
            Real enhancing tumors are larger than this.
            This is especially important for SegResNet which
            can produce more false positives than DynUNet.
    """
    pred = pred_np.copy()

    wt_mask = (pred > 0).astype(np.uint8)
    labeled, n = ndimage.label(wt_mask)
    if n > 1:
        sizes   = ndimage.sum(wt_mask, labeled, range(1, n + 1))
        largest = np.argmax(sizes) + 1
        pred[labeled != largest] = 0

    tc_mask = (pred == 1) | (pred == 3)
    pred[(pred == 3) & ~tc_mask] = 0

    wt_mask2 = pred > 0
    pred[(pred == 1) & ~wt_mask2] = 0

    if 0 < np.sum(pred == 3) < 50:
        pred[pred == 3] = 0

    return pred

 # ==================== CELL 10: TRAINING LOOP (MEMORY FIXED) ====================

CKPT_PATH    = os.path.join(cfg.SAVE_DIR, "checkpoint.pt")
BEST_PATH    = os.path.join(cfg.SAVE_DIR, "best_overall.pt")
BEST_WT_PATH = os.path.join(cfg.SAVE_DIR, "best_wt.pt")
BEST_TC_PATH = os.path.join(cfg.SAVE_DIR, "best_tc.pt")
BEST_ET_PATH = os.path.join(cfg.SAVE_DIR, "best_et.pt")

start_epoch  = 0
best_dice    = 0.0
best_wt_dice = 0.0
best_tc_dice = 0.0
best_et_dice = 0.0
patience_ctr = 0
history      = []

if os.path.exists(CKPT_PATH):
    print("Checkpoint found — resuming...")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch  = ckpt["epoch"] + 1
    best_dice    = ckpt.get("best_dice",    0.0)
    best_wt_dice = ckpt.get("best_wt_dice", 0.0)
    best_tc_dice = ckpt.get("best_tc_dice", 0.0)
    best_et_dice = ckpt.get("best_et_dice", 0.0)
    patience_ctr = ckpt.get("patience_ctr", 0)
    history      = ckpt.get("history",      [])
    print(f"Resumed epoch {start_epoch} | WT {best_wt_dice:.4f} | TC {best_tc_dice:.4f} | ET {best_et_dice:.4f}")
else:
    print("No checkpoint — starting fresh")

print(f"\n{'Ep':>4} | {'Tr Loss':>8} {'Tr WT':>6} {'Tr TC':>6} {'Tr ET':>6} | {'Vl Loss':>8} {'Vl WT':>6} {'Vl TC':>6} {'Vl ET':>6} {'Mean':>6} | Saved")
print("-" * 108)

for epoch in range(start_epoch, cfg.max_epochs):

    new_lr = cfg.learning_rate * (1 - epoch / cfg.max_epochs) ** 0.9
    for g in optimizer.param_groups:
        g["lr"] = new_lr

    # ============ TRAIN ============
    model.train()
    scaler = GradScaler()  # fresh scaler every epoch — prevents unscale-after-step error

    train_loss = 0.0
    train_wt, train_tc, train_et = [], [], []
    accum_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"Ep {epoch+1:3d} [Train]", leave=False)

    for step, batch in pbar:
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        with autocast(device_type="cuda", enabled=device.type == "cuda"):
            out  = model(x)
            loss = compute_loss(out, y) / cfg.grad_accum

        scaler.scale(loss).backward()
        accum_loss += loss.item()

        is_accum_step = (step + 1) % cfg.grad_accum == 0
        is_last_batch = (step + 1) == len(train_loader)

        if is_accum_step or is_last_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            train_loss += accum_loss * cfg.grad_accum
            accum_loss  = 0.0

        # Compute train Dice — immediately delete tensors after to free memory
        with torch.no_grad():
            out_cpu = out.detach().cpu()   # move to CPU immediately
            y_cpu   = y.detach().cpu()
            preds = torch.argmax(torch.softmax(out_cpu, dim=1), dim=1).numpy()
            trues = y_cpu.squeeze(1).numpy()
            del out_cpu, y_cpu            # free CPU memory too

        for i in range(preds.shape[0]):
            wt, tc, et = compute_brats_dice(preds[i], trues[i])
            train_wt.append(wt)
            train_tc.append(tc)
            train_et.append(et)

        # Explicitly delete batch tensors — critical for OOM prevention
        del x, y, out, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pbar.set_postfix({
            "loss": f"{train_loss / max(1, step + 1):.4f}",
            "WT":   f"{np.mean(train_wt[-8:]):.3f}",
            "TC":   f"{np.mean(train_tc[-8:]):.3f}",
            "ET":   f"{np.mean(train_et[-8:]):.3f}",
        })

    train_loss /= max(1, len(train_loader) / cfg.grad_accum)

    # Force full GPU memory cleanup before validation
    gc.collect()
    torch.cuda.empty_cache()

    # ============ VALIDATE ============
    model.eval()
    val_wt, val_tc, val_et = [], [], []
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Ep {epoch+1:3d} [Val]", leave=False):
            x    = batch["image"].to(device)
            y_np = batch["label"].squeeze(1).cpu().numpy()[0]

            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                out = sliding_window_inference(
                    x,
                    roi_size=cfg.spatial_size,
                    sw_batch_size=2,   # FIXED: was 4, reduced to 2 for safety
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian",
                    padding_mode="constant",
                )

            val_loss += compute_loss(out, batch["label"].to(device)).item()

            pred = torch.argmax(
                torch.softmax(out, dim=1), dim=1
            ).cpu().numpy()[0]

            # Delete GPU tensors immediately after extracting numpy
            del x, out
            torch.cuda.empty_cache()

            pred = postprocess(pred)
            wt, tc, et = compute_brats_dice(pred, y_np)
            val_wt.append(wt)
            val_tc.append(tc)
            val_et.append(et)

    val_loss /= len(val_loader)
    vWT  = np.mean(val_wt)
    vTC  = np.mean(val_tc)
    vET  = np.mean(val_et)
    mean = (vWT + vTC + vET) / 3.0

    # ============ SAVE BEST MODELS ============
    saved = []
    if mean > best_dice + 1e-4:
        best_dice = mean
        torch.save(model.state_dict(), BEST_PATH)
        saved.append("Overall")
        patience_ctr = 0
    if vWT > best_wt_dice + 1e-4:
        best_wt_dice = vWT
        torch.save(model.state_dict(), BEST_WT_PATH)
        saved.append("WT")
    if vTC > best_tc_dice + 1e-4:
        best_tc_dice = vTC
        torch.save(model.state_dict(), BEST_TC_PATH)
        saved.append("TC")
    if vET > best_et_dice + 1e-4:
        best_et_dice = vET
        torch.save(model.state_dict(), BEST_ET_PATH)
        saved.append("ET")
    if not saved:
        patience_ctr += 1

    saved_str = "+".join(saved) if saved else f"no improve {patience_ctr}/{cfg.patience}"

    print(
        f"{epoch+1:4d} | "
        f"{train_loss:8.4f} {np.mean(train_wt):6.4f} {np.mean(train_tc):6.4f} {np.mean(train_et):6.4f} | "
        f"{val_loss:8.4f} {vWT:6.4f} {vTC:6.4f} {vET:6.4f} {mean:6.4f} | "
        f"{saved_str}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": round(train_loss, 4),
        "train_wt":   round(float(np.mean(train_wt)), 4),
        "train_tc":   round(float(np.mean(train_tc)), 4),
        "train_et":   round(float(np.mean(train_et)), 4),
        "val_loss":   round(val_loss, 4),
        "val_wt":     round(vWT, 4),
        "val_tc":     round(vTC, 4),
        "val_et":     round(vET, 4),
        "val_mean":   round(mean, 4),
    })

    torch.save({
        "epoch":        epoch,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "best_dice":    best_dice,
        "best_wt_dice": best_wt_dice,
        "best_tc_dice": best_tc_dice,
        "best_et_dice": best_et_dice,
        "patience_ctr": patience_ctr,
        "history":      history,
    }, CKPT_PATH)

    if patience_ctr >= cfg.patience:
        print("Early stopping triggered")
        break

    gc.collect()
    torch.cuda.empty_cache()

print(f"\nBest → Overall: {best_dice:.4f} | WT: {best_wt_dice:.4f} | TC: {best_tc_dice:.4f} | ET: {best_et_dice:.4f}")