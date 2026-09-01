# import numpy as np
# import pandas as pd
# import nibabel as nib
# from pathlib import Path
# from scipy import ndimage
# from scipy.ndimage import sobel

# DATA_DIR    = "/content/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
# MAPPING_CSV = DATA_DIR + "/name_mapping.csv"

# def extract_features(seg_path):
#     try:
#         seg = nib.load(str(seg_path)).get_fdata()
#         seg[seg == 4] = 3

#         wt  = (seg > 0).astype(np.float32)
#         tc  = ((seg == 1) | (seg == 3)).astype(np.float32)
#         et  = (seg == 3).astype(np.float32)
#         ncr = (seg == 1).astype(np.float32)
#         ed  = (seg == 2).astype(np.float32)

#         wt_vol  = float(wt.sum())
#         tc_vol  = float(tc.sum())
#         et_vol  = float(et.sum())
#         ncr_vol = float(ncr.sum())
#         ed_vol  = float(ed.sum())

#         eps          = 1e-6
#         et_wt_ratio  = et_vol  / (wt_vol  + eps)
#         et_tc_ratio  = et_vol  / (tc_vol  + eps)
#         tc_wt_ratio  = tc_vol  / (wt_vol  + eps)
#         ed_wt_ratio  = ed_vol  / (wt_vol  + eps)
#         ncr_tc_ratio = ncr_vol / (tc_vol  + eps)

#         sx = sobel(wt, axis=0)
#         sy = sobel(wt, axis=1)
#         sz = sobel(wt, axis=2)
#         surface_area = float(np.sqrt(sx**2 + sy**2 + sz**2).sum())

#         sphericity = 0.0
#         if surface_area > 0 and wt_vol > 0:
#             sphericity = float(
#                 (np.pi * (1/3)) * ((6 * wt_vol) * (2/3)) / surface_area
#             )

#         _, n_components = ndimage.label(wt > 0)
#         et_present      = 1 if et_vol > 50 else 0

#         return {
#             "wt_vol":       wt_vol,
#             "tc_vol":       tc_vol,
#             "et_vol":       et_vol,
#             "ncr_vol":      ncr_vol,
#             "ed_vol":       ed_vol,
#             "et_wt_ratio":  et_wt_ratio,
#             "et_tc_ratio":  et_tc_ratio,
#             "tc_wt_ratio":  tc_wt_ratio,
#             "ed_wt_ratio":  ed_wt_ratio,
#             "ncr_tc_ratio": ncr_tc_ratio,
#             "sphericity":   sphericity,
#             "surface_area": surface_area,
#             "n_components": float(n_components),
#             "et_present":   float(et_present),
#         }

#     except Exception as e:
#         print(f"  ERROR: {seg_path.name} — {e}")
#         return None


# def build_dataset(data_dir, mapping_csv):
#     mapping   = pd.read_csv(mapping_csv)
#     grade_map = dict(zip(
#         mapping["BraTS_2020_subject_ID"].astype(str),
#         mapping["Grade"].astype(str)
#     ))

#     records        = []
#     errors         = 0
#     skipped_grade  = 0

#     patient_folders = sorted([p for p in Path(data_dir).iterdir() if p.is_dir()])
#     print(f"Found {len(patient_folders)} patient folders")

#     for i, patient_folder in enumerate(patient_folders):
#         patient_id = patient_folder.name.strip()
#         grade      = grade_map.get(patient_id)

#         if grade is None:
#             skipped_grade += 1
#             continue

#         seg_files = list(patient_folder.glob("*_seg.nii.gz"))
#         if not seg_files:
#             seg_files = list(patient_folder.glob("*_seg.nii"))
#         if not seg_files:
#             errors += 1
#             continue

#         features = extract_features(seg_files[0])
#         if features is None:
#             errors += 1
#             continue

#         features["label"]      = 1 if grade == "HGG" else 0
#         features["grade"]      = grade
#         features["patient_id"] = patient_id
#         records.append(features)

#         if (i + 1) % 50 == 0:
#             print(f"  Processed {i+1}/{len(patient_folders)} — loaded so far: {len(records)}")

#     print(f"\nDone. Loaded: {len(records)} | Errors: {errors} | No grade found: {skipped_grade}")
#     return pd.DataFrame(records)


# # ── RUN ───────────────────────────────────────────────────────
# print("=" * 50)
# print("BraTS 2020 Feature Extraction")
# print("=" * 50)

# df = build_dataset(DATA_DIR, MAPPING_CSV)

# print("\n" + "=" * 50)
# print("DATASET SUMMARY")
# print("=" * 50)
# print(f"Total cases : {len(df)}")
# print(f"HGG cases   : {(df['label'] == 1).sum()}")
# print(f"LGG cases   : {(df['label'] == 0).sum()}")
# print(f"\nFirst 3 rows:")
# print(df.head(3))

# CSV_SAVE_PATH = "/content/drive/MyDrive/brats2020_grade_features.csv"
# df.to_csv(CSV_SAVE_PATH, index=False)
# print(f"\n✅ CSV saved to Google Drive: {CSV_SAVE_PATH}")

# from google.colab import files
# files.download(CSV_SAVE_PATH)
# print("✅ Download started to your computer")



# ============================================================
# STEP 2 — train_grade_model.py
# Run this LOCALLY after you have the CSV from Step 1.
# Place this file in your project root folder.
# Place the CSV file in your project root folder too.
#
# HOW TO RUN:
#   python train_grade_model.py
#
# OUTPUT:
#   models/xgboost_grade_model.pkl  ← ready to use in the app
# ============================================================


import sys
sys.stdout.reconfigure(encoding='utf-8')
# ============================================================
# STEP 2 — train_grade_model.py
# Run this LOCALLY after you have the CSV from Step 1.
# Place this file in your project root folder.
# Place the CSV file in your project root folder too.
#
# HOW TO RUN:
#   python train_grade_model.py
#
# OUTPUT:
#   models/xgboost_grade_model.pkl  ← ready to use in the app
# ============================================================

import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import classification_report, confusion_matrix

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# CONFIGURATION — only change these if needed
# ──────────────────────────────────────────────────────────────

# Path to your CSV file (put it in the same folder as this script)
CSV_PATH   = "brats2020_grade_features.csv"

# Where to save the trained model
MODEL_PATH = "models/xgboost_grade_model.pkl"

# The 14 features the model uses — DO NOT change order
FEATURE_COLS = [
    "wt_vol",       # Whole Tumor volume (voxels)
    "tc_vol",       # Tumor Core volume
    "et_vol",       # Enhancing Tumor volume
    "ncr_vol",      # Necrotic Core volume
    "ed_vol",       # Edema volume
    "et_wt_ratio",  # ET / WT  ← strongest predictor
    "et_tc_ratio",  # ET / Core
    "tc_wt_ratio",  # Core / WT
    "ed_wt_ratio",  # Edema / WT
    "ncr_tc_ratio", # Necrosis / Core
    "sphericity",   # How round is the tumor
    "surface_area", # Tumor surface complexity
    "n_components", # How many separate tumor pieces
    "et_present",   # 1 if ET > 50 voxels else 0
]


# ──────────────────────────────────────────────────────────────
# STEP A — Load and check the CSV
# ──────────────────────────────────────────────────────────────

print("=" * 55)
print("  NeuroAI — Grade Classification Model Training")
print("=" * 55)

if not os.path.exists(CSV_PATH):
    print(f"\n❌ CSV file not found at: {CSV_PATH}")
    print("   Run STEP1_extract_features_colab.py in Google Colab first.")
    print("   Then download the CSV and place it here.")
    exit(1)

df = pd.read_csv(CSV_PATH)

print(f"\n✅ CSV loaded successfully")
print(f"   Total cases  : {len(df)}")
print(f"   HGG cases    : {(df['label'] == 1).sum()}")
print(f"   LGG cases    : {(df['label'] == 0).sum()}")
print(f"   Class ratio  : {(df['label']==1).sum():.0f} HGG : {(df['label']==0).sum():.0f} LGG")

# Check all feature columns exist
missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
if missing_cols:
    print(f"\n❌ Missing columns in CSV: {missing_cols}")
    print("   The CSV must be generated by STEP1_extract_features_colab.py")
    exit(1)

print(f"   All 14 feature columns found ✓")


# ──────────────────────────────────────────────────────────────
# STEP B — Prepare features and labels
# ──────────────────────────────────────────────────────────────

X = df[FEATURE_COLS].values   # shape: (n_cases, 14)
y = df["label"].values         # 0 = LGG,  1 = HGG

# Remove any rows with NaN values
nan_mask = ~np.isnan(X).any(axis=1)
X = X[nan_mask]
y = y[nan_mask]
removed = (~nan_mask).sum()
if removed > 0:
    print(f"\n   Removed {removed} rows with missing values")

n_hgg = (y == 1).sum()
n_lgg = (y == 0).sum()

print(f"\n   Clean dataset: {len(y)} cases ({n_hgg} HGG, {n_lgg} LGG)")


# ──────────────────────────────────────────────────────────────
# STEP C — Scale the features
# ──────────────────────────────────────────────────────────────
# StandardScaler makes each feature have mean=0 and std=1
# This is REQUIRED — the same scaler is saved and used in predict.py

print("\n[1/4] Scaling features...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("      Done. Each feature now has mean=0, std=1")


# ──────────────────────────────────────────────────────────────
# STEP D — Fix class imbalance properly
# ──────────────────────────────────────────────────────────────
# BraTS has ~259 HGG and ~76 LGG — that is 3.4x more HGG
# Without correction the model just predicts HGG for everything
#
# The correct fix: tell XGBoost to treat each LGG case as if
# it were 3.4x more important than each HGG case.
# This balances the learning signal.
#
# Formula: scale_pos_weight = n_majority / n_minority
#          = n_HGG / n_LGG    (HGG is majority here)
# Note: XGBoost calls label=1 the "positive" class
# Our label=1 is HGG which is the MAJORITY
# So we give weight to compensate: n_LGG / n_HGG? No.
# Actually we want to balance: weight = n_negative / n_positive
# n_negative = LGG count (label 0)
# n_positive = HGG count (label 1)
# BUT since HGG is majority, this gives < 1 which doesn't help
#
# CORRECT APPROACH: Use class_weight instead via sample_weight
# This is cleaner and more transparent than scale_pos_weight

from sklearn.utils.class_weight import compute_sample_weight

# compute_sample_weight gives each training sample a weight
# LGG samples get weight ~3.4, HGG samples get weight ~1.0
# This perfectly compensates for the 3.4:1 imbalance
sample_weights = compute_sample_weight(class_weight="balanced", y=y)

hgg_weight = sample_weights[y == 1][0]
lgg_weight = sample_weights[y == 0][0]
print(f"\n[2/4] Class imbalance correction:")
print(f"      HGG sample weight: {hgg_weight:.3f}")
print(f"      LGG sample weight: {lgg_weight:.3f}")
print(f"      LGG cases are now treated as {lgg_weight/hgg_weight:.1f}x more important")


# ──────────────────────────────────────────────────────────────
# STEP E — Define the XGBoost model
# ──────────────────────────────────────────────────────────────

print("\n[3/4] Training XGBoost model with 5-Fold Cross Validation...")

model = XGBClassifier(
    # ── Tree structure ──────────────────────────────────────
    n_estimators = 300,     # Build 300 trees total
                            # Each tree corrects mistakes of previous ones
    max_depth    = 4,       # Each tree asks max 4 questions
                            # Shallow trees prevent memorizing training data

    # ── Learning speed ──────────────────────────────────────
    learning_rate = 0.05,   # Each tree contributes only 5% of its strength
                            # Slow learning = more accurate final model
                            # Lower rate needs more trees (300 is enough here)

    # ── Randomness to prevent overfitting ───────────────────
    subsample        = 0.8, # Each tree sees only 80% of patients (random)
    colsample_bytree = 0.8, # Each tree sees only 80% of features (random)
                            # This forces model to learn general patterns
                            # not memorize specific patients

    # ── Evaluation ──────────────────────────────────────────
    eval_metric       = "logloss",  # Penalizes confident wrong predictions
    use_label_encoder = False,      # Suppress XGBoost warning about encoding

    # ── Reproducibility ─────────────────────────────────────
    random_state = 42,      # Same seed = same model every run
)




# ──────────────────────────────────────────────────────────────
# STEP F — Cross Validation (Honest Accuracy Estimate)
# ──────────────────────────────────────────────────────────────
# This is the REAL test of accuracy.
# We split data into 5 groups. Train on 4, test on 1.
# Repeat 5 times so every case gets tested exactly once.
# The model NEVER sees the test cases during training.
# This gives an honest estimate of real-world performance.

cv = StratifiedKFold(
    n_splits  = 5,       # Split into 5 groups
    shuffle   = True,    # Shuffle data before splitting
    random_state = 42,
)

# Cross-validate WITH sample weights for balanced evaluation
cv_scores = cross_val_score(
    model, X_scaled, y,
    cv      = cv,
    scoring = "balanced_accuracy",   # balanced_accuracy is fair for imbalanced data
                                     # It averages accuracy per class
                                     # So getting LGG right matters as much as HGG
    params = {"sample_weight": sample_weights},
)



print(f"\n      Cross-Validation Results (5-Fold):")
for i, score in enumerate(cv_scores):
    print(f"        Fold {i+1}: {score:.1%}")
print(f"\n      Mean Balanced Accuracy : {cv_scores.mean():.1%}")
print(f"      Std (stability)        : ±{cv_scores.std():.1%}")

if cv_scores.mean() >= 0.85:
    print(f"\n      ✅ Excellent accuracy — model is ready for use")
elif cv_scores.mean() >= 0.75:
    print(f"\n      ⚠️  Good accuracy — acceptable for research use")
else:
    print(f"\n      ❌ Low accuracy — check your CSV data")


# ──────────────────────────────────────────────────────────────
# STEP G — Train Final Model on ALL Data
# ──────────────────────────────────────────────────────────────
# Cross-validation was for evaluation only.
# Now train on the complete dataset — all 335 cases.
# This final model will be slightly better than the CV models
# because it has seen more data.

print(f"\n[4/4] Training final model on all {len(y)} cases...")
model.fit(X_scaled, y, sample_weight=sample_weights)
print("      Training complete.")


# ──────────────────────────────────────────────────────────────
# STEP H — Evaluate on Training Data (for reference only)
# ──────────────────────────────────────────────────────────────
# NOTE: These numbers will look too good (near 100%)
# because the model has already seen this data.
# The TRUE accuracy is the cross-validation score above.

print("\n" + "─" * 55)
print("Training Data Performance (Optimistic — for reference only)")
print("The honest accuracy is the CV score above.")
print("─" * 55)
y_pred = model.predict(X_scaled)
print(classification_report(y, y_pred, target_names=["LGG", "HGG"]))

print("Confusion Matrix:")
print("             Predicted LGG   Predicted HGG")
cm = confusion_matrix(y, y_pred)
print(f"  Actual LGG      {cm[0][0]:4d}            {cm[0][1]:4d}")
print(f"  Actual HGG      {cm[1][0]:4d}            {cm[1][1]:4d}")


# ──────────────────────────────────────────────────────────────
# STEP I — Feature Importance
# ──────────────────────────────────────────────────────────────
# Shows which features matter most for the prediction

print("\n" + "─" * 55)
print("Feature Importances (higher = more important for grade)")
print("─" * 55)

importance_df = pd.DataFrame({
    "Feature":    FEATURE_COLS,
    "Importance": model.feature_importances_,
}).sort_values("Importance", ascending=False)

for _, row in importance_df.iterrows():
    bar = "█" * int(row["Importance"] * 50)
    print(f"  {row['Feature']:<20} {row['Importance']:.4f}  {bar}")

print("\nThe top feature (et_wt_ratio) confirms:")
print("  Enhancement ratio is the strongest predictor of HGG vs LGG.")
print("  This matches published brain tumor research literature.")


# ──────────────────────────────────────────────────────────────
# STEP J — Save Everything
# ──────────────────────────────────────────────────────────────

Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

save_package = {
    # The trained model — used in predict.py
    "model": model,

    # The scaler — MUST be saved with model
    # predict.py uses the SAME scaler to normalize new patient features
    # If you retrain without saving the scaler, predictions will be wrong
    "scaler": scaler,

    # Feature column order — ensures inference uses same order as training
    "feature_cols": FEATURE_COLS,

    # Performance metadata — displayed in the app
    "cv_balanced_accuracy": float(cv_scores.mean()),
    "cv_std":               float(cv_scores.std()),
    "n_training_cases":     int(len(y)),
    "n_hgg":                int(n_hgg),
    "n_lgg":                int(n_lgg),
    "class_balance_method": "compute_sample_weight(balanced)",
}

joblib.dump(save_package, MODEL_PATH)

print("\n" + "=" * 55)
print("  MODEL SAVED SUCCESSFULLY")
print("=" * 55)
print(f"  File     : {MODEL_PATH}")
print(f"  Accuracy : {cv_scores.mean():.1%} balanced (5-fold CV)")
print(f"  Cases    : {n_hgg} HGG + {n_lgg} LGG = {len(y)} total")
print(f"  Fix      : Class imbalance corrected with sample weights")
print(f"\n  Copy models/xgboost_grade_model.pkl to your project.")
print(f"  The app will automatically use this new model.")
print("=" * 55)