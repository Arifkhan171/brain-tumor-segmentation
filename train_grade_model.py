# ============================================================
# !! VERSION v2.0 — UPDATED FILE WITH CROSS-VALIDATION !!
# If you see this comment at top → you have the CORRECT file.
# If you ran a version without CV output → you had the OLD file.
#
# THIS VERSION INCLUDES:
#   [1] 5-Fold Stratified Cross-Validation (for thesis reporting)
#   [2] Final model retrained on ALL 368 cases (better accuracy)
#   [3] Auto Colab detection (mounts Drive, unzips, trains, saves)
#   [4] All 5 bug fixes from original script
#
# HOW TO RUN IN COLAB:
#   1. Upload THIS file (train_grade_model.py) from your laptop to Colab
#   2. Check DRIVE_ZIP_PATH below matches your zip name in Drive
#   3. Run: !python train_grade_model.py
#   4. Download xgboost_grade_model.pkl from Drive → BraTS_DynUNet folder
#   5. Place in: auto tumor/models/xgboost_grade_model.pkl
# ============================================================
#
# LOCALLY (only if you download the zip to your laptop):
#   1. Set LOCAL_DATA_DIR below to your extracted BraTS folder
#   2. Run: python train_grade_model.py
# ============================================================

import os, sys, glob
import numpy as np
import pandas as pd
import nibabel as nib
import joblib
from scipy import ndimage
from scipy.ndimage import sobel
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# !! CONFIGURE PATHS HERE !!
# ============================================================

# --- COLAB: path to your zip file inside Google Drive ---
DRIVE_ZIP_PATH  = "/content/drive/MyDrive/Brats_data.zip"   # change if your zip has a different name
COLAB_EXTRACT   = "/content/brats2020"                       # fast local Colab disk
COLAB_SAVE_DIR  = "/content/drive/MyDrive/BraTS_DynUNet"    # results saved back to Drive

# --- LOCAL: path to already-extracted BraTS folder ---
# Example Windows: r"C:\Users\YourName\brats2020\MICCAI_BraTS2020_TrainingData"
# Example Linux:   "/home/yourname/brats2020/MICCAI_BraTS2020_TrainingData"
LOCAL_DATA_DIR  = r"C:\BraTS2020\MICCAI_BraTS2020_TrainingData"   # change this if running locally


# ============================================================
# AUTO-DETECT ENVIRONMENT
# ============================================================

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

IN_COLAB = is_colab()
print(f"Environment: {'Google Colab' if IN_COLAB else 'Local (VS Code / Terminal)'}")


# ============================================================
# COLAB SETUP — mount drive + unzip
# ============================================================

if IN_COLAB:
    from google.colab import drive
    print("Mounting Google Drive...")
    drive.mount('/content/drive')

    # Unzip BraTS data to fast local Colab disk (only once per session)
    if not os.path.exists(COLAB_EXTRACT) or len(os.listdir(COLAB_EXTRACT)) == 0:
        print(f"\nUnzipping {DRIVE_ZIP_PATH} → {COLAB_EXTRACT}")
        print("This takes ~10-15 min the first time...")
        os.makedirs(COLAB_EXTRACT, exist_ok=True)
        ret = os.system(f"unzip -q '{DRIVE_ZIP_PATH}' -d '{COLAB_EXTRACT}'")
        if ret != 0:
            print("ERROR: unzip failed. Check DRIVE_ZIP_PATH above.")
            sys.exit(1)
        print("Unzip complete.")
    else:
        print(f"BraTS data already extracted at {COLAB_EXTRACT}")

    os.makedirs(COLAB_SAVE_DIR, exist_ok=True)
    OUTPUT_DIR = COLAB_SAVE_DIR

else:
    # Running locally — use LOCAL_DATA_DIR directly
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# AUTO-FIND BraTS TRAINING FOLDER
# (handles different zip structures from different downloads)
# ============================================================

def find_brats_dir(base):
    """Walk up to 4 levels deep to find the folder containing BraTS20_Training_* cases."""
    for root, dirs, _ in os.walk(base):
        if any(d.startswith("BraTS20_Training_") for d in dirs):
            count = len([d for d in dirs if d.startswith("BraTS20_Training_")])
            print(f"Found {count} training cases in: {root}")
            return root
        if root.replace(base, "").count(os.sep) >= 4:
            break
    return None

if IN_COLAB:
    DATA_DIR = find_brats_dir(COLAB_EXTRACT)
    if DATA_DIR is None:
        print(f"ERROR: Could not find BraTS20_Training_* folders in {COLAB_EXTRACT}")
        print("Check your zip file structure or set DATA_DIR manually.")
        sys.exit(1)
else:
    DATA_DIR = LOCAL_DATA_DIR
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: LOCAL_DATA_DIR does not exist: {DATA_DIR}")
        print("Set LOCAL_DATA_DIR at the top of this file.")
        sys.exit(1)

print(f"Using DATA_DIR: {DATA_DIR}")


# ============================================================
# STEP 1 — FIND GRADE CSV
# BraTS 2020 CSVs:
#   name_mapping.csv  → has 'Grade' column (HGG/LGG)  ← what we need
#   survival_info.csv → has Age, Survival_days         ← no Grade here
# ============================================================

def find_grade_csv(data_dir):
    candidates = [
        os.path.join(data_dir, "name_mapping.csv"),
        os.path.join(data_dir, "survival_info.csv"),
        os.path.join(os.path.dirname(data_dir), "name_mapping.csv"),
        os.path.join(os.path.dirname(data_dir), "survival_info.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            grade_cols = [c for c in df.columns if 'grade' in c.lower()]
            if grade_cols:
                print(f"Found grade CSV: {path}")
                print(f"  Columns: {df.columns.tolist()}")
                return path, grade_cols[0]

    # Broad search as fallback
    all_csvs = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    for path in all_csvs:
        df = pd.read_csv(path)
        grade_cols = [c for c in df.columns if 'grade' in c.lower()]
        if grade_cols:
            print(f"Found grade CSV: {path}")
            return path, grade_cols[0]

    return None, None


csv_path, grade_col = find_grade_csv(DATA_DIR)
if csv_path is None:
    print("ERROR: Could not find a CSV with a 'Grade' column in DATA_DIR.")
    print("Expected file: name_mapping.csv  with column 'Grade'")
    sys.exit(1)

df_labels = pd.read_csv(csv_path)
print(f"\nGrade column: '{grade_col}'")
print(df_labels[grade_col].value_counts())


# ============================================================
# STEP 2 — EXTRACT FEATURES FROM SEGMENTATION MASKS
# ============================================================

def extract_features(seg_path, case_id):
    try:
        seg_data = nib.load(seg_path).get_fdata()
    except Exception as e:
        print(f"  Error loading {case_id}: {e}")
        return None

    # Remap label 4 → 3  (same as DynUNet training map_brats_labels)
    seg_data[seg_data == 4] = 3

    wt_vol  = float(np.sum(seg_data > 0))
    tc_vol  = float(np.sum((seg_data == 1) | (seg_data == 3)))
    et_vol  = float(np.sum(seg_data == 3))
    ncr_vol = float(np.sum(seg_data == 1))
    ed_vol  = float(np.sum(seg_data == 2))

    et_wt_ratio  = et_vol  / (wt_vol  + 1e-6)
    et_tc_ratio  = et_vol  / (tc_vol  + 1e-6)
    tc_wt_ratio  = tc_vol  / (wt_vol  + 1e-6)
    ed_wt_ratio  = ed_vol  / (wt_vol  + 1e-6)
    ncr_tc_ratio = ncr_vol / (tc_vol  + 1e-6)

    wt_binary = (seg_data > 0).astype(float)
    sx = sobel(wt_binary, axis=0)
    sy = sobel(wt_binary, axis=1)
    sz = sobel(wt_binary, axis=2)
    surface_area = float(np.sqrt(sx**2 + sy**2 + sz**2).sum())

    sphericity = 0.0
    if surface_area > 0 and wt_vol > 0:
        sphericity = float(
            (np.pi ** (1/3)) * ((6 * wt_vol) ** (2/3)) / surface_area
        )

    _, n_components = ndimage.label(wt_binary)
    et_present = 1 if et_vol > 50 else 0

    return {
        "case_id":      case_id,
        "wt_vol":       wt_vol,
        "tc_vol":       tc_vol,
        "et_vol":       et_vol,
        "ncr_vol":      ncr_vol,
        "ed_vol":       ed_vol,
        "et_wt_ratio":  et_wt_ratio,
        "et_tc_ratio":  et_tc_ratio,
        "tc_wt_ratio":  tc_wt_ratio,
        "ed_wt_ratio":  ed_wt_ratio,
        "ncr_tc_ratio": ncr_tc_ratio,
        "sphericity":   sphericity,
        "surface_area": surface_area,
        "n_components": float(n_components),
        "et_present":   float(et_present),
    }


FEATURE_COLS = [
    "wt_vol", "tc_vol", "et_vol", "ncr_vol", "ed_vol",
    "et_wt_ratio", "et_tc_ratio", "tc_wt_ratio", "ed_wt_ratio",
    "ncr_tc_ratio", "sphericity", "surface_area", "n_components", "et_present",
]

print("\n" + "="*50)
print("Extracting features from segmentation masks...")
print("(~5-10 min for 369 cases)")
print("="*50)

case_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "BraTS20_Training_*")))
if len(case_dirs) == 0:
    print(f"ERROR: No BraTS20_Training_* folders in {DATA_DIR}")
    sys.exit(1)

all_features = []
for i, case in enumerate(case_dirs):
    name    = os.path.basename(case)
    seg_path = None
    for ext in [".nii.gz", ".nii"]:
        p = os.path.join(case, f"{name}_seg{ext}")
        if os.path.exists(p):
            seg_path = p
            break
    if seg_path is None:
        continue
    feats = extract_features(seg_path, name)
    if feats:
        all_features.append(feats)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(case_dirs)} done...")

df_features = pd.DataFrame(all_features)
print(f"Features extracted: {len(df_features)} cases")


# ============================================================
# STEP 3 — MERGE FEATURES WITH GRADE LABELS
# ============================================================

# Auto-detect the ID column that best matches folder names
id_col = None
for col in df_labels.columns:
    if col == grade_col:
        continue
    sample_csv  = df_labels[col].astype(str).tolist()[:30]
    sample_fold = df_features["case_id"].tolist()[:30]
    hits = sum(any(c in f or f in c for f in sample_fold) for c in sample_csv)
    if hits > 2:
        id_col = col
        print(f"ID column detected: '{col}' ({hits} matches)")
        break

if id_col is None:
    id_col = [c for c in df_labels.columns if c != grade_col][0]
    print(f"Warning: using fallback ID column '{id_col}'")

df_merged = df_features.merge(
    df_labels[[id_col, grade_col]],
    left_on="case_id", right_on=id_col, how="inner"
)

# Fallback: match on the 3-digit case number (e.g. "001")
if len(df_merged) < 10:
    print("Direct merge failed. Trying numeric suffix match...")
    df_labels["_num"]   = df_labels[id_col].astype(str).str.extract(r"(\d{3})")
    df_features["_num"] = df_features["case_id"].str.extract(r"(\d{3})")
    df_merged = df_features.merge(df_labels[["_num", grade_col]], on="_num", how="inner")
    df_merged.drop(columns=["_num"], inplace=True)

print(f"Merged: {len(df_merged)} cases")
print(df_merged[grade_col].value_counts())

if len(df_merged) < 50:
    print("ERROR: Too few cases merged. Check your CSV column names.")
    sys.exit(1)


# ============================================================
# STEP 4 — TRAIN XGBOOST
# ============================================================

X = df_merged[FEATURE_COLS].values
y = (df_merged[grade_col].str.upper() == "HGG").astype(int).values

print(f"\nHGG: {y.sum()} | LGG: {(1-y).sum()} | Total: {len(y)}")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

n_lgg = int((1 - y_train).sum())
n_hgg = int(y_train.sum())
spw   = n_lgg / n_hgg   # scale_pos_weight for class imbalance

model_params = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=2, gamma=0.1,
    scale_pos_weight=spw,
    random_state=42, eval_metric="logloss", verbosity=0,
)

model = xgb.XGBClassifier(**model_params)
print("Training XGBoost (hold-out split)...")
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("Done.")


# ============================================================
# STEP 4b — 5-FOLD STRATIFIED CROSS-VALIDATION
# This is what thesis examiners look for — proves accuracy
# is consistent across all data, not just one lucky split.
# The CV score is what you report in your thesis.
# ============================================================

print("\nRunning 5-fold Stratified Cross-Validation...")
print("(This trains 5 models — takes ~2 min extra)")

skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accs  = []
cv_aucs  = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_scaled, y), 1):
    X_tr, X_va = X_scaled[tr_idx], X_scaled[va_idx]
    y_tr, y_va = y[tr_idx],        y[va_idx]

    n_l = int((1 - y_tr).sum())
    n_h = int(y_tr.sum())
    fold_model = xgb.XGBClassifier(
        **{**model_params, "scale_pos_weight": n_l / n_h}
    )
    fold_model.fit(X_tr, y_tr, verbose=False)
    preds = fold_model.predict(X_va)
    probs = fold_model.predict_proba(X_va)[:, 1]
    cv_accs.append(accuracy_score(y_va, preds))
    cv_aucs.append(roc_auc_score(y_va, probs))
    print(f"  Fold {fold}: Acc={cv_accs[-1]*100:.1f}%  AUC={cv_aucs[-1]:.4f}")

print(f"\n  CV Accuracy : {np.mean(cv_accs)*100:.1f}% \u00b1 {np.std(cv_accs)*100:.1f}%")
print(f"  CV ROC-AUC  : {np.mean(cv_aucs):.4f} \u00b1 {np.std(cv_aucs):.4f}")
print("  ← Use THESE numbers in your thesis (more reliable than single split)")


# ============================================================
# STEP 5 — EVALUATE & PLOTS (On hold-out set from Step 4)
# ============================================================

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print("\n" + "="*55)
print("  RESULTS")
print("="*55)
print(f"  Accuracy : {acc*100:.1f}%")
print(f"  ROC-AUC  : {auc:.4f}")
print(classification_report(y_test, y_pred, target_names=["LGG", "HGG"]))

# Feature importance
importance = model.feature_importances_
idx = np.argsort(importance)
plt.figure(figsize=(10, 7))
plt.barh([FEATURE_COLS[i] for i in idx], importance[idx],
         color=["#e53935" if importance[i] > 0.1 else "#90caf9" for i in idx])
plt.title("XGBoost Feature Importance — HGG vs LGG", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["LGG","HGG"], yticklabels=["LGG","HGG"],
            annot_kws={"size":14,"weight":"bold"})
plt.title(f"Confusion Matrix  (Acc={acc*100:.1f}%)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="#1a73e8", lw=2, label=f"AUC={auc:.3f}")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve — Glioma Grade")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
plt.close()

print(f"\nPlots saved to: {OUTPUT_DIR}")


# ============================================================
# STEP 4c — RETRAIN FINAL MODEL ON ALL DATA
# After CV confirms accuracy, train on 100% of data for
# maximum performance when deployed in the app.
# ============================================================

print("\nRetraining final model on ALL data (100%)...")
n_l_all = int((1 - y).sum())
n_h_all = int(y.sum())
final_model = xgb.XGBClassifier(
    **{**model_params, "scale_pos_weight": n_l_all / n_h_all}
)
final_model.fit(X_scaled, y, verbose=False)
model = final_model   # overwrite — this is the model that gets saved
print("Final model trained on all 368 cases.")


# ============================================================
# STEP 6 — SAVE MODEL  (joblib, matches predict.py)
# ============================================================

model_save_path = os.path.join(OUTPUT_DIR, "xgboost_grade_model.pkl")
joblib.dump({"model": model, "scaler": scaler, "feature_cols": FEATURE_COLS}, model_save_path)

print(f"\n✅ Model saved: {model_save_path}")
print("\n" + "="*55)
if IN_COLAB:
    print("NEXT STEP (Colab):")
    print(f"  Download this file from Drive:")
    print(f"  {model_save_path}")
    print("  Place it in your project folder:")
    print("  auto tumor/models/xgboost_grade_model.pkl")
else:
    print("NEXT STEP:")
    print("  Run: streamlit run app.py")
    print("  Go to Grade Prediction — model loads automatically.")
print("="*55)


# ================================================================
# OFFICIAL RESULTS — NeuroAI Grade Classifier (BraTS 2020)
# Trained: May 2025 | Dataset: BraTS 2020 Training Set
# ================================================================
#
# ── Dataset ──────────────────────────────────────────────────
#   Total cases        : 368  (1 case skipped — no seg file)
#   HGG (High Grade)   : 292  (79.3%)
#   LGG (Low Grade)    : 76   (20.7%)
#   Train / Test split : 80% / 20%  (stratified)
#
# ── Cross-Validation (5-Fold Stratified) ─────────────────────
#   Fold 1  Accuracy: 87.8%   AUC: 0.8237
#   Fold 2  Accuracy: 93.2%   AUC: 0.9593
#   Fold 3  Accuracy: 90.5%   AUC: 0.9300
#   Fold 4  Accuracy: 93.2%   AUC: 0.9885
#   Fold 5  Accuracy: 91.8%   AUC: 0.8586
#   ─────────────────────────────────────
#   CV Mean Accuracy : 91.3% ± 2.0%   ← REPORT THIS IN THESIS
#   CV Mean AUC      : 0.912 ± 0.062  ← REPORT THIS IN THESIS
#
# ── Per-Class Performance (Hold-out Test, 74 cases) ──────────
#               Precision   Recall   F1-Score   Support
#   LGG           0.86       0.80      0.83       15
#   HGG           0.95       0.97      0.96       59
#   ─────────────────────────────────────
#   Accuracy                           0.93       74
#   Macro Avg     0.90       0.88      0.89       74
#   Weighted Avg  0.93       0.93      0.93       74
#
# ── Model Details ─────────────────────────────────────────────
#   Algorithm      : XGBoost (Extreme Gradient Boosting)
#   Features       : 14 volumetric + shape features from DynUNet seg mask
#   Key feature    : ET/WT ratio  (strongest predictor of HGG)
#   Class balancing: scale_pos_weight = LGG_count / HGG_count
#   Final model    : Retrained on ALL 368 cases for deployment
#
# ── Clinical Interpretation ───────────────────────────────────
#   HGG recall 97% → misses only 3% of high-grade cases (safe)
#   LGG recall 80% → 20% LGG misclassified as HGG (conservative)
#   In clinical practice: over-predicting HGG is the safer error
#   (aggressive treatment > missed aggressive tumor)
#
# ── Saved Files ───────────────────────────────────────────────
#   xgboost_grade_model.pkl   → deployed in Streamlit Grade Prediction page
#   feature_importance.png    → top features: et_present, et_wt_ratio, et_vol
#   confusion_matrix.png      → visual performance summary
#   roc_curve.png             → AUC = 0.912
# ================================================================
