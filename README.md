# NeuroAI — Brain Tumour MRI Analysis System

An advanced AI-powered brain tumour segmentation and analysis system built as a Final Year Project. The system uses an ensemble of three deep learning models trained on the BraTS 2020 dataset to segment brain tumours from MRI scans, extract radiomics features, predict tumour grade, and track disease progression over time.

---

## System Overview

```
MRI Upload (4 modalities: T1, T2, FLAIR, T1ce)
        ↓
Automatic background segmentation pipeline
        ↓
Ensemble of 3 models: DynUNet + SwinUNETR + SegResNet
        ↓
Tumour region extraction: WT, TC, ET, NCR, ED
        ↓
30+ radiomics feature extraction
        ↓
XGBoost grade classifier (HGG vs LGG)
        ↓
Clinical report generation (PDF)
        ↓
Longitudinal comparison across patient visits
```

---

## Key Features

- 3-model ensemble segmentation for higher accuracy than any single model
- Automatic background pipeline — no manual trigger needed
- 30+ radiomics features including volumetric, shape, texture, and GLCM features
- SHAP explainability for grade prediction
- Longitudinal tumour tracking across multiple patient visits
- PDF clinical report generation
- Patient archive system with date-based session management
- Interactive 3D visualizations using Plotly

---

## Project Structure

```
brain-tumor-segmentation/
│
├── app.py                        # Main Streamlit app (ensemble system)
├── app-pre-ensemble.py           # Single model version (DynUNet only)
│
├── core/
│   ├── pipeline.py               # Background auto-segmentation pipeline
│   ├── predict.py                # Model inference and grade prediction
│   ├── feature_report.py         # 30+ radiomics feature extraction
│   ├── followup.py               # Follow-up session management
│   ├── longitudinal_compare.py   # Cross-visit tumour comparison
│   ├── dashboard.py              # Dashboard visualizations
│   ├── auto_segment.py           # Auto segmentation controller
│   ├── patient.py                # Patient record management
│   ├── tumor_habitate.py         # Tumour habitat analysis
│   ├── viewer.py                 # MRI viewer component
│   └── uploader.py               # File upload handler
│
├── utils/
│   ├── file_ops.py               # NIfTI file operations
│   └── plotting.py               # Visualization utilities
│
├── training scripts/
│   ├── training-dynunite-code.py # DynUNet training (Google Colab)
│   ├── training-swinunitr.py     # SwinUNETR training (Google Colab)
│   ├── training-segresnet.py     # SegResNet training (Google Colab)
│   ├── train_grade_model.py      # XGBoost grade classifier training
│   └── xgboost-training.py       # Feature extraction for XGBoost
│
├── patient_archives/             # Sample patient MRI sessions for demo
│
├── MODELS_INFO.txt               # Model download instructions
├── requirements.txt
└── README.md
```

---

## Models

Three deep learning models trained on BraTS 2020 (369 cases):

| Model | WT Dice | TC Dice | ET Dice |
|-------|---------|---------|---------|
| DynUNet | ~89% | ~85% | ~77% |
| SwinUNETR | ~91% | ~87% | ~79% |
| SegResNet | ~90% | ~86% | ~78% |
| Ensemble | Higher | Higher | Higher |

Model weights are stored on Google Drive. See `MODELS_INFO.txt` for download link and setup instructions.

---

## Tumour Regions

```
WT  = Whole Tumour       (all tumour regions combined)
TC  = Tumour Core        (necrotic core + enhancing tumour)
ET  = Enhancing Tumour   (most aggressive region)
NCR = Necrotic Core      (dead tumour tissue)
ED  = Edema              (swelling around tumour)
```

---

## Radiomics Features Extracted

- Volumetric: WT, TC, ET, NCR, ED volumes and ratios
- Shape: sphericity, surface area, component count
- Texture: GLCM contrast, correlation, energy, homogeneity
- Intensity: mean, std, skewness, kurtosis, percentiles
- Advanced: gradient magnitude, local binary patterns, habitat zones

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-00ADEF?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=python&logoColor=white)

---

## How to Run

```bash
# 1. Clone repository
git clone https://github.com/Arifkhan171/brain-tumor-segmentation.git
cd brain-tumor-segmentation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models from Google Drive (see MODELS_INFO.txt)
# Place all .pt and .pkl files in models/ folder

# 4. Run the application
streamlit run app.py
```

---

## Dataset

- BraTS 2020 (Brain Tumour Segmentation Challenge)
- 369 training cases with 4 MRI modalities each
- Trained on Google Colab Pro with NVIDIA T4/A100 GPU

---

## Author

**Arif Khan**
BS Computer Science — Final Year Project
University of Loralai

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/arif-khan-71a711376)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github)](https://github.com/Arifkhan171)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail)](mailto:arif.cs.bs@gmail.com)
