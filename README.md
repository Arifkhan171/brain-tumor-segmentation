<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d0d0d,25:0a1628,50:0d2137,75:003366,100:00ADEF&height=220&section=header&text=NeuroAI&fontSize=72&fontColor=00FFFF&animation=fadeIn&fontAlignY=44&desc=Brain%20Tumour%20MRI%20Analysis%20System%20%7C%20Final%20Year%20Project&descAlignY=66&descSize=20&descColor=87CEEB&stroke=00ADEF&strokeWidth=2" width="100%"/>

<br/>

[![YouTube Demo](https://img.shields.io/badge/▶️_Watch_Full_Demo-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/zeKYup9Rga8)
&nbsp;
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
&nbsp;
[![MONAI](https://img.shields.io/badge/MONAI-00ADEF?style=for-the-badge&logo=python&logoColor=white)](https://monai.io)
&nbsp;
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

<br/>

![Status](https://img.shields.io/badge/Status-Final_Year_Project-2ea44f?style=flat-square)
&nbsp;
![Models](https://img.shields.io/badge/Models-3_Ensemble-00ADEF?style=flat-square)
&nbsp;
![Dataset](https://img.shields.io/badge/Dataset-BraTS_2020-003366?style=flat-square)
&nbsp;
![Features](https://img.shields.io/badge/Radiomics-30+_Features-EE4C2C?style=flat-square)

</div>

<img src="https://capsule-render.vercel.app/api?type=soft&color=0:0d0d0d,50:003366,100:00ADEF&height=6" width="100%"/>

---

## 📌 Overview

> "From raw MRI scan to full clinical report — fully automated, no manual steps."

*NeuroAI* is a clinical-grade AI system that takes raw brain MRI scans as input and delivers complete tumour analysis — segmentation, habitat mapping, grade prediction, radiomic feature extraction, and longitudinal comparison — all through an automated pipeline powered by a 3-model deep learning ensemble.

Built as a Final Year Project at *University of Loralai*, this system demonstrates production-level medical AI engineering using state-of-the-art architectures trained on the BraTS 2020 challenge dataset.

---

## 🎬 Demo

<div align="center">

[![Watch Demo](https://img.shields.io/badge/▶️_Watch_Full_System_Demo_on_YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white&labelColor=0d0d0d)](https://youtu.be/zeKYup9Rga8)

</div>

---

## 📸 System Screenshots

> *Setup:* Create a folder named screenshots in your repo root and upload images with the exact filenames below.

---

### 🏥 Patient Registration — MRI Upload Portal

> Patient details are registered before MRI upload. All information flows automatically into reports and clinical records.

![Patient Registration](screenshots/patient-registration.jpg)

---

### 🧠 MRI Viewer — 3-Plane Tumour Visualisation

> Interactive 3-plane viewer (Axial · Coronal · Sagittal) with colour-coded tumour overlays. Blue = Whole Tumour, Red = Enhancing Tumour core.

![MRI 3-Plane Viewer](screenshots/mri-3plane-viewer.jpg)

---

### 🗺️ Habitat Analysis — Tumour Habitat Map Overlay

> K-means based tumour habitat clustering displayed as a colour overlay on the MRI axial slice. Reveals internal tumour microenvironment zones.

![Habitat Overlay](screenshots/habitat-overlay.jpg)

---

### 📊 Habitat Analysis — Habitat Statistics

> Quantified habitat statistics: voxel count, volume (cm³), mean intensity, and standard deviation for each habitat zone.

![Habitat Statistics](screenshots/habitat-statistics.jpg)

---

### 🔬 Grade Prediction — HGG vs LGG Classification

> XGBoost classifier predicts tumour grade with 99.4% confidence. Clinical interpretation and standard treatment protocol displayed automatically.

![Grade Prediction](screenshots/grade-prediction.jpg)

---

### 📈 Compare Visits — Longitudinal Tumour Tracking

> Side-by-side comparison of two patient visits. Tracks WT Volume, ET Volume, Edema, and Max Axial Diameter changes with percentage deltas and radar signature charts.

![Compare Visits](screenshots/compare-visits.jpg)

---

## ⚙️ Core Features

<table>
<tr>
<td width="50%" valign="top">

### 🤖 AI Pipeline
- ✅ 3-model ensemble: DynUNet + SwinUNETR + SegResNet
- ✅ Automatic background segmentation — no manual trigger
- ✅ 4 MRI modalities: T1, T2, FLAIR, T1ce
- ✅ 5 tumour region extraction: WT, TC, ET, NCR, ED
- ✅ XGBoost grade classifier (HGG vs LGG)
- ✅ SHAP explainability for grade prediction

</td>
<td width="50%" valign="top">

### 📊 Clinical Analysis
- ✅ 30+ radiomic features extracted
- ✅ Tumour habitat analysis with K-means clustering
- ✅ Longitudinal comparison across patient visits
- ✅ Interactive 2D slice viewer, 3-plane viewer, 3D mesh
- ✅ Automated PDF clinical report generation
- ✅ Patient archive with date-based session management

</td>
</tr>
</table>

---

## 🏗️ System Pipeline

mermaid
%%{init: {'theme': 'dark'}}%%
flowchart TD
    A[📁 MRI Upload\n4 Modalities: T1 T2 FLAIR T1ce] --> B

    B[⚙️ Preprocessing\nNormalization + Background Pipeline] --> C

    C[🧠 Ensemble Segmentation] --> C1 & C2 & C3

    C1[DynUNet\n~89% WT Dice]
    C2[SwinUNETR\n~91% WT Dice]
    C3[SegResNet\n~90% WT Dice]

    C1 & C2 & C3 --> D

    D[🎯 Ensemble Output\nWT · TC · ET · NCR · ED] --> E & F & G & H

    E[📐 Radiomic Features\n30+ volumetric shape texture GLCM]
    F[🗺️ Habitat Analysis\nK-means tumour zone mapping]
    G[🔬 Grade Prediction\nXGBoost HGG vs LGG + SHAP]
    H[🖥️ MRI Viewer\n2D · 3-Plane · 3D Mesh]

    E & F & G --> I[📋 PDF Clinical Report\nAuto-generated per patient]
    I --> J[📅 Longitudinal Tracking\nCross-visit comparison]

    style A fill:#003366,color:#00FFFF
    style B fill:#0d2137,color:#fff
    style C fill:#0a1628,color:#87CEEB
    style D fill:#003366,color:#00FFFF
    style E fill:#0d2137,color:#fff
    style F fill:#0d2137,color:#fff
    style G fill:#0d2137,color:#fff
    style H fill:#0d2137,color:#fff
    style I fill:#2ea44f,color:#fff
    style J fill:#00ADEF,color:#000


---

## 🎯 Model Performance

| Model | WT Dice | TC Dice | ET Dice | Architecture |
|---|---|---|---|---|
| DynUNet | ~89% | ~85% | ~77% | Dynamic U-Net |
| SwinUNETR | ~91% | ~87% | ~79% | Swin Transformer |
| SegResNet | ~90% | ~86% | ~78% | Residual Encoder |
| *Ensemble* | *Higher* | *Higher* | *Higher* | *Voted Output* |

> Trained on *BraTS 2020* — 369 cases with 4 MRI modalities each. Training done on Google Colab Pro with NVIDIA T4/A100 GPU.

---

## 🧬 Tumour Regions Explained

| Region | Full Name | Description |
|---|---|---|
| WT | Whole Tumour | All tumour regions combined |
| TC | Tumour Core | Necrotic core + enhancing tumour |
| ET | Enhancing Tumour | Most aggressive, actively growing region |
| NCR | Necrotic Core | Dead tumour tissue inside |
| ED | Edema | Brain swelling around the tumour |

---

## 📐 Radiomic Features Extracted (30+)

<details>
<summary><b>Click to expand full feature list</b></summary>

<br/>

| Category | Features |
|---|---|
| *Volumetric* | WT, TC, ET, NCR, ED volumes and ratios |
| *Shape* | Sphericity, surface area, component count |
| *Texture (GLCM)* | Contrast, correlation, energy, homogeneity |
| *Intensity* | Mean, std, skewness, kurtosis, percentiles |
| *Advanced* | Gradient magnitude, local binary patterns, habitat zones |

</details>

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| 🧠 *Segmentation* | MONAI + PyTorch | DynUNet, SwinUNETR, SegResNet models |
| 📊 *Grade Classification* | XGBoost + SHAP | HGG vs LGG tumour grade prediction |
| 🗺️ *Habitat Analysis* | scikit-learn K-means | Tumour microenvironment zone mapping |
| 📐 *Radiomics* | PyRadiomics + NumPy | 30+ feature extraction |
| 🖥️ *Interface* | Streamlit | Interactive clinical dashboard |
| 📋 *Reports* | ReportLab / FPDF | Automated PDF generation |
| 🧮 *Image Processing* | SimpleITK + OpenCV | NIfTI handling and MRI preprocessing |
| 📈 *Visualisation* | Plotly | 3D tumour mesh and interactive charts |
| 🐍 *Language* | Python 3.11 | Core system language |

---

## 🚀 Setup Instructions

<details>
<summary><b>Click to expand setup guide</b></summary>

<br/>

### Prerequisites
- Python 3.11+
- NVIDIA GPU recommended (CPU works but is slow)
- 8GB+ RAM

### Installation

bash
# 1. Clone the repository
git clone https://github.com/Arifkhan171/brain-tumor-segmentation.git
cd brain-tumor-segmentation

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Download model weights from Google Drive
# See MODELS_INFO.txt for the download link
# Place all .pt and .pkl files inside the models/ folder

# 4. Launch the application
streamlit run app.py


### Folder Structure After Setup

brain-tumor-segmentation/
├── models/
│   ├── dynunet_model.pt
│   ├── swinunetr_model.pt
│   ├── segresnet_model.pt
│   └── xgboost_grade_model.pkl
├── screenshots/          ← Upload your screenshots here
├── app.py
└── requirements.txt


</details>

---

## 📁 Project Structure

<details>
<summary><b>Click to expand full structure</b></summary>

<br/>


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
│   ├── training-dynunite-code.py # DynUNet training
│   ├── training-swinunitr.py     # SwinUNETR training
│   ├── training-segresnet.py     # SegResNet training
│   └── train_grade_model.py      # XGBoost classifier training
│
├── patient_archives/             # Sample patient MRI sessions
├── screenshots/                  # README screenshots
├── MODELS_INFO.txt               # Model download instructions
└── requirements.txt


</details>

---

## 🔒 Repository Access

This is a *private repository*.

To request access for review or collaboration:

[![Email](https://img.shields.io/badge/Request_Access-arif.cs.bs%40gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:arif.cs.bs@gmail.com)
&nbsp;
[![LinkedIn](https://img.shields.io/badge/Connect-Arif_Khan-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arif-khan-71a711376)

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=soft&color=0:0d0d0d,50:003366,100:00ADEF&height=8" width="100%"/>

<br/>

*Built by Arif Khan — AI Engineer | Final Year Project | University of Loralai, Pakistan*

⚠️ For research and educational use only. Clinical diagnosis requires histopathological confirmation.

<br/>

If this project helped you — a ⭐ on the repo means the world!

<br/>

</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00ADEF,40:003366,70:0a1628,100:0d0d0d&height=130&section=footer" width="100%"/>
