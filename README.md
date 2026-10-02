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

### 🏥 Patient Registration — MRI Upload Portal

> Patient details are registered before MRI upload. All information flows automatically into reports and clinical records.

<img width="1600" height="738" alt="WhatsApp Image 2026-10-02 at 1 05 26 PM (5)" src="https://github.com/user-attachments/assets/126e4bc3-f79d-4893-b2f2-f58866c286ca" />
<img width="1600" height="745" alt="WhatsApp Image 2026-10-02 at 1 05 26 PM (4)" src="https://github.com/user-attachments/assets/86adf5a9-7ab9-449b-ba58-8fc831b85312" />


---

### 🧠 MRI Viewer — 3-Plane Tumour Visualisation

> Interactive 3-plane viewer (Axial · Coronal · Sagittal) with colour-coded tumour overlays. Blue = Whole Tumour, Red = Enhancing Tumour core.

<img width="1600" height="704" alt="WhatsApp Image 2026-10-02 at 1 05 26 PM (3)" src="https://github.com/user-attachments/assets/5258341f-f6c4-4fd6-acc5-41ff320b944b" />


---

### 🗺️ Habitat Analysis — Tumour Habitat Map Overlay

> K-means based tumour habitat clustering displayed as a colour overlay on the MRI axial slice. Reveals internal tumour microenvironment zones.

<img width="1600" height="754" alt="WhatsApp Image 2026-10-02 at 1 05 26 PM (2)" src="https://github.com/user-attachments/assets/59cd0693-6f5a-4381-9c87-6882463ad126" />


---

### 📊 Habitat Analysis — Habitat Statistics

> Quantified habitat statistics: voxel count, volume cm³, mean intensity, and standard deviation for each habitat zone.

<img width="1600" height="736" alt="WhatsApp Image 2026-10-02 at 1 05 26 PM (1)" src="https://github.com/user-attachments/assets/f568f19b-b8b0-4695-9113-65d0f0e9cde5" />


---

### 🔬 Grade Prediction — HGG vs LGG Classification

> XGBoost classifier predicts tumour grade with 99.4% confidence. Clinical interpretation and standard treatment protocol displayed automatically.

<img width="1600" height="738" alt="WhatsApp Image 2026-10-02 at 1 05 26 PM" src="https://github.com/user-attachments/assets/0b961c0f-be92-496f-a098-edf938a9e3f9" />


---

### 📈 Compare Visits — Longitudinal Tumour Tracking

> Side-by-side comparison of two patient visits with radar signature charts and volume change percentages.

<img width="1600" height="726" alt="WhatsApp Image 2026-10-02 at 1 05 26 PM (6)" src="https://github.com/user-attachments/assets/a25641a5-f9ad-490b-8dc6-50304ad1c923" />


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
- ✅ XGBoost grade classifier — HGG vs LGG
- ✅ SHAP explainability for grade prediction

</td>
<td width="50%" valign="top">

### 📊 Clinical Analysis
- ✅ 30+ radiomic features extracted
- ✅ Tumour habitat analysis with K-means clustering
- ✅ Longitudinal comparison across patient visits
- ✅ 2D slice viewer, 3-plane viewer, 3D mesh
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

> Trained on *BraTS 2020* — 369 cases, 4 MRI modalities each. Google Colab Pro with NVIDIA T4/A100 GPU.

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

## 📐 Radiomic Features Extracted

<details>
<summary><b>Click to expand full feature list</b></summary>
<br/>

| Category | Features |
|---|---|
| *Volumetric* | WT, TC, ET, NCR, ED volumes and ratios |
| *Shape* | Sphericity, surface area, component count |
| *Texture GLCM* | Contrast, correlation, energy, homogeneity |
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
| 📋 *Reports* | ReportLab | Automated PDF generation |
| 🧮 *Image Processing* | SimpleITK + OpenCV | NIfTI handling and MRI preprocessing |
| 📈 *Visualisation* | Plotly | 3D tumour mesh and interactive charts |
| 🐍 *Language* | Python 3.11 | Core system language |

---

## 🚀 Setup Instructions

<details>
<summary><b>Click to expand setup guide</b></summary>
<br/>

bash
# 1. Clone the repository
git clone https://github.com/Arifkhan171/brain-tumor-segmentation.git
cd brain-tumor-segmentation

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Download model weights
# See MODELS_INFO.txt for Google Drive download link
# Place all .pt and .pkl files inside the models/ folder

# 4. Launch the application
streamlit run app.py


</details>

---

## 🔒 Repository Access

This is a *private repository*.

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

</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00ADEF,40:003366,70:0a1628,100:0d0d0d&height=130&section=footer" width="100%"/>
