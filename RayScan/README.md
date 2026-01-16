# RayScan – Chest X-ray AI Screening Demo

RayScan is a Streamlit-based web application that provides AI-assisted screening insights from chest X-ray images.
It is designed for **research and demonstration purposes only** and is **not a medical diagnostic tool**.

---

## 🚀 Features

* Upload chest X-ray images (PNG, JPG, JPEG, or DICOM)
* AI-based binary classification:

  * **Cancer**
  * **Normal**
* Confidence scores and probability breakdown
* Grad-CAM visual explanation for Cancer predictions
* Image quality and validity checks
* Interactive viewer controls (brightness, contrast, zoom)
* Downloadable PDF screening report

---

## 🧠 Model Overview

* **Architecture:** DenseNet-121
* **Framework:** PyTorch
* **Task:** Binary image classification
* **Classes:**

  * Class 0: Cancer
  * Class 1: Normal
* **Training:** Supervised learning on a curated chest X-ray dataset
* **Checkpoint:** `best_model_fine-tuning.pth`

The model is loaded in evaluation mode and runs entirely on CPU unless a GPU is available.

---

## 🧪 Preprocessing Pipeline

### Training (offline)

During training, images were processed using a custom medical imaging pipeline:

* Grayscale conversion
* Gaussian noise reduction
* CLAHE contrast enhancement
* Conversion back to RGB
* Z-score normalization

### Inference (this app)

For robustness and deployment simplicity, inference uses:

* Image resizing to 224×224
* Conversion to tensor
* ImageNet-style normalization

This difference is intentional and documented.
While it may slightly affect performance on out-of-distribution images, it improves stability for real-world uploads.

---

## 📂 Supported File Formats

* PNG
* JPG / JPEG
* DICOM (.dcm)

DICOM support requires `pydicom`. If unavailable, standard image formats can still be used.

---

## 🔍 Explainability (Grad-CAM)

When the model predicts **Cancer**, Grad-CAM is applied to highlight regions that contributed most to the prediction.
This visualization is intended for interpretability only and does not indicate confirmed pathology.

---

## 📄 PDF Report

Users can generate a downloadable PDF report containing:

* Uploaded image
* Model input preview
* Grad-CAM heatmap (if applicable)
* Prediction result and confidence
* Safety disclaimers

---

## ⚠️ Important Disclaimer

**Chest X-rays have significant limitations for detecting lung cancer.**
CT scans are commonly used for confirmation.

RayScan is:

* A research/demo application
* An early screening aid
* **NOT a diagnostic system**

Do **not** use this application for clinical decision-making.

---

## 🛠 Installation & Deployment

### Requirements

```
streamlit
torch
torchvision
numpy
opencv-python-headless
pillow
reportlab
pydicom
```

### Run locally

```bash
streamlit run app.py
```

### Deploy (Streamlit Community Cloud)

1. Push this repository to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New app**
4. Select repository and `app.py`
5. Click **Deploy**

---

## 📜 License & Use

This project is intended for:

* Academic use
* Demonstrations
* Learning purposes

No patient data is stored, transmitted, or retained.

---

## 👤 Author

Developed as part of an academic project to explore AI-assisted medical image analysis.
