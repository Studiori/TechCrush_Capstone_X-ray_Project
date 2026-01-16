# app.py
# RayScan – Chest X-ray Image Classification Web App (Normal vs Cancer)

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import cv2  # use opencv-python-headless in deployment

# Optional DICOM support
try:
    import pydicom
    DICOM_AVAILABLE = True
except Exception:
    DICOM_AVAILABLE = False

# PDF generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch


# =========================
# App Configuration
# =========================
st.set_page_config(
    page_title="RayScan – Chest X-ray AI",
    page_icon="🩻",
    layout="wide"
)

# =========================
# Branding
# =========================
APP_NAME = "RayScan"
TAGLINE = "Fast AI screening insights from chest X-rays (research/demo)"
BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "Logo.jpg"
LOGIT_TEMPERATURE = 1.0214163064956665

   # <-- place your logo.jpeg beside app.py

# =========================
# Model / Data Config
# =========================
CHECKPOINT_PATH = BASE_DIR / "best_model_fine-tuning.pth"
CLASS_NAMES = ["Cancer", "Normal"] # Binary classifier
IMAGE_SIZE = 224
CANCER_INDEX = 0
THRESHOLD = 0.50  # decision threshold for class index 1


# =========================
# Preprocessing (MATCHES TRAINING PIPELINE)
# Grayscale -> GaussianBlur -> CLAHE -> Gray->RGB -> Z-score normalize
# (from Preprocessing_pipeline.py)
# =========================
# =========================
# Preprocessing (CORRECT)
# =========================
from torchvision import transforms

IMAGE_SIZE = 224

# Name the transform something DIFFERENT
inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),  # [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_for_model(pil_img: Image.Image) -> torch.Tensor:
    """
    Output: [1, 3, 224, 224]
    """
    x = inference_transform(pil_img.convert("RGB"))
    return x.unsqueeze(0)

# =========================
# Model (DenseNet121 w/ classifier as Sequential(Dropout, Linear))
# =========================
class CancerClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.base_model = models.densenet121(weights=None)

        # Your checkpoint expects base_model.classifier.1.weight (Linear) with in_features=1024
        num_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


def infer_num_classes_from_checkpoint(state_dict: dict) -> int:
    key = "base_model.classifier.1.weight"
    if key in state_dict and hasattr(state_dict[key], "shape"):
        return int(state_dict[key].shape[0])
    return 2


@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint missing 'model_state_dict'.")

    sd = checkpoint["model_state_dict"]
    num_classes = infer_num_classes_from_checkpoint(sd)

    model = CancerClassifier(num_classes=num_classes, dropout_rate=0.5)
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)

    meta = {
        "device": str(device),
        "num_classes": num_classes,
        "epoch": checkpoint.get("epoch"),
        "val_acc": checkpoint.get("val_acc"),
        "val_loss": checkpoint.get("val_loss"),
        "stage": checkpoint.get("stage"),
    }
    return model, meta


def predict(model: nn.Module, pil_img: Image.Image, device: torch.device):
    x = preprocess_for_model(pil_img).to(device)

    with torch.no_grad():
        logits = model(x)                    # [1, 2]
        logits = logits / LOGIT_TEMPERATURE
        probs = F.softmax(logits, dim=1)[0]  # [2]

        cancer_prob = float(probs[0].item())   # Cancer
        normal_prob = float(probs[1].item())   # NORMAL

    # Index 0 = Cancer, Index 1 = Normal
    if cancer_prob >= THRESHOLD:
        return "Cancer", cancer_prob, np.array([cancer_prob, normal_prob], dtype=np.float32)
    else:
        return "Normal", normal_prob, np.array([cancer_prob, normal_prob], dtype=np.float32)


# =========================
# Image loading + validation + quality checks
# =========================
def read_image(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".dcm"):
        if not DICOM_AVAILABLE:
            raise RuntimeError("DICOM upload requires pydicom. Install pydicom or upload PNG/JPG.")
        ds = pydicom.dcmread(uploaded_file)
        img = ds.pixel_array.astype(np.float32)

        # Normalize to 0..255 for display + downstream preprocessing
        img = img - img.min()
        img = img / (img.max() + 1e-8)
        img = (img * 255.0).clip(0, 255).astype(np.uint8)

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pil = Image.fromarray(img_rgb)
        return pil

    # standard image
    pil = Image.open(uploaded_file).convert("RGB")
    return pil


def looks_like_xray(pil_img: Image.Image) -> (bool, list): # type: ignore
    """
    Heuristic validation (not perfect): checks size, grayscale-like nature, and dynamic range.
    We DO NOT hard-block unless it's clearly invalid; we warn.
    """
    warnings = []

    w, h = pil_img.size
    if w < 150 or h < 150:
        warnings.append("Image resolution is very low; results may be unreliable.")

    arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # grayscale-likeness: channel differences should be small for most X-rays
    ch_diff = np.mean(np.abs(arr[..., 0].astype(np.int16) - arr[..., 1].astype(np.int16))) \
              + np.mean(np.abs(arr[..., 1].astype(np.int16) - arr[..., 2].astype(np.int16)))
    if ch_diff > 30:
        warnings.append("Image looks strongly colored; many X-rays are grayscale. Please confirm this is an X-ray.")

    # dynamic range check
    p5, p95 = np.percentile(gray, [5, 95])
    if (p95 - p5) < 30:
        warnings.append("Low contrast detected; try using a clearer scan if available.")

    # edge density check (extremely crude)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.mean(edges > 0))
    if edge_ratio < 0.005:
        warnings.append("Very few edges detected; image may be blank/over-smoothed or not an X-ray.")

    # We'll treat it as "valid" unless it's obviously broken
    valid = True
    if np.std(gray) < 5:
        valid = False
        warnings.append("Image appears nearly uniform; file may be corrupted or not a valid scan.")

    return valid, warnings


def image_quality_check(pil_img: Image.Image):
    """
    Returns warnings about blur/lighting.
    """
    arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))

    warnings = []
    if blur_score < 50:
        warnings.append("Image appears blurry (low sharpness).")
    if brightness < 60:
        warnings.append("Image is quite dark; details may be lost.")
    if brightness > 200:
        warnings.append("Image is very bright; details may be washed out.")

    return warnings


def apply_viewer_controls(pil_img: Image.Image, brightness: float, contrast: float, zoom: float):
    img = pil_img.copy()
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    if zoom > 1.0:
        w, h = img.size
        new_w = int(w / zoom)
        new_h = int(h / zoom)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        img = img.crop((left, top, left + new_w, top + new_h)).resize((w, h))

    return img


# =========================
# Grad-CAM (DenseNet features)
# =========================
def grad_cam(model: CancerClassifier, pil_img: Image.Image, class_index: int, device: torch.device):
    """
    Returns heatmap in [0,1] resized to IMAGE_SIZE x IMAGE_SIZE
    """
    model.zero_grad(set_to_none=True)

    target_module = model.base_model.features
    activations = {}
    gradients = {}

    def fwd_hook(_, __, output):
        # Keep a reference to the activation tensor (needed for autograd.grad)
        activations["value"] = output

    h1 = target_module.register_forward_hook(fwd_hook)

    x = preprocess_for_model(pil_img).to(device)
    x.requires_grad_(True)

    logits = model(x)
    score = logits[:, class_index].sum()

    # Use autograd.grad to compute gradients w.r.t. the captured activations
    grad = torch.autograd.grad(score, activations["value"], retain_graph=False, allow_unused=False)[0]

    h1.remove()

    act = activations["value"]     # [1, C, H, W]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return cam


def overlay_heatmap(pil_img: Image.Image, heatmap_01: np.ndarray):
    """
    Returns overlay image (uint8 RGB) for display/report.
    """
    base = pil_img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    base_arr = np.array(base, dtype=np.uint8)

    hm = (heatmap_01 * 255).clip(0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)          # BGR
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)        # RGB

    overlay = cv2.addWeighted(base_arr, 0.65, hm_color, 0.35, 0)
    return overlay


# =========================
# PDF Report
# =========================
def generate_pdf(original_pil: Image.Image, processed_preview_rgb: np.ndarray, overlay_rgb: np.ndarray | None,
                 label: str, confidence: float):
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp_pdf.name, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"<b>{APP_NAME} – AI Screening Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.15 * inch))
    elements.append(Paragraph(f"Result: <b>{label}</b>", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {confidence*100:.2f}%", styles["Normal"]))
    elements.append(Spacer(1, 0.15 * inch))

    if label == "Cancer":
        note = "AI suggests possible findings consistent with Cancer. Please see a clinician for confirmation."
    else:
        note = "AI suggests Normal on this scan. If symptoms persist, consult a clinician."
    elements.append(Paragraph(f"<i>Note:</i> {note}", styles["Italic"]))
    elements.append(Spacer(1, 0.2 * inch))

    # Save images temporarily for reportlab
    tmp_img1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    original_pil.convert("RGB").save(tmp_img1)

    tmp_img2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    Image.fromarray(processed_preview_rgb).save(tmp_img2)

    elements.append(Paragraph("Uploaded Image", styles["Heading2"]))
    elements.append(RLImage(tmp_img1, width=5.5*inch, height=5.5*inch))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Model Input Preview (training preprocessing)", styles["Heading2"]))
    elements.append(RLImage(tmp_img2, width=5.5*inch, height=5.5*inch))
    elements.append(Spacer(1, 0.2 * inch))

    if overlay_rgb is not None:
        tmp_img3 = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        Image.fromarray(overlay_rgb).save(tmp_img3)

        elements.append(Paragraph("Grad-CAM Heatmap (Cancer prediction)", styles["Heading2"]))
        elements.append(RLImage(tmp_img3, width=5.5*inch, height=5.5*inch))
        elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(
        "Disclaimer: Chest X-ray images have limitations for detecting lung cancer. CT scans are commonly used for confirmation. "
        "RayScan is an early screening aid for research/demo use and is not a definitive diagnostic tool.",
        styles["Normal"]
    ))

    doc.build(elements)
    return tmp_pdf.name


# =========================
# UI
# =========================
# Header
c1, c2 = st.columns([1, 6], gap="medium")
with c1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=120)
    else:
        st.markdown("🩻")
with c2:
    st.markdown(f"## {APP_NAME}")
    st.caption(TAGLINE)

st.divider()

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.05)
    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.05)
    zoom = st.slider("Zoom", 1.0, 3.0, 1.0, 0.05)

    st.divider()
    st.subheader("Prediction Threshold")
    st.write(f"Cancer if probability ≥ *{THRESHOLD:.2f}*")

    st.divider()
    st.subheader("Formats")
    st.write("PNG, JPG, JPEG")
    st.write("DICOM (.dcm)" + (" ✅" if DICOM_AVAILABLE else " ❌ (install pydicom)"))


# Load model
try:
    model, meta = load_trained_model()
    device = torch.device(meta["device"])
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# Model details intentionally hidden in the public UI to avoid exposing internal metadata

# Upload
uploaded_file = st.file_uploader(
    "Upload a chest X-ray (PNG/JPG/JPEG or DICOM)",
    type=["png", "jpg", "jpeg", "dcm"]
)

if not uploaded_file:
    st.info("Upload an image to begin.")
    st.stop()

# Read + viewer controls
try:
    original_pil = read_image(uploaded_file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

view_pil = apply_viewer_controls(original_pil, brightness=brightness, contrast=contrast, zoom=zoom)

# Validation & quality checks
valid, xray_warnings = looks_like_xray(view_pil)
quality_warnings = image_quality_check(view_pil)

if not valid:
    st.warning("This file may not be a valid X-ray image. You can still continue, but results may be meaningless.")
if xray_warnings:
    st.warning("Validation notes: " + " | ".join(xray_warnings))
if quality_warnings:
    st.warning("Image quality warnings: " + " | ".join(quality_warnings))

# Layout: viewer + results
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.subheader("Image Viewer")
    st.image(view_pil, caption="Displayed (viewer controls applied)", use_container_width=True)

    # show model-input preview (what the model sees after your training preprocessing)
    pre = preprocess_for_model(view_pil).squeeze(0).permute(1, 2, 0).cpu().numpy()  # z-scored float32
    disp = pre.copy()
    disp = disp - disp.min()
    disp = disp / (disp.max() + 1e-8)
    disp_uint8 = (disp * 255).astype(np.uint8)
    st.image(disp_uint8, caption="Model input preview (training preprocessing)", use_container_width=True)

with right:
    st.subheader("AI Output")

    with st.spinner("Analyzing..."):
        label, confidence, probs = predict(model, original_pil, device)

    st.metric("Diagnosis", label)
    st.metric("Confidence", f"{confidence*100:.2f}%")

    st.write("*Class probabilities*")
    st.dataframe(
        [{"Class": CLASS_NAMES[i], "Probability": float(probs[i])} for i in range(len(CLASS_NAMES))],
        use_container_width=True
    )
    st.bar_chart({CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))})

    if label == "Cancer":
        st.error("You should see a doctor for confirmation.")
    else:
        st.success("You don’t need to worry based on this scan. If symptoms persist, consult a clinician.")


# Grad-CAM only for Cancer
overlay_rgb = None
if label == "Cancer":
    st.subheader("Grad-CAM Heatmap (Cancer prediction)")
    try:
        cam = grad_cam(model, view_pil, class_index=CANCER_INDEX, device=device)
        overlay_rgb = overlay_heatmap(view_pil, cam)
        st.image(overlay_rgb, use_container_width=True)
    except Exception as e:
        st.warning("Unable to generate explanation heatmap for this image.")

# PDF Report
st.subheader("Report")
if st.button("Generate PDF Report"):
    with st.spinner("Generating report..."):
        pdf_path = generate_pdf(
            original_pil=original_pil,
            processed_preview_rgb=disp_uint8,
            overlay_rgb=overlay_rgb,
            label=label,
            confidence=confidence
        )
    with open(pdf_path, "rb") as f:
        st.download_button(
            "Download RayScan Report (PDF)",
            f,
            file_name="RayScan_Report.pdf",
            mime="application/pdf"
        )

st.divider()


# Conspicuous disclaimer: heading + red error box for visibility
st.markdown("<h3 style='color:#8B0000;margin:0 0 6px 0;'>Important Disclaimer</h3>", unsafe_allow_html=True)
st.error(
    "Chest X-ray images have significant limitations for detecting lung cancer. CT scans are commonly used for confirmation.\n\n"
    "RayScan is an early screening aid for research/demo use and is not a definitive diagnostic tool. Do NOT use this for clinical diagnosis."
)
# End of app.py