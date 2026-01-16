# app.py
# RayScan – Lung X-ray Image Classification Web App

import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import torch
from PIL import Image
import torch.nn.functional as F
import tensorflow as tf
import pydicom
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# =========================
# App Configuration
# =========================
st.set_page_config(
    page_title="RayScan – Lung X-ray AI",
    page_icon="🩻",
    layout="wide"
)

# =========================
# Branding
# =========================
APP_NAME = "RayScan"
TAGLINE = "AI-powered early insight from chest X-rays"

# =========================
# Load Logo
# =========================
LOGO_PATH = "/mnt/data/f6bfc212-17b3-4f10-add6-2d78b689455d.jpg"

# =========================
# Load Model
# =========================
@st.cache_resource
def load_trained_model():
    model = torch.load(best_model_fine-tuning.pth, map_location="cpu") #figure out how to write paths
    model.eval()
    return model

model = load_trained_model() 

# =========================
# Import preprocessing
# =========================
from Preprocessing_pipeline import custom_preprocess

IMG_SIZE = (224, 224)

# =========================
# Utility Functions
# =========================

def read_image(uploaded_file):
    if uploaded_file.name.lower().endswith(".dcm"):
        dicom = pydicom.dcmread(uploaded_file)
        img = dicom.pixel_array
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
    else:
        img = np.array(Image.open(uploaded_file).convert("RGB"))
    return img


def image_quality_check(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)

    warnings = []
    if blur_score < 50:
        warnings.append("Image appears blurry")
    if brightness < 60 or brightness > 200:
        warnings.append("Poor lighting detected")

    return warnings


def prepare_image(img):
    img = cv2.resize(img, (224, 224))
    img = custom_preprocess(img)

    img = torch.tensor(img, dtype=torch.float32)
    img = img.permute(2, 0, 1)  # HWC → CHW
    img = img.unsqueeze(0)      # Add batch dim

    return img


def predict(img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

    label = "Cancer" if prob >= 0.5 else "Normal"
    confidence = prob if label == "Cancer" else 1 - prob
    return label, confidence


"""
# =========================
# Grad-CAM
# =========================

def generate_gradcam(model, img_array, layer_name=None):
    if layer_name is None:
        layer_name = [l.name for l in model.layers if 'conv' in l.name][-1]

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    return heatmap


def overlay_heatmap(img, heatmap):
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay

"""

# =========================
# PDF Report
# =========================

def generate_pdf(image, heatmap, label, confidence):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"<b>{APP_NAME} Diagnostic Report</b>", styles['Title']))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(f"Prediction: <b>{label}</b>", styles['Normal']))
    elements.append(Paragraph(f"Confidence Score: {confidence*100:.2f}%", styles['Normal']))

    prompt = (
        "You should see a doctor for confirmation." if label == "Cancer"
        else "You don’t need to worry based on this scan."
    )
    elements.append(Paragraph(f"AI Note: {prompt}", styles['Italic']))
    elements.append(Spacer(1, 0.2 * inch))

    img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    Image.fromarray(image).save(img_path)
    elements.append(RLImage(img_path, width=4*inch, height=4*inch))

    if heatmap is not None:
        hm_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        Image.fromarray(heatmap).save(hm_path)
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph("Grad-CAM Heatmap", styles['Heading2']))
        elements.append(RLImage(hm_path, width=4*inch, height=4*inch))

    doc.build(elements)
    return tmp.name


# =========================
# UI Layout
# =========================

col1, col2 = st.columns([1, 5])
with col1:
    st.image(LOGO_PATH, width=120)
with col2:
    st.markdown(f"## {APP_NAME}")
    st.caption(TAGLINE)

st.divider()

uploaded_file = st.file_uploader(
    "Upload a Chest X-ray (PNG, JPG, JPEG, DICOM)",
    type=["png", "jpg", "jpeg", "dcm"]
)

if uploaded_file:
    image = read_image(uploaded_file)
    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    warnings = image_quality_check(image)
    if warnings:
        st.warning("Image Quality Warning: " + ", ".join(warnings))

    with st.spinner("Analyzing..."):
        img_tensor = prepare_image(image)
        label, confidence = predict(img_tensor)

    st.subheader("Prediction Result")
    st.metric(label="Diagnosis", value=label)
    st.metric(label="Confidence", value=f"{confidence*100:.2f}%")

    if label == "Cancer":
        st.error("You should see a doctor for confirmation.")
        heatmap = generate_gradcam(model, img_tensor)
        overlay = overlay_heatmap(cv2.resize(image, IMG_SIZE), heatmap)
        st.subheader("Grad-CAM Heatmap")
        st.image(overlay, use_container_width=True)
    else:
        st.success("You don’t need to worry based on this scan.")
        heatmap = None
        overlay = None

    pdf_path = generate_pdf(image, overlay, label, confidence)
    with open(pdf_path, "rb") as f:
        st.download_button("Download PDF Report", f, file_name="RayScan_Report.pdf")

st.divider()
st.caption(
    "Disclaimer: Chest X-ray images are limited in detecting lung cancer. "
    "CT scans are most commonly used. RayScan is an early detection aid and "
    "not a tool for definitive diagnosis or medical decision-making."
)