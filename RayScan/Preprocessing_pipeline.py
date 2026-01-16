"""**PREPROCESSING**"""

import cv2
import numpy as np

# Z-score normalization function
def zscore_normalize(image, eps=1e-7):
    image = image.astype(np.float32)
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / (std + eps)

def custom_preprocess(img):
    img = img.astype(np.uint8)

    # Grayscale handling
    if img.ndim == 3 and img.shape[-1] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.squeeze()

    # Noise reduction
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Convert grayscale → RGB
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Z-score normalization
    img = zscore_normalize(img)

    return img

# Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocess,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocess
)

train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/Capstone_X-ray_Project/Data/chest_xray_lung_v1_cleaned/train',
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='sparse',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    '/content/drive/MyDrive/Capstone_X-ray_Project/Data/chest_xray_lung_v1_cleaned/val',
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='sparse',
    shuffle=False
)

train_generator.class_indices