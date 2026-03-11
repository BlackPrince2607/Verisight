import os
import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-Model"

print(f"[MODEL] Loading on {DEVICE}...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model     = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
print(f"[MODEL] Ready.")

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extract_face(image: Image.Image) -> Image.Image:
    img_array = np.array(image)
    gray  = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return image
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.2 * min(w, h))
    return image.crop((
        max(0, x - pad),
        max(0, y - pad),
        min(image.width,  x + w + pad),
        min(image.height, y + h + pad)
    ))


def run_inference(image: Image.Image) -> tuple[str, float]:
    image  = extract_face(image)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs      = torch.nn.functional.softmax(outputs.logits, dim=1)
    conf, pred = torch.max(probs, dim=1)
    label      = model.config.id2label[pred.item()]
    return label, round(float(conf.item()), 4)


def run_inference_from_path(file_path: str) -> tuple[str, float]:
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    try:
        image = Image.open(file_path).convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"Cannot read image: {e}")
    return run_inference(image)