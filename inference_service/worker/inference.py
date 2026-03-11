import os
import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification
from insightface.app import FaceAnalysis

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "dima806/deepfake_vs_real_image_detection"

print(f"[MODEL] Loading on {DEVICE}...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model     = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
print(f"[MODEL] Ready.")
face_detector = FaceAnalysis(name="buffalo_l")
face_detector.prepare(ctx_id=0 if DEVICE == "cuda" else -1)
def extract_face(image: Image.Image) -> Image.Image:
    img_array = np.array(image)

    faces = face_detector.get(img_array)

    if not faces:
        return image

    # pick the largest face
    largest_face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    x1, y1, x2, y2 = map(int, largest_face.bbox)

    pad = int(0.2 * (x2 - x1))

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.width, x2 + pad)
    y2 = min(image.height, y2 + pad)

    return image.crop((x1, y1, x2, y2))


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