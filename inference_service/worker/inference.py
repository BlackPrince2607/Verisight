import os
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification
from insightface.app import FaceAnalysis

# ── CONFIG ──────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-Model"

CONFIDENCE_THRESHOLD = 0.72   # below this → return "uncertain"
FACE_PADDING         = 0.22   # fractional padding around face crop

# ── LOAD MODEL ──────────────────────────────────────────────────────
print(f"[MODEL] Loading {MODEL_NAME} on {DEVICE}...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model     = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
print("[MODEL] Ready.")

# ── LOAD FACE DETECTOR ──────────────────────────────────────────────
# buffalo_sc = detection only, faster than buffalo_l
# ctx_id=-1 forces CPU even if CUDA is available (stable for inference workers)
print("[FACE] Loading InsightFace detector...")
face_detector = FaceAnalysis(
    name="buffalo_sc",
    allowed_modules=["detection"],   # skip recognition, landmark, etc.
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if DEVICE == "cuda"
              else ["CPUExecutionProvider"]
)
face_detector.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
print("[FACE] Ready.")


# ── HELPERS ─────────────────────────────────────────────────────────

def normalize_label(raw_label: str) -> str:
    """
    Normalize raw HuggingFace label to 'fake' or 'real'.
    Handles: 'Fake', 'FAKE', 'deepfake', 'Real', 'REAL', etc.
    """
    label = raw_label.lower().strip()
    if any(word in label for word in ["fake", "deepfake", "synthetic", "manipulated"]):
        return "fake"
    return "real"


def extract_face(image: Image.Image) -> tuple[Image.Image, bool]:
    """
    Detect and crop the largest face from a PIL image.

    Returns:
        (cropped_image, face_found)
        If no face found, returns (original_image, False)
    """
    img_array = np.array(image.convert("RGB"))
    faces     = face_detector.get(img_array)

    if not faces:
        return image, False

    # Use the largest detected face
    largest = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    x1, y1, x2, y2 = map(int, largest.bbox)
    w, h = x2 - x1, y2 - y1

    # Add padding
    pad_x = int(w * FACE_PADDING)
    pad_y = int(h * FACE_PADDING)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image.width,  x2 + pad_x)
    y2 = min(image.height, y2 + pad_y)

    return image.crop((x1, y1, x2, y2)), True


def frequency_artifact_score(image: Image.Image) -> float:
    """
    Secondary deepfake signal using FFT frequency analysis.
    Deepfake generators leave high-frequency artifacts invisible to the eye.

    Returns a suspicion score 0.0–1.0 (higher = more suspicious).
    """
    gray      = np.array(image.convert("L").resize((128, 128)), dtype=np.float32)
    f         = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log(np.abs(f) + 1)

    h, w      = magnitude.shape
    center    = magnitude[h//4 : 3*h//4, w//4 : 3*w//4]

    high_freq  = magnitude.sum() - center.sum()
    total      = magnitude.sum() + 1e-8
    ratio      = high_freq / total

    # Normalize — tuned empirically, adjust after running test_accuracy.py
    score = float(np.clip((ratio - 0.30) / 0.40, 0.0, 1.0))
    return round(score, 4)


# ── CORE INFERENCE ───────────────────────────────────────────────────

def run_inference(image: Image.Image) -> dict:
    """
    Run deepfake detection on a PIL Image.

    Returns dict with:
        result       — 'fake' | 'real' | 'uncertain'
        confidence   — float 0–1
        face_found   — bool
        freq_score   — float 0–1 (secondary signal)
        label_raw    — original model label string
    """
    face_crop, face_found = extract_face(image)

    # ── Primary model inference ──
    inputs = processor(images=face_crop, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs      = torch.nn.functional.softmax(outputs.logits, dim=1)
    conf, pred = torch.max(probs, dim=1)

    raw_label  = model.config.id2label[pred.item()]
    label      = normalize_label(raw_label)
    base_conf  = float(conf.item())

    # ── Secondary: frequency artifact score ──
    freq_score = frequency_artifact_score(face_crop)

    # ── Ensemble: weight model heavily, freq as tiebreaker ──
    if label == "fake":
        # Freq score boosts confidence when it agrees (high = suspicious)
        adjusted_conf = base_conf * 0.82 + freq_score * 0.18
    else:
        # Freq score reduces confidence when it disagrees (high = suspicious despite "real")
        adjusted_conf = base_conf * 0.82 + (1.0 - freq_score) * 0.18

    adjusted_conf = round(float(np.clip(adjusted_conf, 0.0, 1.0)), 4)

    # ── Apply confidence threshold ──
    if adjusted_conf < CONFIDENCE_THRESHOLD:
        result = "uncertain"
    else:
        result = label

    return {
        "result":     result,
        "confidence": adjusted_conf,
        "face_found": face_found,
        "freq_score": freq_score,
        "label_raw":  raw_label,
    }


def run_inference_from_path(file_path: str) -> dict:
    """
    Load an image from disk and run inference.
    Raises ValueError on missing file or unreadable image.
    """
    file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    try:
        image = Image.open(file_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot read image: {e}")

    return run_inference(image)