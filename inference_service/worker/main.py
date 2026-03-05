import os
import time
import json
import asyncio
import threading
import redis
import torch
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification
from fastapi import FastAPI
from pydantic import BaseModel
from stream_processor import stream_analysis_loop

# ─── CONFIG ──────────────────────────────────────────────────────────
REDIS_HOST      = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT      = int(os.getenv("REDIS_PORT", 6379))
QUEUE_NAME      = "analysis_jobs"
RESULT_QUEUE    = "analysis_results"
STREAM_RESULTS  = "stream_results"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME      = "prithivMLmods/Deep-Fake-Detector-Model"

# ─── REDIS ───────────────────────────────────────────────────────────
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=0,
    decode_responses=True
)

# ─── MODEL (loads once at startup) ───────────────────────────────────
print(f"[MODEL] Loading on {DEVICE}...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model     = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
print(f"[MODEL] Ready on {DEVICE}")

# ─── FACE DETECTOR ───────────────────────────────────────────────────
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ─── INFERENCE HELPERS ───────────────────────────────────────────────
def extract_face(image: Image.Image) -> Image.Image:
    """Crop to largest detected face. Falls back to full image."""
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
    """Run CNN on a PIL image. Returns (label, confidence)."""
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
    """Load image from disk and run inference."""
    file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    try:
        image = Image.open(file_path).convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"Cannot read image: {e}")

    return run_inference(image)


# ─── REDIS JOB WORKER (runs in background thread) ────────────────────
def job_worker_loop():
    """Continuously polls Redis for analysis jobs and processes them."""
    print("[WORKER] Job worker started. Waiting for jobs...")

    while True:
        try:
            job_data = redis_client.brpop(QUEUE_NAME, timeout=5)
            if not job_data:
                continue

            _, message      = job_data
            job_message     = json.loads(message)
            job_id          = job_message["job_id"]
            file_path       = job_message["file_path"]
            media_type      = job_message.get("media_type", "image")

            print(f"[WORKER] Processing job {job_id} (type: {media_type})")
            start_time = time.time()

            try:
                if media_type == "video":
                    # Import here to avoid circular at module level
                    from video_processor import run_video_inference
                    agg  = run_video_inference(file_path)
                    label      = agg["verdict"]
                    confidence = agg["confidence"]
                    extra = {
                        "fake_frame_ratio":      agg["fake_frame_ratio"],
                        "total_frames_analyzed": agg["total_frames_analyzed"],
                    }
                else:
                    label, confidence = run_inference_from_path(file_path)
                    extra = {}

                inference_time = int((time.time() - start_time) * 1000)
                result_message = {
                    "job_id":           job_id,
                    "status":           "completed",
                    "result":           label,
                    "confidence":       confidence,
                    "inference_time_ms": inference_time,
                    **extra
                }
                print(f"[WORKER] Done {job_id} → {label} ({confidence}) in {inference_time}ms")

            except Exception as e:
                print(f"[WORKER] Inference failed for {job_id}: {e}")
                result_message = {
                    "job_id":     job_id,
                    "status":     "failed",
                    "result":     None,
                    "confidence": None,
                    "error":      str(e)
                }

            redis_client.lpush(RESULT_QUEUE, json.dumps(result_message))

        except json.JSONDecodeError as e:
            print(f"[WORKER] Bad JSON in job: {e}")
        except redis.RedisError as e:
            print(f"[WORKER] Redis error: {e}. Retrying in 3s...")
            time.sleep(3)
        except Exception as e:
            print(f"[WORKER] Unexpected error: {e}")


# ─── FASTAPI APP ──────────────────────────────────────────────────────
app = FastAPI(title="VeriSight Inference Service")

# Stream state
_stream_stop_event: asyncio.Event | None = None
_stream_task:       asyncio.Task  | None = None


class StreamStartRequest(BaseModel):
    url: str


async def _send_result_to_redis(payload: dict):
    """Callback used by stream_analysis_loop to push results."""
    redis_client.lpush(STREAM_RESULTS, json.dumps(payload))


@app.post("/stream/start", tags=["Stream"])
async def start_stream(body: StreamStartRequest):
    global _stream_stop_event, _stream_task

    # Cancel existing stream if running
    if _stream_stop_event:
        _stream_stop_event.set()
    if _stream_task and not _stream_task.done():
        _stream_task.cancel()
        try:
            await _stream_task
        except asyncio.CancelledError:
            pass

    _stream_stop_event = asyncio.Event()
    _stream_task = asyncio.create_task(
        stream_analysis_loop(body.url, _send_result_to_redis, _stream_stop_event)
    )

    print(f"[STREAM] Started for URL: {body.url[:60]}")
    return {"status": "started", "url": body.url}


@app.post("/stream/stop", tags=["Stream"])
async def stop_stream():
    global _stream_stop_event, _stream_task

    if _stream_stop_event:
        _stream_stop_event.set()
    if _stream_task and not _stream_task.done():
        _stream_task.cancel()
        try:
            await _stream_task
        except asyncio.CancelledError:
            pass

    print("[STREAM] Stopped")
    return {"status": "stopped"}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model":  MODEL_NAME,
    }


# ─── STARTUP: launch job worker in background thread ─────────────────
@app.on_event("startup")
def startup():
    t = threading.Thread(target=job_worker_loop, daemon=True)
    t.start()
    print("[STARTUP] Job worker thread started")