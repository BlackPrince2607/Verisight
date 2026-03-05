import cv2
import base64
import asyncio
import subprocess
import json
from collections import deque
from PIL import Image, UnidentifiedImageError
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-Model"
FRAME_INTERVAL = 10       # analyze every 10th frame
SMOOTHING_WINDOW = 5      # rolling average over 5 results

print(f"[STREAM] Loading model on {DEVICE}...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
print("[STREAM] Model ready.")


def resolve_stream_url(url: str) -> str:
    """
    Resolve the actual stream URL.
    - YouTube: use yt-dlp to extract direct stream URL
    - RTMP / RTSP / HTTP: return as-is
    """
    if "youtube.com" in url or "youtu.be" in url:
        print(f"[STREAM] Resolving YouTube URL via yt-dlp...")
        result = subprocess.run(
            ["yt-dlp", "-g", "--no-playlist", "-f", "best[ext=mp4]/best", url],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise ValueError(f"yt-dlp failed: {result.stderr.strip()}")
        resolved = result.stdout.strip().split("\n")[0]
        print(f"[STREAM] Resolved to: {resolved[:80]}...")
        return resolved

    # RTMP, RTSP, IP cam — OpenCV handles directly
    return url


def run_inference_on_frame(frame_rgb: Image.Image) -> tuple[str, float]:
    inputs = processor(images=frame_rgb, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)
    label = model.config.id2label[predicted_class.item()]
    return label, round(float(confidence.item()), 4)


def compute_smoothed_verdict(history: deque) -> dict:
    fake_score = real_score = 0.0
    for label, confidence in history:
        if "fake" in label.lower():
            fake_score += confidence
        else:
            real_score += confidence

    total = fake_score + real_score
    if total == 0:
        return {"verdict": "WARMING UP", "confidence": 0.0, "fake_ratio": 0.0}

    fake_ratio = fake_score / total
    verdict = "FAKE" if fake_ratio > 0.5 else "REAL"
    smoothed_conf = fake_ratio if verdict == "FAKE" else (real_score / total)

    return {
        "verdict": verdict,
        "confidence": round(smoothed_conf, 4),
        "fake_ratio": round(fake_ratio, 4),
    }


def frame_to_base64(frame_bgr) -> str:
    """Convert OpenCV BGR frame to base64 JPEG for sending over WebSocket."""
    _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buffer).decode("utf-8")


async def stream_analysis_loop(url: str, send_callback, stop_event: asyncio.Event):
    """
    Core loop — reads stream, runs inference every Nth frame,
    calls send_callback with results + frame snapshot.
    """
    try:
        resolved_url = resolve_stream_url(url)
    except Exception as e:
        await send_callback({"type": "error", "message": f"Could not resolve stream: {e}"})
        return

    cap = cv2.VideoCapture(resolved_url)

    if not cap.isOpened():
        await send_callback({"type": "error", "message": "Could not open stream. Check the URL."})
        return

    await send_callback({"type": "status", "message": "Stream opened. Analyzing..."})
    print(f"[STREAM] Capture started: {resolved_url[:80]}")

    history = deque(maxlen=SMOOTHING_WINDOW)
    frame_index = 0
    analyzed = 0

    try:
        while not stop_event.is_set():
            ret, frame_bgr = cap.read()

            if not ret:
                await send_callback({"type": "error", "message": "Stream ended or disconnected."})
                break

            if frame_index % FRAME_INTERVAL == 0:
                try:
                    # Convert BGR → RGB for PIL
                    frame_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    label, confidence = run_inference_on_frame(frame_rgb)
                    history.append((label, confidence))
                    smoothed = compute_smoothed_verdict(history)
                    analyzed += 1

                    # Encode frame snapshot as base64
                    frame_b64 = frame_to_base64(frame_bgr)

                    payload = {
                        "type": "result",
                        "frame": f"data:image/jpeg;base64,{frame_b64}",
                        "raw": {"label": label, "confidence": confidence},
                        "smoothed": smoothed,
                        "frames_analyzed": analyzed,
                    }
                    await send_callback(payload)

                except Exception as e:
                    print(f"[STREAM] Frame inference error: {e}")

            frame_index += 1

            # Yield control so WebSocket stays responsive
            await asyncio.sleep(0)

    finally:
        cap.release()
        print("[STREAM] Capture released.")