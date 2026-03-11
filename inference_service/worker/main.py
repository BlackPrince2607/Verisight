import os
import time
import redis
import json
import asyncio
import threading
from fastapi import FastAPI
from pydantic import BaseModel
from inference import run_inference_from_path, DEVICE, MODEL_NAME
from stream_processor import stream_analysis_loop

# ─── REDIS ───────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=0,
    decode_responses=True
)

QUEUE_NAME   = "analysis_jobs"
RESULT_QUEUE = "analysis_results"


# ─── JOB WORKER (background thread) ──────────────────────
def job_worker_loop():
    print("[WORKER] Job worker started. Waiting for jobs...")

    while True:
        try:
            job_data = redis_client.brpop(QUEUE_NAME, timeout=5)
            if not job_data:
                continue

            _, message  = job_data
            job_message = json.loads(message)
            job_id      = job_message["job_id"]
            file_path   = job_message["file_path"]
            media_type  = job_message.get("media_type", "image")

            print(f"[WORKER] Processing {job_id} ({media_type})")
            start_time = time.time()

            try:
                if media_type == "video":
                    from video_processor import run_video_inference
                    agg        = run_video_inference(file_path)
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
                    "job_id":            job_id,
                    "status":            "completed",
                    "result":            label,
                    "confidence":        confidence,
                    "inference_time_ms": inference_time,
                    **extra
                }
                print(f"[WORKER] {job_id} → {label} ({confidence}) in {inference_time}ms")

            except Exception as e:
                print(f"[WORKER] Failed {job_id}: {e}")
                result_message = {
                    "job_id":     job_id,
                    "status":     "failed",
                    "result":     None,
                    "confidence": None,
                    "error":      str(e)
                }

            redis_client.lpush(RESULT_QUEUE, json.dumps(result_message))

        except json.JSONDecodeError as e:
            print(f"[WORKER] Bad JSON: {e}")
        except redis.RedisError as e:
            print(f"[WORKER] Redis error: {e}. Retrying in 3s...")
            time.sleep(3)
        except Exception as e:
            print(f"[WORKER] Unexpected: {e}")


# ─── FASTAPI ──────────────────────────────────────────────
app = FastAPI(title="VeriSight Inference Service")

_stream_stop_event: asyncio.Event | None = None
_stream_task:       asyncio.Task  | None = None


class StreamStartRequest(BaseModel):
    url: str


async def _push_to_redis(payload: dict):
    redis_client.lpush("stream_results", json.dumps(payload))


@app.post("/stream/start", tags=["Stream"])
async def start_stream(body: StreamStartRequest):
    global _stream_stop_event, _stream_task

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
        stream_analysis_loop(body.url, _push_to_redis, _stream_stop_event)
    )
    print(f"[STREAM] Started: {body.url[:60]}")
    return {"status": "started"}


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


@app.on_event("startup")
def startup():
    t = threading.Thread(target=job_worker_loop, daemon=True)
    t.start()
    print("[STARTUP] Job worker thread running")