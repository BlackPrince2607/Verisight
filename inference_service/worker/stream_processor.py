import cv2
import base64
import asyncio
import subprocess
import queue as thread_queue
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from PIL import Image
from inference import run_inference

# ── TUNING CONSTANTS ─────────────────────────────────────────────────
FRAME_INTERVAL   = 1          # every Nth frame from FFmpeg output (FFmpeg fps filter handles throttling)
SMOOTHING_WINDOW = 15          # rolling window for verdict smoothing
FRAME_BATCH      = 12          # frames per inference batch — bigger = better GPU util
FRAME_WIDTH      = 640        # reduced from 640 — big compute win, face detection still reliable
FRAME_HEIGHT     = 360         # reduced from 360

# max_workers=3: allows reader thread + 2 overlapping inference calls
# GPU-bound models still serialize on CUDA, but CPU pre/postprocessing overlaps
_reader_executor = ThreadPoolExecutor(max_workers=1)
_inference_executor = ThreadPoolExecutor(max_workers=3)

# Separate thread-safe queue for reader → main loop handoff
# Using queue.Queue (not asyncio.Queue) avoids run_coroutine_threadsafe overhead per frame
_READER_QUEUE_SIZE = 15


# ── URL RESOLVER ─────────────────────────────────────────────────────

def resolve_stream_url(url: str) -> str:
    """
    Resolve a YouTube or other URL to a direct streamable URL using yt-dlp.
    For live streams this returns an HLS manifest — FFmpeg handles it natively.
    """
    if "youtube.com" in url or "youtu.be" in url:
        print("[STREAM] Resolving YouTube URL via yt-dlp...")
        result = subprocess.run(
            ["yt-dlp", "-g", "--no-playlist", "--js-runtimes", "node", url],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise ValueError(f"yt-dlp failed: {result.stderr.strip()}")

        resolved = result.stdout.strip().split("\n")[0]
        print(f"[STREAM] Resolved (full):\n{resolved}")
        return resolved

    return url


# ── FFMPEG STREAM OPENER ──────────────────────────────────────────────

def open_ffmpeg_stream(url: str) -> subprocess.Popen:
    """
    Open FFmpeg subprocess piping raw RGB24 frames to stdout.
    FFmpeg handles HLS natively. YouTube requires browser User-Agent to avoid 403.

    Low-latency flags:
      -probesize 32 -analyzeduration 0  → skip slow stream analysis on startup
      -fflags nobuffer -flags low_delay → minimize internal buffering
      -threads 2                        → FFmpeg decoder threads
      fps=4 filter                      → throttle output so reader never floods
      -flush_packets 1                  → force immediate stdout flush per frame
      bufsize=0                         → unbuffered pipe read
    """
    is_youtube = "googlevideo.com" in url or "youtube.com" in url or "youtu.be" in url

    cmd = ["ffmpeg", "-loglevel", "error"]

    if is_youtube:
        cmd += [
            "-user_agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "-headers", "Referer: https://www.youtube.com/\r\nOrigin: https://www.youtube.com\r\n",
            "-multiple_requests", "1",
        ]

    cmd += [
        "-probesize", "32",
        "-analyzeduration", "0",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-threads", "2",

        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",

        "-rw_timeout", "15000000",
        "-i", url,

        "-map", "0:v:0",

        "-vf", f"fps=4,scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-an",
        "-f", "rawvideo",
        "-flush_packets", "1",
        "pipe:1"
    ]

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0    # unbuffered — critical, otherwise frames sit in OS pipe buffer
    )


# ── FRAME READER ──────────────────────────────────────────────────────
def read_frame(proc: subprocess.Popen) -> np.ndarray | None:
    frame_size = FRAME_WIDTH * FRAME_HEIGHT * 3
    buffer = bytearray()

    while len(buffer) < frame_size:
        chunk = proc.stdout.read(frame_size - len(buffer))

        # If ffmpeg hasn't produced data yet, wait briefly
        if not chunk:
            if proc.poll() is not None:
                return None  # ffmpeg actually exited
            continue        # otherwise keep waiting

        buffer.extend(chunk)

    frame = np.frombuffer(buffer, dtype=np.uint8)
    return frame.reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))


# ── FRAME ENCODING ────────────────────────────────────────────────────

def frame_to_base64(frame_rgb: np.ndarray) -> str:
    """Encode RGB numpy frame → base64 JPEG. Quality 50: fast, acceptable for deepfake display."""
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _, buffer  = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return base64.b64encode(buffer).decode("utf-8")


# ── SMOOTHING ─────────────────────────────────────────────────────────

def compute_smoothed_verdict(history: deque) -> dict:
    """
    Confidence-weighted vote over rolling window.
    'uncertain' frames excluded — don't count for or against either side.
    """
    confident = [(lbl, conf) for lbl, conf in history if lbl in ("fake", "real")]

    if len(confident) < 2:
        return {"verdict": "uncertain", "confidence": 0.0, "fake_ratio": 0.0, "state": "WARMING_UP"}

    fake_score = sum(conf**2 for lbl, conf in confident if lbl == "fake")
    real_score = sum(conf**2 for lbl, conf in confident if lbl == "real")

    fake_score += len([1 for lbl,_ in confident if lbl=="fake"]) * 0.05
    real_score += len([1 for lbl,_ in confident if lbl=="real"]) * 0.05
    total      = fake_score + real_score    
    fake_ratio = fake_score / total
    verdict    = "fake" if fake_ratio >= 0.5 else "real"
    s_conf     = fake_ratio if verdict == "fake" else (real_score / total)

    return {
        "verdict":    verdict,
        "confidence": round(s_conf, 4),
        "fake_ratio": round(fake_ratio, 4),
        "state":      "LIVE",
    }


# ── BATCH INFERENCE ───────────────────────────────────────────────────

def _run_batch(frames: list[Image.Image]) -> list[dict]:
    """Run inference on a list of PIL frames. Runs in thread executor."""
    results = []
    for frame in frames:
        try:
            results.append(run_inference(frame))
        except Exception as e:
            print(f"[STREAM] Inference error: {e}")
    return results

CONF_THRESHOLD = 0.6

def _aggregate_batch(results: list[dict]) -> tuple[str, float]:
    """Aggregate batch results using weighted confidence averaging."""

    confident = [
        r for r in results
        if r["result"] in ("fake", "real") and r["confidence"] >= CONF_THRESHOLD
    ]

    if not confident:
        return "uncertain", 0.5

    fake_weights = []
    real_weights = []

    for r in confident:
        weight = r["confidence"] ** 2   # square = stronger frames influence more

        if r["result"] == "fake":
            fake_weights.append(weight)
        else:
            real_weights.append(weight)

    fake_score = sum(fake_weights)
    real_score = sum(real_weights)

    total = fake_score + real_score
    if total == 0:
        return "uncertain", 0.5

    fake_ratio = fake_score / total
    real_ratio = real_score / total

    label = "fake" if fake_ratio >= real_ratio else "real"
    confidence = max(fake_ratio, real_ratio)

    return label, round(confidence, 4)


# ── FFMPEG READER THREAD ──────────────────────────────────────────────

def _read_frames_from_ffmpeg(
    proc: subprocess.Popen,
    raw_queue: thread_queue.Queue,   # thread-safe — no asyncio overhead
    stop_event: asyncio.Event,
):
    """
    Runs in a ThreadPoolExecutor thread.
    Reads raw frames from FFmpeg stdout → pushes to thread-safe queue.

    Key optimizations vs previous version:
    - Uses queue.Queue (not asyncio.Queue) — no run_coroutine_threadsafe overhead per frame
    - Drops frames when queue is full — prevents lag buildup when inference is slow
    - No frame.copy() — numpy frombuffer already returns a new array each read
    """
    print("[STREAM] Reader thread started")
    try:
        while not stop_event.is_set():
            print("[STREAM] Waiting for frame data...")
            frame = read_frame(proc)
            if frame is not None:
                print("[STREAM] Frame received from FFmpeg")

            if frame is None:
                if proc.poll() is not None:
                    # FFmpeg really exited
                    raw_queue.put(None)
                    break
                continue

            # Drop frame if queue is full — keeps system real-time
            # Without this, frames accumulate and stream falls behind live
            if not raw_queue.full():
                raw_queue.put(frame)   # no .copy() needed — read_frame returns fresh array

    finally:
        try:
            err = proc.stderr.read().decode("utf-8", errors="ignore")
            if err:
                print("[FFMPEG STDERR]")
                print(err)

            proc.stdout.close()
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass
        print("[STREAM] FFmpeg reader thread exited")


# ── MAIN STREAM LOOP ──────────────────────────────────────────────────

async def stream_analysis_loop(
    url: str,
    send_callback,
    stop_event: asyncio.Event,
):
    # ── Resolve URL ──
    try:
        resolved_url = resolve_stream_url(url)
    except Exception as e:
        await send_callback({"type": "error", "message": f"Could not resolve URL: {e}"})
        return

    # ── Start FFmpeg ──
    try:
        proc = open_ffmpeg_stream(resolved_url)
    except FileNotFoundError:
        await send_callback({"type": "error", "message": "FFmpeg not found. Run: sudo apt install ffmpeg"})
        return
    except Exception as e:
        await send_callback({"type": "error", "message": f"FFmpeg failed to start: {e}"})
        return

    # Send started immediately — HLS takes 15-25s to buffer before first frame
    await send_callback({"type": "started"})
    print(f"[STREAM] FFmpeg started on: {resolved_url[:80]}")
    print(f"[STREAM] FFmpeg PID: {proc.pid}")
    if proc.poll() is not None:
        err = proc.stderr.read().decode("utf-8", errors="ignore")
        print("[FFMPEG ERROR]")
        print(err)

    # ── Thread-safe queue: reader thread → async main loop ──
    # Using queue.Queue avoids per-frame asyncio scheduling overhead
    raw_queue = thread_queue.Queue(maxsize=_READER_QUEUE_SIZE)

    _reader_executor.submit(
    _read_frames_from_ffmpeg, proc, raw_queue, stop_event
    )

    loop         = asyncio.get_running_loop()
    history      = deque(maxlen=SMOOTHING_WINDOW)
    frame_buffer = []
    analyzed     = 0
    last_frame   = None

    try:
        while not stop_event.is_set():
            # Poll thread queue without blocking event loop
            # run_in_executor lets us block on queue.get() in a thread
            try:
                frame = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: raw_queue.get(timeout=30)),
                    timeout=35.0
                )
            except (asyncio.TimeoutError, thread_queue.Empty):
                await send_callback({"type": "error", "message": "Stream timed out — no frames after 30s"})
                break

            if frame is None:
                await send_callback({"type": "error", "message": "Stream ended"})
                break

            last_frame = frame
            frame_buffer.append(Image.fromarray(frame))   # no copy — frame is already unique

            if len(frame_buffer) >= FRAME_BATCH:
                batch = frame_buffer                       # hand off directly
                frame_buffer = []                          # reset without copy

                # display frame — encode before inference so it's not delayed
                display_frame = last_frame                 # no copy needed — not mutated

                results = await loop.run_in_executor(_inference_executor, _run_batch, batch)
                label, confidence = _aggregate_batch(results)

                if label in ("fake", "real"):
                    history.append((label, confidence))

                smoothed = compute_smoothed_verdict(history)
                analyzed += 1

                no_face = sum(1 for r in results if not r.get("face_found", True))
                if no_face:
                    print(f"[STREAM] {no_face}/{len(results)} frames had no face detected")

                await send_callback({
                    "type":  "result",
                    "frame": f"data:image/jpeg;base64,{frame_to_base64(display_frame)}",
                    "raw":   {"result": label, "confidence": confidence},
                    "smoothed":        smoothed,
                    "frames_analyzed": analyzed,
                })

    finally:
        stop_event.set()

        # clear frame buffers
        frame_buffer.clear()
        history.clear()

        # empty queue
        while not raw_queue.empty():
            try:
                raw_queue.get_nowait()
            except:
                break

        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass

        print("[STREAM] Stream loop exited, FFmpeg terminated")