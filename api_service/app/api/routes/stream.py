import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from inference_service import stream_analysis_loop

router = APIRouter()

# Global state — one stream at a time
current_stop_event: asyncio.Event | None = None


@router.websocket("/ws/stream")
async def stream_detection(websocket: WebSocket):
    global current_stop_event

    await websocket.accept()
    print("[WS] Stream client connected")

    # Stop any existing stream
    if current_stop_event:
        current_stop_event.set()

    stop_event = asyncio.Event()
    current_stop_event = stop_event

    async def send(payload: dict):
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception:
            pass  # client disconnected mid-send

    try:
        # First message must be the stream URL
        init = await websocket.receive_text()
        message = json.loads(init)

        if message.get("type") != "start" or not message.get("url"):
            await send({"type": "error", "message": "Send {type: 'start', url: '...'} to begin"})
            return

        url = message["url"].strip()
        stream_type = _detect_stream_type(url)
        await send({"type": "status", "message": f"Starting {stream_type} stream analysis..."})

        # Run the analysis loop as a background task
        # so we can also listen for stop messages
        analysis_task = asyncio.create_task(
            stream_analysis_loop(url, send, stop_event)
        )

        # Listen for stop message from client
        while not stop_event.is_set():
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                data = json.loads(msg)
                if data.get("type") == "stop":
                    stop_event.set()
                    break
            except asyncio.TimeoutError:
                continue  # no message yet, keep looping
            except Exception:
                break

        stop_event.set()
        await analysis_task

    except WebSocketDisconnect:
        print("[WS] Stream client disconnected")
        stop_event.set()
    except Exception as e:
        print(f"[WS] Stream error: {e}")
        stop_event.set()


def _detect_stream_type(url: str) -> str:
    if "youtube.com" in url or "youtu.be" in url:
        return "YouTube"
    if url.startswith("rtmp://"):
        return "RTMP"
    if url.startswith("rtsp://"):
        return "RTSP / IP Camera"
    if url.endswith(".m3u8"):
        return "HLS"
    return "HTTP"