import asyncio
import json
import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.api.core.redis_client import redis_client

router = APIRouter()

INFERENCE_SERVICE_URL = "http://localhost:8001"

current_stop_event: asyncio.Event | None = None


async def _notify_inference_stop():
    """Tell inference service to stop the current stream."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{INFERENCE_SERVICE_URL}/stream/stop", timeout=5)
    except Exception as e:
        print(f"[STREAM] Could not notify inference service to stop: {e}")


@router.websocket("/ws/stream")
async def stream_detection(websocket: WebSocket):
    global current_stop_event

    await websocket.accept()
    print("[WS] Stream client connected")

    # Stop any existing stream before starting a new one
    if current_stop_event:
        current_stop_event.set()
        await _notify_inference_stop()

    stop_event = asyncio.Event()
    current_stop_event = stop_event

    async def send(payload: dict):
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception:
            pass

    try:
        # Wait for start message
        init = await websocket.receive_text()
        start_message = json.loads(init)

        if start_message.get("type") != "start" or not start_message.get("url"):
            await send({
                "type": "error",
                "message": "First message must be {type: 'start', url: '...'}"
            })
            return

        url = start_message["url"].strip()

        # Tell inference service to start
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{INFERENCE_SERVICE_URL}/stream/start",
                    json={"url": url},
                    timeout=10
                )
                if resp.status_code != 200:
                    await send({"type": "error", "message": "Inference service rejected the request"})
                    return
        except httpx.ConnectError:
            await send({"type": "error", "message": "Cannot reach inference service — is it running on port 8001?"})
            return
        except httpx.TimeoutException:
            await send({"type": "error", "message": "Inference service timed out"})
            return

        await send({"type": "status", "message": f"Stream started. Analyzing..."})

        # Read results from Redis → forward to browser
        loop = asyncio.get_running_loop()   # ✅ correct for 3.10+

        while not stop_event.is_set():
            try:
                result_data = await loop.run_in_executor(
                    None,
                    lambda: redis_client.brpop("stream_results", timeout=2)
                )

                if result_data:
                    _, result_str = result_data
                    payload = json.loads(result_str)
                    await send(payload)

            except json.JSONDecodeError as e:
                print(f"[STREAM] Bad JSON from Redis: {e}")
            except Exception as e:
                print(f"[STREAM] Redis read error: {e}")
                await asyncio.sleep(1)  # brief pause before retrying

        # Graceful stop
        await _notify_inference_stop()
        print("[WS] Stream stopped cleanly")

    except WebSocketDisconnect:
        print("[WS] Client disconnected abruptly")
        stop_event.set()
        await _notify_inference_stop()   # ✅ always stop inference on disconnect

    except json.JSONDecodeError:
        await send({"type": "error", "message": "Invalid JSON in start message"})

    except Exception as e:
        print(f"[WS] Unexpected error: {e}")
        stop_event.set()
        await _notify_inference_stop()