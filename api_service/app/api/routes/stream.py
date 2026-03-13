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
    print("[WS] Client connected")

    # Stop any existing stream
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
        # ── Wait for start message ──
        init         = await websocket.receive_text()
        start_msg    = json.loads(init)

        if start_msg.get("type") != "start" or not start_msg.get("url"):
            await send({"type": "error", "message": "First message must be {type: 'start', url: '...'}"})
            return

        url = start_msg["url"].strip()

        # ── Tell inference service to start ──
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
            await send({"type": "error", "message": "Cannot reach inference service on port 8001"})
            return
        except httpx.TimeoutException:
            await send({"type": "error", "message": "Inference service timed out on start"})
            return

        # ── FIX 1: send "started" not "status" ──
        # The frontend waits for {"type": "started"} before showing the canvas.
        await send({"type": "started"})
        print(f"[WS] Stream started: {url[:60]}")

        loop = asyncio.get_running_loop()

        # ── FIX 2: listen for stop messages concurrently ──
        # Previously the Redis poll loop blocked forever and never read
        # incoming WebSocket messages — so clicking Stop did nothing.
        # Now two coroutines run concurrently:
        #   - redis_poller: reads results from Redis → forwards to browser
        #   - ws_listener:  reads incoming messages → handles stop
        # Either one finishing cancels the other.

        async def redis_poller():
            """Read results from Redis and forward to browser."""
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
                    await asyncio.sleep(1)

        async def ws_listener():
            """Listen for stop or other messages from the browser."""
            while not stop_event.is_set():
                try:
                    raw = await websocket.receive_text()
                    msg = json.loads(raw)

                    if msg.get("type") == "stop":
                        print("[WS] Stop requested by client")
                        stop_event.set()
                        return

                except WebSocketDisconnect:
                    print("[WS] Client disconnected")
                    stop_event.set()
                    return
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    print(f"[WS] Listener error: {e}")
                    stop_event.set()
                    return

        # Run both concurrently — first one to finish cancels the other
        poller_task   = asyncio.create_task(redis_poller())
        listener_task = asyncio.create_task(ws_listener())

        done, pending = await asyncio.wait(
            [poller_task, listener_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel whatever is still running
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Always stop inference on exit
        stop_event.set()
        await _notify_inference_stop()
        await send({"type": "stopped"})
        print("[WS] Stream ended cleanly")

    except WebSocketDisconnect:
        print("[WS] Client disconnected before stream started")
        stop_event.set()
        await _notify_inference_stop()

    except json.JSONDecodeError:
        await send({"type": "error", "message": "Invalid JSON in start message"})

    except Exception as e:
        print(f"[WS] Unexpected error: {e}")
        stop_event.set()
        await _notify_inference_stop()