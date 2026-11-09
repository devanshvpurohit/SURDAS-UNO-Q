import asyncio
import threading
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import os
import json
import urllib.request
import time
from typing import Optional, Callable
import cv2

app = FastAPI(title="SURDAS UNO Q Telemetry Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

connected_clients: set = set()
_loop = None
_started = False
_lock = threading.Lock()

# Global providers registered by brain or standalone runner
_frame_getter: Optional[Callable[[], Optional[bytes]]] = None
_imu_provider = None
_bridge = None
_last_imu_reading = None


def set_frame_getter(fn: Callable[[], Optional[bytes]]):
    global _frame_getter
    _frame_getter = fn


def set_imu_provider(provider):
    global _imu_provider
    _imu_provider = provider


def set_bridge(bridge):
    global _bridge
    _bridge = bridge


# ──────────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            text = await websocket.receive_text()
            # Handle inbound commands from dashboard
            try:
                cmd_data = json.loads(text)
                cmd_type = cmd_data.get("type")
                if cmd_type == "emergency_stop":
                    if _bridge:
                        _bridge.trigger_emergency_stop()
                    broadcast_event("emergency_stop_triggered", {"time": time.time()})
                elif cmd_type == "test_fall":
                    if _imu_provider:
                        _imu_provider.trigger_simulated_fall()
            except Exception:
                pass
    except Exception:
        connected_clients.discard(websocket)


# ──────────────────────────────────────────────────
# IMU / Fall Detection Endpoint
# ──────────────────────────────────────────────────
@app.get("/imu")
def get_imu():
    """
    Returns standardized IMU & fall detection telemetry.
    Replaces legacy ESP32 endpoint (http://192.168.4.2/imu).
    Clearly specifies whether source is 'HARDWARE' or 'SIMULATION'.
    """
    global _imu_provider
    if _imu_provider is not None:
        reading = _imu_provider.get_reading()
        if reading is not None:
            return reading.to_dict()

    # Fallback status when no provider is active
    return {
        "ax": 0.0,
        "ay": 0.0,
        "az": 1.0,
        "acceleration": 1.0,
        "pitch": 0.0,
        "roll": 0.0,
        "falls": 0,
        "state": "UNAVAILABLE",
        "source": "UNAVAILABLE",
        "timestamp": time.time(),
        "mpu": "NONE"
    }


@app.post("/test_fall")
def trigger_test_fall():
    """Trigger a simulated fall for dashboard verification."""
    global _imu_provider
    if _imu_provider is not None:
        _imu_provider.trigger_simulated_fall()
        return {"status": "ok", "message": "Simulated fall triggered"}
    return {"status": "error", "message": "IMU provider not initialized"}


@app.post("/emergency_stop")
def emergency_stop():
    """Deterministic hardware emergency stop."""
    global _bridge
    if _bridge:
        _bridge.trigger_emergency_stop()
    broadcast_event("emergency_stop_triggered", {"time": time.time()})
    return {"status": "ok", "emergency_stop": True}


# ──────────────────────────────────────────────────
# Video Stream Endpoint (MJPEG)
# ──────────────────────────────────────────────────
def _mjpeg_generator():
    while True:
        if _frame_getter is not None:
            frame_bytes = _frame_getter()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.066)  # ~15 FPS


@app.get("/video_feed")
def video_feed():
    """Live MJPEG video stream from camera provider."""
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ──────────────────────────────────────────────────
# Location & Health Endpoints
# ──────────────────────────────────────────────────
@app.get("/location")
def get_location():
    try:
        req = urllib.request.Request(
            "http://ipinfo.io/json",
            headers={"User-Agent": "SURDAS/1.0"}
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "loc" in data:
                lat, lng = data["loc"].split(",")
                return {
                    "lat": float(lat),
                    "lng": float(lng),
                    "city": data.get("city", "Unknown"),
                    "region": data.get("region", ""),
                    "country": data.get("country", ""),
                    "org": data.get("org", ""),
                }
    except Exception:
        pass
    return {"lat": 17.3616, "lng": 78.4747, "city": "Hyderabad", "region": "TS", "country": "IN", "org": "SURDAS"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "clients": len(connected_clients),
        "imu_connected": _imu_provider.is_connected() if _imu_provider else False,
        "imu_simulated": _imu_provider.is_simulated() if _imu_provider else False,
        "platform": "Arduino UNO Q"
    }


# ──────────────────────────────────────────────────
# Mount Caregiver Dashboard Static Build if present
# ──────────────────────────────────────────────────
_dashboard_dist = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caregiver_dashboard", "dist")
if os.path.isdir(_dashboard_dist):
    app.mount("/", StaticFiles(directory=_dashboard_dist, html=True), name="dashboard_static")


@app.on_event("startup")
async def startup_event():
    global _loop
    _loop = asyncio.get_running_loop()
    print("[TELEMETRY] Dashboard server ready — ws://localhost:8000/ws")


# ──────────────────────────────────────────────────
# Broadcast helper
# ──────────────────────────────────────────────────
def broadcast_event(event_type: str, payload: dict):
    """Send an event to all connected dashboard clients."""
    if not _loop:
        return
    message = json.dumps({"type": event_type, "data": payload})

    async def _send():
        dead = set()
        for client in list(connected_clients):
            try:
                await client.send_text(message)
            except Exception:
                dead.add(client)
        connected_clients.difference_update(dead)

    asyncio.run_coroutine_threadsafe(_send(), _loop)


# ──────────────────────────────────────────────────
# Start server
# ──────────────────────────────────────────────────
def start_telemetry(port: int = 8000):
    global _started
    with _lock:
        if _started:
            return
        _started = True

    def _run():
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            log_level="error",
            loop="none",
        )
        server = uvicorn.Server(config)
        asyncio.run(server.serve())

    t = threading.Thread(target=_run, daemon=True, name="TelemetryServer")
    t.start()
    print(f"[TELEMETRY] Unified dashboard & sensor server starting on http://0.0.0.0:{port}")
