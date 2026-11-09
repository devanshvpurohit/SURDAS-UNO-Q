"""
HAL Camera Provider for SURDAS on Arduino UNO Q.
Provides V4L2/USB camera auto-discovery, robust acquisition, and mock fallback.
"""

import abc
import os
import glob
import time
import threading
import logging
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np

logger = logging.getLogger("surdas.hal.camera")


def scan_available_cameras(probe_indices: bool = False) -> List[Dict[str, Any]]:
    """
    Scan the system for available camera devices.
    Prioritizes Linux V4L2 devices (/dev/video*).
    Returns list of discovered camera descriptors.
    """
    available = []
    
    # 1. Check /dev/video* paths if on Linux
    linux_devices = sorted(glob.glob("/dev/video*"))
    for dev_path in linux_devices:
        try:
            # Check device permissions and openability
            if os.access(dev_path, os.R_OK):
                available.append({
                    "id": dev_path,
                    "type": "v4l2",
                    "name": f"V4L2 Device ({dev_path})"
                })
        except Exception:
            pass

    # 2. If no V4L2 devices found, probe camera indices (macOS / Windows / USB index)
    if not available:
        for idx in (0, 1, 2):
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        available.append({
                            "id": idx,
                            "type": "index",
                            "name": f"Camera Index {idx}"
                        })
            except Exception:
                pass

    return available


class CameraProvider(abc.ABC):
    """Abstract interface for camera capture providers."""

    @abc.abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Return the latest acquired frame as BGR numpy array, or None if unavailable."""
        pass

    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Return True if camera hardware is connected and delivering frames."""
        pass

    @abc.abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return device status dictionary (fps, resolution, dropped frames, etc.)."""
        pass

    @abc.abstractmethod
    def release(self) -> None:
        """Release underlying hardware resources and stop threads."""
        pass


class V4L2CameraProvider(CameraProvider):
    """
    OpenCV V4L2 / USB camera capture provider.
    Runs a non-blocking background thread to capture frames, avoiding latency build-up.
    """

    def __init__(self, device: Any = 0, target_fps: float = 15.0, width: int = 640, height: int = 480):
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        self.device = device
        self.target_fps = target_fps
        self.width = width
        self.height = height
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.lock = threading.RLock()
        self.stopped = False
        self.connected = False
        self.last_frame_time = 0.0
        self.fps_actual = 0.0
        self.frame_count = 0
        self.error_count = 0

        self.thread = None
        opened = self._init_capture()
        if opened:
            self.thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraCaptureThread")
            self.thread.start()

    def _init_capture(self) -> bool:
        try:
            logger.info(f"Connecting to camera device: {self.device}")
            self.cap = cv2.VideoCapture(self.device)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                self.connected = True
                return True
            else:
                logger.warning(f"Failed to open camera device: {self.device}")
                self.connected = False
                return False
        except Exception as e:
            logger.error(f"Error opening camera {self.device}: {e}")
            self.connected = False
            return False

    def _capture_loop(self) -> None:
        last_t = time.time()
        while not self.stopped:
            if not self.connected or self.cap is None or not self.cap.isOpened():
                for _ in range(10):
                    if self.stopped:
                        break
                    time.sleep(0.1)
                if self.stopped:
                    break
                # Only attempt reconnect if device is numeric index or existing filesystem path
                if isinstance(self.device, int) or (isinstance(self.device, str) and os.path.exists(str(self.device))):
                    self._init_capture()
                continue

            try:
                ret, frame = self.cap.read()
                now = time.time()
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame
                        self.connected = True
                        self.last_frame_time = now
                        self.frame_count += 1
                        dt = now - last_t
                        if dt > 0:
                            self.fps_actual = 0.9 * self.fps_actual + 0.1 * (1.0 / dt)
                        last_t = now
                else:
                    self.error_count += 1
                    if self.error_count > 10:
                        self.connected = False
                    time.sleep(0.05)
            except Exception as e:
                self.error_count += 1
                self.connected = False
                time.sleep(0.1)

            # Throttle to avoid burning 100% CPU on capture thread
            time.sleep(max(0.001, 1.0 / (self.target_fps * 1.5)))

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def is_connected(self) -> bool:
        with self.lock:
            # Check if frame is fresh (< 2 seconds old)
            fresh = (time.time() - self.last_frame_time) < 2.0
            return self.connected and fresh and (self.frame is not None)

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "device": str(self.device),
                "connected": self.is_connected(),
                "fps": round(self.fps_actual, 1),
                "frame_count": self.frame_count,
                "error_count": self.error_count,
                "is_mock": False
            }

    def release(self) -> None:
        self.stopped = True
        self.connected = False
        if hasattr(self, "thread") and self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.3)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


class MockCameraProvider(CameraProvider):
    """
    Mock camera provider for testing and environments without physical cameras.
    Generates synthetic frames with timestamp and animated test pattern.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: float = 15.0):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        self.stopped = False
        self.last_time = time.time()

    def get_frame(self) -> Optional[np.ndarray]:
        if self.stopped:
            return None
        self.frame_count += 1
        now = time.time()
        self.last_time = now

        # Create canvas
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background gradient
        frame[:, :, 0] = 30
        frame[:, :, 1] = 40
        frame[:, :, 2] = 50

        # Draw ground plane
        cv2.rectangle(frame, (0, int(self.height * 0.6)), (self.width, self.height), (40, 50, 40), -1)

        # Draw simulated obstacle / test target
        cx = int((self.width // 2) + 120 * np.sin(self.frame_count * 0.05))
        cy = int(self.height * 0.65)
        cv2.rectangle(frame, (cx - 40, cy - 60), (cx + 40, cy + 60), (0, 165, 255), -1)
        cv2.putText(frame, "OBSTACLE", (cx - 35, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Overlay metadata
        cv2.putText(frame, "SURDAS UNO Q [SIMULATED CAMERA]", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
        cv2.putText(frame, f"Frame: {self.frame_count} | Time: {time.strftime('%H:%M:%S')}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame

    def is_connected(self) -> bool:
        return not self.stopped

    def get_status(self) -> Dict[str, Any]:
        return {
            "device": "MOCK_SYNTHETIC",
            "connected": True,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "error_count": 0,
            "is_mock": True
        }

    def release(self) -> None:
        self.stopped = True


def create_camera_provider(device: Any = None, allow_mock: bool = True) -> CameraProvider:
    """
    Factory to create best available camera provider.
    Scans for hardware camera, falls back to MockCameraProvider if unavailable.
    """
    if str(device).lower() == "mock":
        return MockCameraProvider()

    if device is not None and str(device).lower() != "auto":
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        provider = V4L2CameraProvider(device=device)
        for _ in range(15):
            if provider.is_connected():
                return provider
            time.sleep(0.1)
        provider.release()
        if not allow_mock:
            raise RuntimeError(f"Unable to connect to camera device: {device}")

    # Auto-scan hardware
    cams = scan_available_cameras()
    if cams:
        chosen = cams[0]["id"]
        logger.info(f"Auto-selected detected camera: {chosen}")
        provider = V4L2CameraProvider(device=chosen)
        for _ in range(15):
            if provider.is_connected():
                return provider
            time.sleep(0.1)
        provider.release()

    if allow_mock:
        logger.info("No physical camera detected. Initializing MockCameraProvider.")
        return MockCameraProvider()
    
    raise RuntimeError("No camera device found and allow_mock is False.")
