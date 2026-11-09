import os
import sys
import warnings
import argparse

# Suppress background PyTorch MPS & OpenCV warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["YOLO_VERBOSE"] = "False"
os.environ["YOLO_OFFLINE"] = "True"
os.environ["YOLO_SETTINGS_ANALYTICS"] = "False"
os.environ["OFFLINE_MODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCH_HOME"] = os.path.expanduser("~/.cache/torch")

# ── CLI argument parsing (must happen before heavy imports) ──────────────────
_parser = argparse.ArgumentParser(description="SURDAS Assistive Vision Brain for Arduino UNO Q")
_parser.add_argument(
    "--camera", "-c",
    default=None,
    metavar="CAMERA",
    help="Camera device path, index, or 'mock'. Defaults to auto-discovery.",
)
_parser.add_argument(
    "--mock-camera",
    action="store_true",
    help="Force mock camera simulation.",
)
_parser.add_argument(
    "--mock-sensors",
    action="store_true",
    help="Enable simulated IMU sensor data.",
)
_parser.add_argument(
    "--mic",
    default=None,
    metavar="DEVICE",
    help="Audio input device name keyword or index. Auto-detects by default.",
)
_parser.add_argument(
    "--mic-gain",
    type=float,
    default=1.0,
    metavar="GAIN",
    help="Software gain multiplier for microphone. Defaults to 1.0.",
)
_parser.add_argument(
    "--gui",
    action="store_true",
    help="Launch desktop GUI instead of headless/terminal mode.",
)
_parser.add_argument(
    "--headless",
    action="store_true",
    help="Run without displaying OpenCV video window.",
)
_args, _ = _parser.parse_known_args()
CLI_CAMERA = "mock" if _args.mock_camera else _args.camera
CLI_MOCK_SENSORS = _args.mock_sensors
MIC_DEVICE = _args.mic
MIC_GAIN = _args.mic_gain
USE_GUI = _args.gui
CLI_HEADLESS = _args.headless
LLM_MODEL = ""

# Get script directory for path resolution (DO NOT change working directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Import configuration BEFORE other modules
import sys
from typing import Optional, List, Dict, Any, Tuple
sys.path.insert(0, SCRIPT_DIR)
from config import *

import cv2
import numpy as np
import threading
import queue
import time
import urllib.request

try:
    import torch
except ImportError:
    torch = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import easyocr
except ImportError:
    easyocr = None

from hal.vision import LightweightYOLO, GeometricDepthProvider

from voice.tts import VoiceEngine
from voice.assistant import VoiceAssistant
from voice.app_launcher import open_app, close_app
from telemetry import (
    start_telemetry, broadcast_event, set_frame_getter,
    set_imu_provider, set_bridge
)
from hal import (
    create_camera_provider, create_imu_provider,
    MockMicrocontrollerBridge, UnoQBridge
)
from system_state import (
    SystemState, SpeechPriority, VisionState, CameraState,
    NavigationState, SubsystemHealth
)
from workers import VisionWorker, SafetyWorker

# =====================================================
# CONFIGURATION
# =====================================================
# Best available compute accelerator (CUDA / Apple Silicon MPS / CPU)
DEVICE = get_device()
print(f"[SYSTEM] Platform: Arduino UNO Q | Compute Device: {DEVICE}")


# =====================================================
# 1. ROBUST VIDEO STREAM RECEIVER
# =====================================================
class CameraStream:
    def __init__(self, src):
        self.src = src
        print(f"[STREAM] Attempting to connect to {src}...")
        self.cap = cv2.VideoCapture(src)
        
        # Fallback to local webcam (0) if stream is completely unreachable
        if not self.cap.isOpened():
            print(f"[STREAM] Warning: {src} unreachable. Falling back to local webcam ({CAMERA_FALLBACK}).")
            self.cap = cv2.VideoCapture(CAMERA_FALLBACK)
            
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False
        self.connected = False
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.connected = True
                with self.lock:
                    self.frame = frame
            else:
                self.connected = False
                time.sleep(0.05)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()

# =====================================================
# 2. SURDAS PERCEPTION & VOICE BRAIN (YOLO + MIDAS WALLS + VOICE)
# =====================================================
class SurdasBrain:
    def __init__(self):
        # Central thread-safe state (independent of all subsystems)
        self.state = SystemState()

        # Start Dashboard Telemetry
        start_telemetry()
        
        # 1. TTS Voice Output Engine
        self.voice = VoiceEngine()
        self.voice.speak("SURDAS Vision and Navigation System starting on Arduino UNO Q.")

        # Initialize Camera Provider (Linux V4L2 auto-discovery or mock)
        cam_dev = CLI_CAMERA if CLI_CAMERA is not None else CAMERA_DEVICE
        allow_mock = ALLOW_MOCK_CAMERA or (CLI_CAMERA == "mock")
        print(f"[CAMERA] Initializing CameraProvider (device={cam_dev}, allow_mock={allow_mock})...")
        self.camera = create_camera_provider(cam_dev, allow_mock=allow_mock)
        self.stream = self.camera  # Backwards compatibility alias for workers

        # Motion & IMU Sensor Provider
        enable_sim = CLI_MOCK_SENSORS or IMU_SIMULATION_ENABLED
        self.imu_provider = create_imu_provider(enable_simulation=enable_sim)

        # Microcontroller Bridge (Arduino UNO Q STM32)
        if UNO_Q_ENABLE_MOCK_BRIDGE:
            self.mcu_bridge = MockMicrocontrollerBridge()
        else:
            self.mcu_bridge = UnoQBridge(port=UNO_Q_BRIDGE_PORT or None, baudrate=UNO_Q_BRIDGE_BAUD)

        # Connect providers to unified telemetry service
        set_frame_getter(self.get_latest_jpeg)
        set_imu_provider(self.imu_provider)
        set_bridge(self.mcu_bridge)

        # 2. Parallel AI Model Loading (YOLOv8, MiDaS, EasyOCR, Voice)
        print("[SYSTEM] ⚡ Loading models in parallel (YOLOv8, MiDaS, EasyOCR, Voice)...")
        t0_load = time.time()
        self.geometric_depth = GeometricDepthProvider()

        def _load_yolo():
            yolo_path = get_yolo_model_path()
            if YOLO is not None:
                try:
                    self.yolo = YOLO(yolo_path)
                    print(f"[AI] ✅ YOLOv8 loaded from {yolo_path}")
                    return
                except Exception as e:
                    print(f"[AI] YOLO load note: {e}. Using Lightweight Torch-Free YOLO.")
            self.yolo = LightweightYOLO(yolo_path)
            print("[AI] ✅ Lightweight Torch-Free YOLO active (OpenCV).")

        def _load_midas():
            if torch is not None and MIDAS_CACHE_DIR.exists():
                try:
                    midas_cache_str = str(MIDAS_CACHE_DIR)
                    ft_midas_path = get_midas_model_path()
                    midas_transforms = torch.hub.load(midas_cache_str, "transforms", source="local")
                    self.transform = midas_transforms.small_transform
                    if ft_midas_path:
                        base = torch.hub.load(midas_cache_str, "MiDaS_small", source="local", pretrained=False)
                        state = torch.load(ft_midas_path, map_location="cpu")
                        base.load_state_dict(state)
                        self.midas = base.to(DEVICE).eval()
                    else:
                        self.midas = torch.hub.load(
                            midas_cache_str, "MiDaS_small", source="local", pretrained=True
                        ).to(DEVICE).eval()
                    print("[AI] ✅ MiDaS Depth Estimator loaded via PyTorch.")
                    return
                except Exception as e:
                    print(f"[AI] MiDaS load note: {e}. Using Lightweight Geometric Depth.")
            self.midas = None
            print("[AI] ✅ Lightweight Torch-Free Geometric Depth active (OpenCV).")

        def _load_ocr():
            if easyocr is not None:
                try:
                    self.ocr = easyocr.Reader(['en', 'hi'], gpu=(DEVICE == "cuda"), download_enabled=False)
                    print("[AI] ✅ EasyOCR loaded (en + hi).")
                    return
                except Exception:
                    pass
            self.ocr = None
            print("[AI] ℹ️  EasyOCR skipped (torch-free mode). Vision & navigation fully active.")

        def _load_voice():
            self.llm = None  # Disabled for Arduino UNO Q deterministic phase
            try:
                self.voice_assistant = VoiceAssistant(self, mic_device=MIC_DEVICE, mic_gain=MIC_GAIN)
                self.voice_assistant.start()
                print("[AI] ✅ Voice Assistant active (deterministic rule-based).")
            except Exception as e:
                print(f"[VOICE] Voice Assistant warning: {e}. Vision continues normally.")
                self.voice_assistant = None

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            f_yolo = executor.submit(_load_yolo)
            f_midas = executor.submit(_load_midas)
            f_ocr = executor.submit(_load_ocr)
            f_voice = executor.submit(_load_voice)

            f_yolo.result()
            f_midas.result()
            f_ocr.result()
            f_voice.result()

        print(f"[SYSTEM] 🚀 Models loaded in parallel in {time.time() - t0_load:.2f}s!")

        self.mode = "NAV"  # 'NAV' or 'OCR'
        self.last_speech_time = 0
        self.last_spoken = ""
        self.led_on = False
        self.conf_threshold = YOLO_CONFIDENCE_THRESHOLD
        self.headless = CLI_HEADLESS or HEADLESS_MODE
        self.latest_display_frame = None
        self.log_history = []
        
        # AI Voice Mode state
        self.ai_voice_mode = False
        self.ai_voice_thread = None
        self.ai_voice_active = False

        # Live Vision Context State (kept for API compat; VisionWorker writes these)
        self.latest_detected_objects = []
        self.latest_closest_obstacle = None
        self.wall_detected = False

        # ── Start background workers ──────────────────────────────────────────
        # VisionWorker: runs YOLO + MiDaS in its own thread, never blocking voice
        self.vision_worker = VisionWorker(self)
        self.vision_worker.start()

        # SafetyWorker: reads PerceptionResult → fires CRITICAL_SAFETY TTS
        self.safety_worker = SafetyWorker(self, self.vision_worker)
        self.safety_worker.start()

        # LLMWorker is omitted in deterministic phase
        self.llm_worker = None
        
        # Offline Navigation with proper location provider
        from navigation import OfflineMapManager, Navigator
        from navigation.location import GPSSerialLocation, ManualLocation, LocationUnavailableError
        
        self.map_manager = OfflineMapManager()
        self.map_manager.auto_load()
        self.navigator = Navigator(self.map_manager, self)
        
        # Indoor Navigation & Spatial Memory
        from spatial_memory import SpatialMemory
        from indoor_navigator import IndoorNavigator
        from indoor_perception import IndoorPerception
        
        print("[SYSTEM] Initializing indoor navigation and spatial memory...")
        self.spatial_memory = SpatialMemory()
        self.indoor_navigator = IndoorNavigator(self.spatial_memory, self.voice)
        self.indoor_perception = IndoorPerception(grid_size=32)
        
        # Get memory stats
        stats = self.spatial_memory.get_stats()
        print(f"[SPATIAL_MEMORY] Loaded: {stats['rooms']} rooms, {stats['objects']} objects, {stats['landmarks']} landmarks")
        
        # State for indoor navigation
        self.indoor_nav_mode = False  # Toggle for indoor navigation processing
        
        # Initialize location provider based on configuration
        gps_enabled = GPS_ENABLED  # Use local variable to avoid shadowing
        if gps_enabled:
            try:
                print(f"[GPS] Initializing GPS on port {GPS_PORT}...")
                self.location_provider = GPSSerialLocation(
                    port=GPS_PORT,
                    baudrate=GPS_BAUD,
                    timeout=GPS_TIMEOUT,
                    stale_threshold=GPS_STALE_THRESHOLD,
                    max_jump_distance=GPS_MAX_JUMP_DISTANCE_M
                )
                self.location_provider.start()
                # Try to get initial fix
                try:
                    initial_loc = self.location_provider.get_location()
                    print(f"[GPS] ✅ GPS fix acquired: ({initial_loc['lat']:.4f}, {initial_loc['lon']:.4f})")
                    self.navigator._current_location = initial_loc
                except Exception as e:
                    print(f"[GPS] ⚠️  Initial GPS fix failed: {e}")
                    print("[GPS] Navigation will work once GPS acquires a fix.")
            except Exception as e:
                print(f"[GPS] ❌ GPS initialization failed: {e}")
                print("[GPS] Falling back to manual test location for development.")
                gps_enabled = False
        
        if not gps_enabled:
            # Use manual location for testing
            test_loc = DEFAULT_TEST_LOCATION
            self.location_provider = ManualLocation(
                lat=test_loc["lat"],
                lon=test_loc["lon"],
                name=test_loc["name"]
            )
            self.navigator._current_location = self.location_provider.get_location()

        self.voice.speak("Ready. Navigation mode active. Say Hey Surdas.")
        
        # Dashboard integration state
        self.log_history = []
        self.latest_display_frame = None
        self.headless = False
        
        # AI Voice Mode state
        self.ai_voice_mode = False
        self.ai_voice_thread = None

    # ── LLM helper ─────────────────────────────────────────────────
    def query_llm(self, prompt: str, speak: bool = True) -> str:
        """Ask the LLM a question. Streams response sentence-by-sentence.
        Returns the full response as a string."""
        full = []
        for sentence in self.llm.query(prompt, vision_context=self.get_vision_context()):
            if sentence:
                full.append(sentence)
                if speak:
                    self.voice.speak(sentence)
        return " ".join(full)

    # ── App launcher helpers ─────────────────────────────────────────
    def launch_app(self, app_name: str, speak: bool = True) -> bool:
        """Open a macOS app by name. Returns True on success."""
        success, msg = open_app(app_name)
        if speak:
            self.voice.speak(msg)
        return success

    def quit_app(self, app_name: str, speak: bool = True) -> bool:
        """Close a macOS app by name. Returns True on success."""
        success, msg = close_app(app_name)
        if speak:
            self.voice.speak(msg)
        return success
        
    def add_log(self, msg: str):
        print(msg)
        timestamp = time.strftime("%H:%M:%S")
        self.log_history.append(f"[{timestamp}] {msg}")
        if len(self.log_history) > 100:
            self.log_history.pop(0)

    def get_vision_context(self) -> dict:
        """Provides current live scene state for LLM and Command Router."""
        ctx = {
            "mode": self.mode,
            "torch_on": self.led_on,
            "detected_objects": list(self.latest_detected_objects),
            "closest_obstacle": self.latest_closest_obstacle,
            "wall_ahead": self.wall_detected
        }
        if hasattr(self, "navigator") and self.navigator:
            ctx["navigation"] = self.navigator.get_status()
        return ctx

    def get_status_summary(self) -> str:
        """Returns verbal status report."""
        stream_stat = "connected" if self.stream.connected else "disconnected"
        torch_stat = "on" if self.led_on else "off"
        obj_count = len(self.latest_detected_objects)
        wall_stat = "Wall detected ahead." if self.wall_detected else "No wall."
        return f"System status: Camera is {stream_stat}. Flashlight is {torch_stat}. Mode is {self.mode}. Detecting {obj_count} objects. {wall_stat}"

    def is_voice_active(self) -> bool:
        """Returns True if a voice command is being received, processed, or spoken."""
        if self.voice_assistant is not None and getattr(self.voice_assistant, "is_active", False):
            return True
        if self.voice.is_speaking or not self.voice._queue.empty():
            return True
        return False
    
    def get_voice_health(self) -> dict:
        """Returns voice system health diagnostics."""
        if self.voice_assistant is None:
            return {"available": False, "reason": "Voice assistant not initialized"}
        
        try:
            health = self.voice_assistant.get_health_status()
            health["available"] = True
            health["tts_speaking"] = self.voice.is_speaking
            health["tts_queue_size"] = self.voice._queue.qsize() if hasattr(self.voice._queue, 'qsize') else 0
            health["tts_priority"] = self.voice.current_priority
            return health
        except Exception as e:
            return {"available": False, "error": str(e)}

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Encodes latest display frame to JPEG bytes for MJPEG streaming."""
        frame = self.latest_display_frame
        if frame is not None:
            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                return buf.tobytes()
        return None

    def toggle_esp32_led(self, state: bool):
        """Toggles hardware illuminator via microcontroller bridge."""
        if hasattr(self, "mcu_bridge") and self.mcu_bridge:
            self.mcu_bridge.send_command("SET_LED", state=state)
        self.led_on = state

    def toggle_led(self, state: bool):
        return self.toggle_esp32_led(state)

    def compute_depth(self, frame):
        """Computes dense relative depth map."""
        if hasattr(self, "midas") and self.midas is not None and torch is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = self.transform(rgb).to(DEVICE)
                with torch.no_grad():
                    depth = self.midas(input_tensor)
                    depth = torch.nn.functional.interpolate(
                        depth.unsqueeze(1),
                        size=frame.shape[:2],
                        mode="bicubic",
                        align_corners=False
                    ).squeeze()
                depth_np = depth.cpu().numpy()
                depth_vis = cv2.normalize(depth_np, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                return depth_np, depth_vis
            except Exception:
                pass

        if not hasattr(self, "geometric_depth") or self.geometric_depth is None:
            self.geometric_depth = GeometricDepthProvider()
        depth_np = self.geometric_depth.estimate_depth(frame)
        depth_vis = cv2.normalize(depth_np, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return depth_np, depth_vis

    def process_navigation(self, frame):
        """
        Processes both Discrete Objects (YOLO) and Dense Continuous Barriers/Walls (MiDaS).
        NOW ALSO: Indoor navigation perception and guidance.
        
        SAFETY CRITICAL: Vision processing continues ALWAYS, even during voice activity.
        Only non-critical voice announcements are suppressed during voice commands.
        """
        # Update navigation location from GPS/location provider
        if hasattr(self, "navigator") and self.navigator:
            try:
                current_loc = self.location_provider.get_location()
                self.navigator.update_location(current_loc)
            except Exception as e:
                # GPS unavailable or stale - navigation continues with last known position
                pass

        h, w, _ = frame.shape
        third_w = w // 3

        # -------------------------------------------------------------
        # ALWAYS RUN SAFETY PERCEPTION (YOLO + MIDAS)
        # Safety perception MUST continue during voice processing
        # -------------------------------------------------------------
        raw_depth, depth_vis = self.compute_depth(frame)
        
        # -------------------------------------------------------------
        # 1. DENSE MIDAS SPATIAL ANALYSIS (WALL & BARRIER DETECTION)
        # -------------------------------------------------------------
        third_w = w // 3
        center_depth_crop = raw_depth[:, third_w : 2 * third_w]
        left_depth_crop = raw_depth[:, :third_w]
        right_depth_crop = raw_depth[:, 2 * third_w:]

        center_med = np.median(center_depth_crop) if center_depth_crop.size > 0 else 0
        left_med = np.median(left_depth_crop) if left_depth_crop.size > 0 else 0
        right_med = np.median(right_depth_crop) if right_depth_crop.size > 0 else 0

        # Close proximity pixel density in central walking corridor
        center_close_ratio = np.mean(center_depth_crop > DEPTH_WALL_DENSE_THRESHOLD) if center_depth_crop.size > 0 else 0

        self.wall_detected = False
        immediate_danger = None
        guidance_text = None

        # -------------------------------------------------------------
        # 2. RUN YOLOV8 OBJECT DETECTION
        # -------------------------------------------------------------
        results = self.yolo(frame, conf=self.conf_threshold, verbose=False)[0]

        detected_obstacles = []
        current_objects_in_view = []
        center_object_found = False
        
        # Prepare YOLO detections for indoor perception
        yolo_detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.yolo.names[cls_id]
            current_objects_in_view.append(label)

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) / 2
            
            # Prepare for indoor perception
            yolo_detections.append({
                "class": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
            
            # Spatial position
            if cx < third_w:
                pos = "on your left"
                color = (255, 200, 0)
            elif cx > 2 * third_w:
                pos = "on your right"
                color = (0, 200, 255)
            else:
                pos = "ahead"
                color = (0, 255, 0)
                center_object_found = True

            # Sample depth inside object bounding box
            crop_depth = raw_depth[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            med_depth = np.median(crop_depth) if crop_depth.size > 0 else 0

            # Classify relative proximity using adaptive or fixed thresholds
            if DEPTH_ADAPTIVE_NORMALIZATION and raw_depth.size > 0:
                proximity = classify_depth_proximity_adaptive(med_depth, raw_depth)
            else:
                proximity = classify_depth_proximity(med_depth)
            
            proximity_desc = get_proximity_description(proximity, lang="en")
            
            # Color coding based on proximity
            if proximity == "VERY_CLOSE" and pos == "ahead":
                color = (0, 0, 255)  # Red for close obstacle directly ahead
            elif proximity == "VERY_CLOSE":
                color = (0, 100, 255)  # Orange for close obstacle to side
            elif proximity == "CLOSE":
                color = (0, 200, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green

            if proximity == "VERY_CLOSE" and pos == "ahead":
                immediate_danger = f"Caution! {label} directly ahead."

            detected_obstacles.append((label, pos, proximity, med_depth, conf, proximity_desc))

            # Draw standard YOLO Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Display relative proximity instead of fake metric distance
            tag = f"{label} {int(conf*100)}% | {proximity_desc}"
            
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(0, y1)), color, -1)
            cv2.putText(frame, tag, (x1 + 3, max(14, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 2.5 INDOOR NAVIGATION PERCEPTION & GUIDANCE
        # -------------------------------------------------------------
        indoor_perception_result = None
        if hasattr(self, "indoor_perception") and hasattr(self, "indoor_navigator"):
            try:
                # Process frame for indoor navigation
                indoor_perception_result = self.indoor_perception.process_frame(
                    yolo_detections=yolo_detections,
                    depth_map=raw_depth,
                    img_width=w,
                    img_height=h
                )
                
                # Update indoor navigator with perception
                self.indoor_navigator.update(indoor_perception_result)
                
                # Check navigation status
                nav_status = self.indoor_navigator.get_status()
                
                # Broadcast indoor navigation telemetry
                if nav_status.state != "IDLE":
                    try:
                        broadcast_event("indoor_navigation", {
                            "state": nav_status.state.value,
                            "destination": nav_status.destination,
                            "confidence": nav_status.confidence.value,
                            "distance_remaining": nav_status.distance_remaining,
                            "safe_directions": nav_status.safe_directions_count,
                            "hold_reason": nav_status.hold_reason.value if nav_status.hold_reason else None
                        })
                    except Exception:
                        pass
                
            except Exception as e:
                # Indoor perception failure should not crash main loop
                print(f"[INDOOR_PERCEPTION] Error: {e}")

        # -------------------------------------------------------------
        # 3. WALL / CONTINUOUS SURFACE DETECTION (MIDAS)
        # -------------------------------------------------------------
        # If central area is blocked across a broad field with high depth and no single object explains it
        if center_close_ratio > DEPTH_WALL_CENTER_RATIO and not center_object_found:
            self.wall_detected = True
            if center_med > DEPTH_WALL_IMMEDIATE_THRESHOLD:
                immediate_danger = "Stop! Wall directly in front of you."
            else:
                immediate_danger = "Caution! Wall ahead."

            # Calculate best clearance bypass
            if left_med < center_med - 150 and left_med < right_med:
                guidance_text = "Wall ahead. Path is clear on your left."
            elif right_med < center_med - 150:
                guidance_text = "Wall ahead. Path is clear on your right."

        # Update Live Vision State
        self.latest_detected_objects = current_objects_in_view

        # -------------------------------------------------------------
        # 4. VOICE FEEDBACK DECISION ENGINE
        # -------------------------------------------------------------
        now = time.time()

        # Check if voice assistant is busy (but NEVER block safety warnings)
        assistant_busy = (
            self.voice_assistant is not None
            and (
                self.voice_assistant._processing
                or getattr(self.voice_assistant, "is_recording", False)
                or not self.voice._queue.empty()
                or self.voice.is_speaking
            )
        )

        # Priority 1: SAFETY-CRITICAL Immediate Collision Hazard
        # These ALWAYS fire and can interrupt voice processing
        if immediate_danger:
            self.latest_closest_obstacle = immediate_danger
            speech_to_say = guidance_text if guidance_text else immediate_danger
            if speech_to_say != self.last_spoken or now - self.last_speech_time > SAFETY_ANNOUNCEMENT_COOLDOWN:
                # Safety warnings use force=True to interrupt other voice activity
                self.voice.speak(speech_to_say, force=True)
                self.last_speech_time = now
                self.last_spoken = speech_to_say

        # Priority 2: Routine navigation announcements
        # These are suppressed during voice commands to avoid interrupting user interaction
        # AND during indoor navigation mode (indoor navigator handles its own guidance)
        elif not assistant_busy and now - self.last_speech_time > ROUTINE_ANNOUNCEMENT_COOLDOWN:
            # Check if indoor navigator is active
            indoor_nav_active = (
                hasattr(self, "indoor_navigator") 
                and self.indoor_navigator.get_status().state not in ["IDLE"]
            )
            
            if not indoor_nav_active and detected_obstacles:
                detected_obstacles.sort(key=lambda x: x[3], reverse=True)
                top_lbl, top_pos, top_prox, _, _, top_prox_desc = detected_obstacles[0]
                speech_text = f"{top_lbl} {top_pos}, {top_prox_desc}."
                self.latest_closest_obstacle = speech_text

                if speech_text != self.last_spoken or now - self.last_speech_time > LONG_ANNOUNCEMENT_COOLDOWN:
                    self.voice.speak(speech_text)
                    self.last_spoken = speech_text
                    self.last_speech_time = now
            elif not indoor_nav_active:
                self.latest_closest_obstacle = None
                if self.last_spoken != "clear":
                    self.voice.speak("Path is clear.")
                    self.last_spoken = "clear"
                    self.last_speech_time = now

        # -------------------------------------------------------------
        # 5. DEPTH HEATMAP & CORRIDOR VISUALIZATION
        # -------------------------------------------------------------
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        
        # Draw spatial corridor lines (Left | Center | Right)
        cv2.line(depth_colormap, (third_w, 0), (third_w, h), (255, 255, 255), 1)
        cv2.line(depth_colormap, (2 * third_w, 0), (2 * third_w, h), (255, 255, 255), 1)

        # Draw Wall indicator on depth map
        if self.wall_detected:
            cv2.rectangle(depth_colormap, (third_w + 10, 10), (2 * third_w - 10, 50), (0, 0, 255), -1)
            cv2.putText(depth_colormap, "WALL AHEAD", (third_w + 20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw indoor navigation status overlay
        if indoor_perception_result and hasattr(self, "indoor_navigator"):
            nav_status = self.indoor_navigator.get_status()
            if nav_status.state != "IDLE":
                # Draw navigation confidence indicator
                conf_color = {
                    "HIGH": (0, 255, 0),
                    "MEDIUM": (0, 200, 255),
                    "LOW": (0, 100, 255),
                    "INVALID": (0, 0, 255)
                }.get(indoor_perception_result.nav_confidence.value, (100, 100, 100))
                
                cv2.rectangle(depth_colormap, (10, 10), (220, 80), conf_color, 2)
                cv2.putText(depth_colormap, f"NAV: {nav_status.state.value}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(depth_colormap, f"Conf: {indoor_perception_result.nav_confidence.value}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if nav_status.destination:
                    dest_text = nav_status.destination[:15]  # Truncate long names
                    cv2.putText(depth_colormap, f"→ {dest_text}", (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Visual indicator when voice command is active (but vision continues)
        if self.is_voice_active():
            cv2.rectangle(frame, (10, h - 40), (200, h - 10), (0, 150, 255), -1)
            cv2.putText(frame, "🎤 Voice Active", (20, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
        # Broadcast for Caregiver Dashboard
        broadcast_event("vision", self.get_vision_context())

        self._last_depth_colormap = depth_colormap.copy()
        return frame, depth_colormap, len(detected_obstacles)

    def process_ocr(self, frame):
        """Document / Sign Text Reading Mode."""
        if self.ocr is None:
            self.voice.speak("Text reading is not enabled in lightweight mode.")
            self.mode = "NAV"
            return

        self.toggle_esp32_led(True)
        self.voice.speak("Reading text...")
        time.sleep(0.3)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.ocr.readtext(rgb, detail=0)
        self.toggle_esp32_led(False)

        if results:
            text = " ".join(results)
            self.add_log(f"[OCR] Text: {text}")
            self.voice.speak(f"Text says: {text}")
        else:
            self.voice.speak("No clear text detected.")

        self.mode = "NAV"
    
    def start_ai_voice_mode(self):
        """Start always-listening AI voice conversation mode.
        
        SINGLE QUESTION MODE: Answers one question then automatically exits.
        Vision processing is paused during AI mode.
        """
        if self.ai_voice_mode:
            return  # Already active
        
        self.ai_voice_mode = True
        
        # Pause vision workers
        print("[AI_VOICE] ⏸️  Pausing vision processing...")
        if hasattr(self, 'vision_worker'):
            self.vision_worker.pause()
        if hasattr(self, 'safety_worker'):
            self.safety_worker.pause()
        
        self.voice.speak("AI voice mode. Ask your question.", force=True)
        
        # Run AI voice loop in background thread
        def ai_voice_loop():
            import numpy as np
            import queue as q
            
            # Initialize if needed
            try:
                from voice.vad import VoiceActivityDetector
                vad = VoiceActivityDetector(sample_rate=16000)
            except:
                print("[AI_VOICE] VAD initialization failed")
                self.ai_voice_mode = False
                return
            
            SAMPLE_RATE = 16000
            CHUNK_MS = 30
            CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)
            SILENCE_CHUNKS = 20
            
            audio_queue = q.Queue()
            question_answered = False  # Track if we've answered one question
            
            def audio_callback(indata, frames, time_info, status):
                audio_queue.put(indata.flatten().copy())
            
            try:
                import sounddevice as sd
                stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype='float32',
                    blocksize=CHUNK_SIZE,
                    callback=audio_callback
                )
                
                stream.start()
                print("[AI_VOICE] 🎤 Listening for ONE question...")
                
                recording = False
                speech_buffer = []
                silence_count = 0
                echo_suppression_active = False
                echo_message_shown = False
                last_spoken_text = ""
                
                while self.ai_voice_mode and not question_answered:
                    try:
                        chunk = audio_queue.get(timeout=0.1)
                    except q.Empty:
                        continue
                    
                    # Echo cancellation
                    if self.voice.is_speaking:
                        if recording:
                            recording = False
                            speech_buffer = []
                            silence_count = 0
                        if not echo_suppression_active:
                            echo_suppression_active = True
                            if not echo_message_shown:
                                print("[AI_VOICE] 🔇 Echo suppression active")
                                echo_message_shown = True
                        continue
                    
                    time_since_tts = time.time() - self.voice.last_speech_time
                    if time_since_tts < 0.5:
                        if recording:
                            recording = False
                            speech_buffer = []
                            silence_count = 0
                        continue
                    
                    if echo_suppression_active:
                        echo_suppression_active = False
                    
                    # Voice activity detection
                    is_speech = vad.is_speech(chunk)
                    
                    if is_speech:
                        if not recording:
                            print("[AI_VOICE] 🔴 Listening...", end="", flush=True)
                            recording = True
                            speech_buffer = []
                            silence_count = 0
                        speech_buffer.append(chunk)
                        silence_count = 0
                    elif recording:
                        speech_buffer.append(chunk)
                        silence_count += 1
                        
                        if silence_count >= SILENCE_CHUNKS:
                            print(" Done.")
                            recording = False
                            
                            audio = np.concatenate(speech_buffer)
                            if audio.ndim > 1:
                                audio = audio.flatten()
                            
                            speech_buffer = []
                            silence_count = 0
                            
                            if len(audio) < SAMPLE_RATE * 0.3:
                                print("  (Too short, try again)\n")
                                continue
                            
                            audio_level = np.abs(audio).max()
                            if audio_level < 0.001:
                                print("  (Too quiet, try again)\n")
                                continue
                            
                            # Transcribe
                            print("[AI_VOICE] ⏳ Transcribing...", end="", flush=True)
                            try:
                                from voice.stt import SpeechToText
                                if not hasattr(self, '_ai_stt'):
                                    self._ai_stt = SpeechToText()
                                text, lang = self._ai_stt.transcribe(audio, sample_rate=SAMPLE_RATE)
                                print(f" Done.")
                            except Exception as e:
                                print(f" Error: {e}\n")
                                continue
                            
                            if not text or not text.strip() or len(text.strip()) < 3:
                                print("  (No speech detected, try again)\n")
                                continue
                            
                            print(f"[AI_VOICE] 💬 You: {text}")
                            
                            # Echo detection
                            if last_spoken_text:
                                text_words = set(text.lower().strip().split())
                                last_words = set(last_spoken_text.lower().strip().split())
                                if text_words and last_words:
                                    overlap = len(text_words & last_words) / len(text_words)
                                    if overlap > 0.6:
                                        print("  (Echo detected, try again)\n")
                                        continue
                            
                            # Route through deterministic command router (no Ollama required)
                            print(f"[AI_VOICE] 🎯 Executing deterministic command: {text}")
                            try:
                                if self.voice_assistant and self.voice_assistant.router:
                                    self.voice_assistant.router.route_command(text, lang=lang)
                                else:
                                    self.voice.speak(f"Command processed: {text}", lang=lang)
                                while self.voice.is_speaking:
                                    time.sleep(0.1)
                                question_answered = True
                                print("[AI_VOICE] Question answered. Exiting AI mode...\n")
                                    
                            except Exception as e:
                                print(f"\n[AI_VOICE] Error: {e}")
                                print("[AI_VOICE] Trying again...\n")
                                continue
                
                stream.stop()
                stream.close()
                
                # Auto-exit after answering
                self.stop_ai_voice_mode()
                
            except Exception as e:
                print(f"[AI_VOICE] Failed to start: {e}")
                self.ai_voice_mode = False
                # Resume vision on error
                if hasattr(self, 'vision_worker'):
                    self.vision_worker.resume()
                if hasattr(self, 'safety_worker'):
                    self.safety_worker.resume()
        
        self.ai_voice_thread = threading.Thread(target=ai_voice_loop, daemon=True)
        self.ai_voice_thread.start()
    
    def stop_ai_voice_mode(self):
        """Stop AI voice mode and resume vision processing."""
        if not self.ai_voice_mode:
            return
        
        self.ai_voice_mode = False
        
        # Resume vision workers
        print("[AI_VOICE] ▶️  Resuming vision processing...")
        if hasattr(self, 'vision_worker'):
            self.vision_worker.resume()
        if hasattr(self, 'safety_worker'):
            self.safety_worker.resume()
        
        self.voice.speak("Returning to navigation mode.", force=True)
        print("[AI_VOICE] Stopped. Vision resumed.")

    def run(self):
        print("\n" + "="*65)
        print(" SURDAS UNO Q ASSISTIVE VISION & NAVIGATION PLATFORM")
        print(" Features:")
        print("   • OpenCV / V4L2 Camera Provider (Linux MPU)")
        print("   • YOLOv8 Object & Obstacle Detection (VisionWorker thread)")
        print("   • MiDaS Dense Depth & Free-Space Perception")
        print("   • Deterministic Indoor Navigation & Spatial Memory")
        print("   • Rule-Based Voice Assistant (Bilingual EN/HI)")
        print("   • STM32 MCU Bridge & Fall Detection Telemetry")
        print(" Controls: [N] Nav | [T] Text/OCR | [A] Voice | [L] Illuminator | [Q] Quit")
        print("="*65 + "\n")

        while True:
            # ── No camera frame yet (camera initializing) ───────────────
            if self.stream.get_frame() is None:
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for Camera Provider...", (60, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.putText(blank, "SURDAS UNO Q Camera Provider initializing", (60, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                cv2.putText(blank, "Connect a USB/V4L2 camera or enable mock provider", (60, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                self.latest_display_frame = blank.copy()
                if not self.headless:
                    cv2.imshow("SURDAS - Vision Monitor", blank)
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        break
                else:
                    time.sleep(0.1)
                continue

            if self.mode == "NAV":
                # ── Read latest result from VisionWorker (non-blocking) ─────
                perception = self.vision_worker.latest

                if perception is not None and perception.frame is not None:
                    rgb_view = perception.frame
                    depth_view = perception.depth_colormap
                    count = len(perception.detected_obstacles)
                    fps = perception.fps

                    if depth_view is not None:
                        # Ensure same height before hstack
                        h = min(rgb_view.shape[0], depth_view.shape[0])
                        rgb_view = rgb_view[:h]
                        depth_view = depth_view[:h]
                        combined = np.hstack((rgb_view, depth_view))
                    else:
                        combined = rgb_view

                    display_frame = cv2.resize(combined, (1024, 400))
                    wall_tag = " | WALL DETECTED" if self.wall_detected else ""
                    cv2.putText(display_frame, f"FPS: {int(fps)} | Objects: {count}{wall_tag}", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                    self.latest_display_frame = display_frame.copy()

                    if not self.headless:
                        cv2.imshow("SURDAS - Vision Monitor", display_frame)
                else:
                    # Vision worker not ready yet — brief sleep
                    time.sleep(0.01)

                # GPS location update (kept in main loop since it's fast)
                if hasattr(self, "navigator") and self.navigator:
                    try:
                        current_loc = self.location_provider.get_location()
                        self.navigator.update_location(current_loc)
                    except Exception:
                        pass

                # Broadcast vision telemetry
                try:
                    broadcast_event("vision", self.get_vision_context())
                except Exception:
                    pass

            elif self.mode == "OCR":
                frame = self.stream.get_frame()
                if frame is not None:
                    self.process_ocr(frame)

            # ── Keyboard controls ─────────────────────────────────────────
            if not self.headless:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    self.mode = "NAV"
                    self.voice.speak("Navigation mode active.")
                elif key == ord('t'):
                    self.mode = "OCR"
                elif key == ord('l'):
                    self.toggle_esp32_led(not self.led_on)
                elif key == ord('a'):
                    # Toggle AI Voice Mode
                    if self.ai_voice_mode:
                        self.stop_ai_voice_mode()
                    else:
                        self.start_ai_voice_mode()
            else:
                time.sleep(0.01)

        # ── Graceful shutdown ─────────────────────────────────────────────
        print("\n[SYSTEM] Shutting down workers...")
        self.vision_worker.stop()
        self.safety_worker.stop()
        if self.llm_worker:
            self.llm_worker.stop()
        if self.voice_assistant:
            self.voice_assistant.stop()
        if hasattr(self.stream, 'release'):
            self.stream.release()
        elif hasattr(self.stream, 'stop'):
            self.stream.stop()
        cv2.destroyAllWindows()
        print("[SYSTEM] SURDAS shutdown complete.")


if __name__ == "__main__":
    if USE_GUI:
        from surdas_gui_tk import main as gui_main
        gui_main()
    else:
        brain = SurdasBrain()
        brain.run()
