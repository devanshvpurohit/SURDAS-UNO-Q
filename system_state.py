"""
system_state.py — Thread-safe Central System State and World Model for SURDAS.

Design principles:
  • Each subsystem updates ONLY its own state.
  • Voice state is COMPLETELY INDEPENDENT from Navigation and Safety state.
  • Path being blocked NEVER disables microphone or wake word.
  • Snapshot method produces immutable read-only WorldState for LLM / telemetry.
  • Thread-safe operations via RLock.
"""
from __future__ import annotations

import time
import threading
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# ── State Enums ───────────────────────────────────────────────────────────────

class VoiceState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    RECORDING = "RECORDING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"


class WakeWordState(str, Enum):
    LISTENING = "LISTENING"
    DETECTED = "DETECTED"
    PAUSED = "PAUSED"
    DISABLED = "DISABLED"


class TTSState(str, Enum):
    IDLE = "IDLE"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"


class VisionState(str, Enum):
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"


class NavigationState(str, Enum):
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    SAFETY_HOLD = "SAFETY_HOLD"
    APPROACHING = "APPROACHING"
    ARRIVED = "ARRIVED"


class SafetyLevel(str, Enum):
    CLEAR = "CLEAR"
    CAUTION = "CAUTION"
    BLOCKED = "BLOCKED"
    CRITICAL = "CRITICAL"


class LocalizationState(str, Enum):
    TRACKING = "TRACKING"
    UNCERTAIN = "UNCERTAIN"
    LOST = "LOST"


class CameraState(str, Enum):
    CONNECTED = "CONNECTED"
    STREAMING = "STREAMING"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"


class LLMState(str, Enum):
    READY = "READY"
    QUERYING = "QUERYING"
    STREAMING = "STREAMING"
    TIMEOUT = "TIMEOUT"
    UNAVAILABLE = "UNAVAILABLE"


class SubsystemHealth(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class SpeechPriority(IntEnum):
    """
    Priority levels for the unified TTS queue.
    Lower number = HIGHER priority (can preempt / interrupt lower priority).
    """
    CRITICAL_SAFETY = 1   # Immediate collision/fall warning (interrupts everything)
    NAVIGATION      = 2   # Turn-by-turn guidance / imminent waypoint
    USER_COMMAND    = 3   # Direct command responses ("Yes?", "Torch on", etc.)
    ASSISTANT_RESPONSE = 4  # Spoken assistance replies
    LLM_RESPONSE    = 4   # Backward compatibility alias
    STATUS          = 5   # Periodic scene status / "Path is clear"


# ── Structured World Model ────────────────────────────────────────────────────

@dataclass
class WorldState:
    """
    Read-only snapshot of current perception and spatial understanding.
    Used by LLM and Caregiver Dashboard.
    Ensures LLM receives only factual, verified environment data.
    """
    user_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (X, Y, Z) in metres
    heading_deg: float = 0.0
    localization_confidence: str = "HIGH"  # HIGH, MEDIUM, LOW, INVALID
    visible_objects: List[Dict[str, Any]] = field(default_factory=list)
    tracked_objects: List[Dict[str, Any]] = field(default_factory=list)
    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    nearest_obstacle: Optional[Dict[str, Any]] = None
    free_space_corridor: str = "CENTER_OPEN"  # LEFT_OPEN, CENTER_OPEN, RIGHT_OPEN, BLOCKED
    wall_ahead: bool = False
    destination: Optional[str] = None
    navigation_state: str = "IDLE"
    safety_level: str = "CLEAR"
    room: Optional[str] = None
    blind_mode: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_fact_string(self) -> str:
        """
        Convert world facts into a strict factual prompt section.
        Prevents LLM from hallucinating non-existent objects or coordinates.
        """
        facts = []
        if self.room:
            facts.append(f"Room: {self.room}")
        facts.append(f"Safety status: {self.safety_level}")
        facts.append(f"Corridor: {self.free_space_corridor}")
        
        if self.wall_ahead:
            facts.append("Wall directly ahead")
            
        if self.nearest_obstacle:
            obs = self.nearest_obstacle
            label = obs.get("class", "obstacle")
            pos = obs.get("position", "ahead")
            prox = obs.get("proximity", "close")
            facts.append(f"Nearest obstacle: {label} ({pos}, {prox})")
            
        if self.tracked_objects:
            obj_summaries = []
            for obj in self.tracked_objects[:5]:
                label = obj.get("class", "object")
                pos = obj.get("position", "in view")
                prox = obj.get("proximity", "medium")
                obj_summaries.append(f"{label} ({pos}, {prox})")
            facts.append("Identified objects: " + ", ".join(obj_summaries))
        elif self.visible_objects:
            names = [o.get("class", "object") for o in self.visible_objects[:4]]
            facts.append("Seeing: " + ", ".join(names))
        else:
            facts.append("No immediate obstacles detected. Path appears open.")

        if self.destination and self.navigation_state != "IDLE":
            facts.append(f"Navigating toward: {self.destination} (State: {self.navigation_state})")

        return " | ".join(facts)


# ── Central Thread-Safe System State ──────────────────────────────────────────

class SystemState:
    """
    Central thread-safe state container for all SURDAS subsystems.
    """

    def __init__(self):
        self._lock = threading.RLock()
        
        # Subsystem States
        self._voice_state = VoiceState.IDLE
        self._wakeword_state = WakeWordState.LISTENING
        self._tts_state = TTSState.IDLE
        self._vision_state = VisionState.ACTIVE
        self._navigation_state = NavigationState.IDLE
        self._safety_level = SafetyLevel.CLEAR
        self._localization_state = LocalizationState.TRACKING
        self._camera_state = CameraState.CONNECTED
        self._llm_state = LLMState.READY
        
        # Blind Mode Flag
        self._blind_mode = False
        
        # Subsystem Health Map: name -> (SubsystemHealth, last_updated_time, details)
        self._health: Dict[str, Dict[str, Any]] = {
            "camera": {"status": SubsystemHealth.OK, "time": time.time(), "details": ""},
            "microphone": {"status": SubsystemHealth.OK, "time": time.time(), "details": ""},
            "wakeword": {"status": SubsystemHealth.OK, "time": time.time(), "details": ""},
            "stt": {"status": SubsystemHealth.OK, "time": time.time(), "details": ""},
            "tts": {"status": SubsystemHealth.OK, "time": time.time(), "details": ""},
            "yolo": {"status": SubsystemHealth.OK, "time": time.time(), "details": ""},
            "midas": {"status": SubsystemHealth.OK, "time": time.time(), "details": ""},
            "mcu_bridge": {"status": SubsystemHealth.OK, "time": time.time(), "details": ""},
            "imu": {"status": SubsystemHealth.OK, "time": time.time(), "details": ""},
            "localization": {"status": SubsystemHealth.OK, "time": time.time(), "details": ""},
            "navigation": {"status": SubsystemHealth.OK, "time": time.time(), "details": ""},
            "spatial_memory": {"status": SubsystemHealth.OK, "time": time.time(), "details": ""},
        }

        # Current World Model Data
        self._world_state = WorldState()

        # Threading Synchronization Events
        self.wake_detected_event = threading.Event()
        self.barge_in_event = threading.Event()
        self.safety_alert_event = threading.Event()
        self.stop_requested_event = threading.Event()

    # ── State Accessors & Mutators (Thread-Safe) ──────────────────────────────

    # Voice State
    @property
    def voice_state(self) -> VoiceState:
        with self._lock:
            return self._voice_state

    @voice_state.setter
    def voice_state(self, val: VoiceState):
        with self._lock:
            self._voice_state = val

    # WakeWord State
    @property
    def wakeword_state(self) -> WakeWordState:
        with self._lock:
            return self._wakeword_state

    @wakeword_state.setter
    def wakeword_state(self, val: WakeWordState):
        with self._lock:
            self._wakeword_state = val

    # TTS State
    @property
    def tts_state(self) -> TTSState:
        with self._lock:
            return self._tts_state

    @tts_state.setter
    def tts_state(self, val: TTSState):
        with self._lock:
            self._tts_state = val

    # Vision State
    @property
    def vision_state(self) -> VisionState:
        with self._lock:
            return self._vision_state

    @vision_state.setter
    def vision_state(self, val: VisionState):
        with self._lock:
            self._vision_state = val

    # Navigation State
    @property
    def navigation_state(self) -> NavigationState:
        with self._lock:
            return self._navigation_state

    @navigation_state.setter
    def navigation_state(self, val: NavigationState):
        with self._lock:
            self._navigation_state = val

    # Safety Level
    @property
    def safety_level(self) -> SafetyLevel:
        with self._lock:
            return self._safety_level

    @safety_level.setter
    def safety_level(self, val: SafetyLevel):
        with self._lock:
            self._safety_level = val

    # Localization State
    @property
    def localization_state(self) -> LocalizationState:
        with self._lock:
            return self._localization_state

    @localization_state.setter
    def localization_state(self, val: LocalizationState):
        with self._lock:
            self._localization_state = val

    # Camera State
    @property
    def camera_state(self) -> CameraState:
        with self._lock:
            return self._camera_state

    @camera_state.setter
    def camera_state(self, val: CameraState):
        with self._lock:
            self._camera_state = val

    # LLM State
    @property
    def llm_state(self) -> LLMState:
        with self._lock:
            return self._llm_state

    @llm_state.setter
    def llm_state(self, val: LLMState):
        with self._lock:
            self._llm_state = val

    # Blind Mode
    @property
    def blind_mode(self) -> bool:
        with self._lock:
            return self._blind_mode

    @blind_mode.setter
    def blind_mode(self, enabled: bool):
        with self._lock:
            self._blind_mode = enabled

    # ── Health Tracking ───────────────────────────────────────────────────────

    def set_health(self, subsystem: str, status: SubsystemHealth, details: str = ""):
        with self._lock:
            if subsystem in self._health:
                self._health[subsystem] = {
                    "status": status,
                    "time": time.time(),
                    "details": details
                }

    def get_health(self, subsystem: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._health.get(subsystem, {"status": SubsystemHealth.FAILED, "time": time.time(), "details": "Unknown"}))

    def get_all_health(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._health.items()}

    # ── World State Management ────────────────────────────────────────────────

    def update_world(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._world_state, k):
                    setattr(self._world_state, k, v)
            self._world_state.timestamp = time.time()
            self._world_state.navigation_state = self._navigation_state.value
            self._world_state.safety_level = self._safety_level.value
            self._world_state.blind_mode = self._blind_mode

    def get_world_snapshot(self) -> WorldState:
        """Returns a detached copy of the WorldState for thread-safe consumption."""
        with self._lock:
            ws = self._world_state
            return WorldState(
                user_pose=ws.user_pose,
                heading_deg=ws.heading_deg,
                localization_confidence=ws.localization_confidence,
                visible_objects=list(ws.visible_objects),
                tracked_objects=list(ws.tracked_objects),
                obstacles=list(ws.obstacles),
                nearest_obstacle=dict(ws.nearest_obstacle) if ws.nearest_obstacle else None,
                free_space_corridor=ws.free_space_corridor,
                wall_ahead=ws.wall_ahead,
                destination=ws.destination,
                navigation_state=ws.navigation_state,
                safety_level=ws.safety_level,
                room=ws.room,
                blind_mode=ws.blind_mode,
                timestamp=ws.timestamp
            )

    # ── Snapshot Summary for Telemetry & GUI ──────────────────────────────────

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "voice_state": self._voice_state.value,
                "wakeword_state": self._wakeword_state.value,
                "tts_state": self._tts_state.value,
                "vision_state": self._vision_state.value,
                "navigation_state": self._navigation_state.value,
                "safety_level": self._safety_level.value,
                "localization_state": self._localization_state.value,
                "camera_state": self._camera_state.value,
                "llm_state": self._llm_state.value,
                "blind_mode": self._blind_mode,
                "wall_ahead": self._world_state.wall_ahead,
                "health": {k: v["status"].value for k, v in self._health.items()},
                "timestamp": time.time()
            }
