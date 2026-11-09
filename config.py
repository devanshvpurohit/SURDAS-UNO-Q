"""
config.py - Centralized configuration for SURDAS

All settings, thresholds, and magic numbers are defined here.
Can be overridden via environment variables.
"""
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
MAPS_DIR = PROJECT_ROOT / "navigation" / "data" / "maps"
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA SETTINGS (Arduino UNO Q Linux / V4L2)
# ─────────────────────────────────────────────────────────────────────────────
# No ESP32 dependency. Supports local USB/V4L2 camera auto-discovery and mock fallback.
CAMERA_DEVICE = os.getenv("SURDAS_CAMERA_DEVICE", "auto")  # "auto", "/dev/video0", 0, or "mock"
CAMERA_TARGET_FPS = float(os.getenv("SURDAS_CAMERA_FPS", "15.0"))
CAMERA_WIDTH = int(os.getenv("SURDAS_CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("SURDAS_CAMERA_HEIGHT", "480"))
ALLOW_MOCK_CAMERA = os.getenv("SURDAS_ALLOW_MOCK_CAMERA", "true").lower() == "true"

# Backward compatibility alias for legacy references
CAMERA_FALLBACK = 0

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE DEVICE
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = os.getenv("SURDAS_DEVICE", "auto")  # "cuda", "mps", "cpu", or "auto"

# ─────────────────────────────────────────────────────────────────────────────
# VISION - YOLO SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("SURDAS_YOLO_CONF", "0.25"))
YOLO_MODEL_PATH = str(MODELS_DIR / "yolov8n_india_ft.pt")
YOLO_MODEL_FALLBACK = "yolov8n.pt"  # Stock model fallback

# ─────────────────────────────────────────────────────────────────────────────
# VISION - MIDAS DEPTH ESTIMATION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: MiDaS produces RELATIVE inverse depth, NOT metric distance
# Higher values = closer to camera
# These thresholds are based on typical MiDaS_small output range (~200-600)

# Depth thresholds (higher value = closer to camera in MiDaS output)
# Typical MiDaS range: 200-600, mean ~400
DEPTH_VERY_CLOSE_THRESHOLD = float(os.getenv("SURDAS_DEPTH_VERY_CLOSE", "480"))  # Top 20% = very close
DEPTH_CLOSE_THRESHOLD = float(os.getenv("SURDAS_DEPTH_CLOSE", "420"))            # Top 40% = close
DEPTH_MEDIUM_THRESHOLD = float(os.getenv("SURDAS_DEPTH_MEDIUM", "360"))          # Top 60% = medium
# Below 360 = FAR

# Wall detection thresholds (for dense depth analysis)
DEPTH_WALL_DENSE_THRESHOLD = float(os.getenv("SURDAS_DEPTH_WALL_DENSE", "450"))
DEPTH_WALL_CENTER_RATIO = float(os.getenv("SURDAS_DEPTH_WALL_RATIO", "0.38"))
DEPTH_WALL_IMMEDIATE_THRESHOLD = float(os.getenv("SURDAS_DEPTH_WALL_IMMEDIATE", "500"))

# Adaptive depth normalization (automatically adjust to observed range)
DEPTH_ADAPTIVE_NORMALIZATION = os.getenv("SURDAS_DEPTH_ADAPTIVE", "true").lower() == "true"
DEPTH_PERCENTILE_VERY_CLOSE = 80  # 80th percentile = very close
DEPTH_PERCENTILE_CLOSE = 60       # 60th percentile = close
DEPTH_PERCENTILE_MEDIUM = 40      # 40th percentile = medium

# MiDaS model paths
MIDAS_MODEL_PATH = str(MODELS_DIR / "midas_small_india_ft.pt")
MIDAS_CACHE_DIR = Path.home() / ".cache" / "torch" / "hub" / "intel-isl_MiDaS_master"

# ─────────────────────────────────────────────────────────────────────────────
# SAFETY PERCEPTION
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: Increased cooldowns to prevent continuous TTS that blocks wake-word detection
# Safety announcements (critical obstacles) still fire frequently (3s)
# Routine announcements (path clear, objects detected) reduced to 8s minimum
# This ensures large listening windows for wake-word detection
SAFETY_ANNOUNCEMENT_COOLDOWN = float(os.getenv("SURDAS_SAFETY_COOLDOWN", "3.0"))  # seconds - critical warnings
ROUTINE_ANNOUNCEMENT_COOLDOWN = float(os.getenv("SURDAS_ROUTINE_COOLDOWN", "8.0"))  # seconds - increased from 4.0
LONG_ANNOUNCEMENT_COOLDOWN = float(os.getenv("SURDAS_LONG_COOLDOWN", "12.0"))  # seconds - increased from 8.0

# ─────────────────────────────────────────────────────────────────────────────
# GPS / LOCATION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
GPS_ENABLED = os.getenv("SURDAS_GPS_ENABLED", "false").lower() == "true"
GPS_PORT = os.getenv("SURDAS_GPS_PORT", "/dev/ttyUSB0")
GPS_BAUD = int(os.getenv("SURDAS_GPS_BAUD", "9600"))
GPS_TIMEOUT = float(os.getenv("SURDAS_GPS_TIMEOUT", "10.0"))  # seconds
GPS_STALE_THRESHOLD = float(os.getenv("SURDAS_GPS_STALE", "5.0"))  # seconds
GPS_ACCURACY_MAX = float(os.getenv("SURDAS_GPS_ACCURACY_MAX", "50.0"))  # meters

# Coordinate validation
GPS_MIN_LAT = -90.0
GPS_MAX_LAT = 90.0
GPS_MIN_LON = -180.0
GPS_MAX_LON = 180.0

# GPS jump detection (reject sudden unrealistic position changes)
GPS_MAX_JUMP_DISTANCE_M = float(os.getenv("SURDAS_GPS_MAX_JUMP", "500.0"))  # meters

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
NAV_REROUTE_THRESHOLD_M = float(os.getenv("SURDAS_NAV_REROUTE", "50.0"))
NAV_TURN_ANNOUNCE_DISTANCE_M = float(os.getenv("SURDAS_NAV_TURN_ANNOUNCE", "50.0"))
NAV_TURN_WARN_DISTANCE_M = float(os.getenv("SURDAS_NAV_TURN_WARN", "15.0"))
NAV_ARRIVAL_THRESHOLD_M = float(os.getenv("SURDAS_NAV_ARRIVAL", "15.0"))
NAV_PROGRESS_ANNOUNCE_MIN_M = float(os.getenv("SURDAS_NAV_PROGRESS_MIN", "100.0"))

# ─────────────────────────────────────────────────────────────────────────────
# VOICE & DETERMINISTIC COMMAND ROUTING SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# Ollama and all LLM dependencies are DISABLED for this Arduino UNO Q phase.
LLM_ENABLED = False
LLM_DEFAULT_MODEL = ""
LLM_BASE_URL = ""

# Voice activity detection
VAD_SAMPLE_RATE = int(os.getenv("SURDAS_VAD_SAMPLE_RATE", "16000"))
VAD_CHUNK_SIZE = int(os.getenv("SURDAS_VAD_CHUNK_SIZE", "512"))

# ─────────────────────────────────────────────────────────────────────────────
# SENSOR / IMU SETTINGS (Arduino UNO Q MCU)
# ─────────────────────────────────────────────────────────────────────────────
IMU_SOURCE = os.getenv("SURDAS_IMU_SOURCE", "auto")  # "bridge", "simulation", "none"
IMU_SIMULATION_ENABLED = os.getenv("SURDAS_IMU_SIMULATION", "true").lower() == "true"
FALL_DETECTION_ENABLED = os.getenv("SURDAS_FALL_DETECTION_ENABLED", "true").lower() == "true"

# ─────────────────────────────────────────────────────────────────────────────
# ARDUINO UNO Q BRIDGE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
UNO_Q_BRIDGE_PORT = os.getenv("SURDAS_BRIDGE_PORT", "")  # auto-detected if empty
UNO_Q_BRIDGE_BAUD = int(os.getenv("SURDAS_BRIDGE_BAUD", "115200"))
UNO_Q_ENABLE_MOCK_BRIDGE = os.getenv("SURDAS_MOCK_BRIDGE", "true").lower() == "true"

# ─────────────────────────────────────────────────────────────────────────────
# CURRENCY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
CURRENCY_HISTORY_SIZE = int(os.getenv("SURDAS_CURRENCY_HISTORY", "6"))
CURRENCY_MIN_VOTES = int(os.getenv("SURDAS_CURRENCY_MIN_VOTES", "2"))

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("SURDAS_LOG_LEVEL", "INFO").upper()
LOG_FILE = LOGS_DIR / "surdas.log"

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
VISION_FPS_TARGET = float(os.getenv("SURDAS_VISION_FPS", "15.0"))
DEPTH_FPS_TARGET = float(os.getenv("SURDAS_DEPTH_FPS", "10.0"))

# ─────────────────────────────────────────────────────────────────────────────
# DEVELOPMENT / TESTING
# ─────────────────────────────────────────────────────────────────────────────
# Default manual location for testing (only used when GPS is disabled)
DEFAULT_TEST_LOCATION = {
    "lat": float(os.getenv("SURDAS_TEST_LAT", "17.3616")),
    "lon": float(os.getenv("SURDAS_TEST_LON", "78.4747")),
    "name": os.getenv("SURDAS_TEST_LOCATION_NAME", "Hyderabad (Test Location)")
}

HEADLESS_MODE = os.getenv("SURDAS_HEADLESS", "false").lower() == "true"

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    """Determine best available compute device."""
    import torch
    
    if DEVICE.lower() != "auto":
        return DEVICE.lower()
    
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_yolo_model_path():
    """Get YOLO model path with fallback."""
    from pathlib import Path
    
    ft_path = Path(YOLO_MODEL_PATH)
    if ft_path.exists():
        return str(ft_path)
    
    fallback = PROJECT_ROOT / YOLO_MODEL_FALLBACK
    if fallback.exists():
        return str(fallback)
    
    # Last resort: use model name directly (will download)
    return YOLO_MODEL_FALLBACK


def get_midas_model_path():
    """Get MiDaS fine-tuned model path if available."""
    from pathlib import Path
    
    ft_path = Path(MIDAS_MODEL_PATH)
    if ft_path.exists():
        return str(ft_path)
    
    return None  # Will use pretrained


def classify_depth_proximity(depth_value: float) -> str:
    """
    Classify MiDaS relative depth into human-readable proximity category.
    
    IMPORTANT: depth_value is RELATIVE inverse depth, not metric distance.
    Higher values = closer to camera.
    
    MiDaS_small typically outputs values in range ~200-600, with mean ~400.
    
    Args:
        depth_value: Raw MiDaS depth value
        
    Returns:
        Proximity category: "VERY_CLOSE", "CLOSE", "MEDIUM", or "FAR"
    """
    if depth_value >= DEPTH_VERY_CLOSE_THRESHOLD:
        return "VERY_CLOSE"
    elif depth_value >= DEPTH_CLOSE_THRESHOLD:
        return "CLOSE"
    elif depth_value >= DEPTH_MEDIUM_THRESHOLD:
        return "MEDIUM"
    else:
        return "FAR"


def classify_depth_proximity_adaptive(depth_value: float, depth_map) -> str:
    """
    Classify depth using adaptive percentile-based thresholds.
    Automatically adjusts to the actual depth distribution in the scene.
    
    Args:
        depth_value: Raw MiDaS depth value to classify
        depth_map: Full depth map (numpy array) for computing percentiles
        
    Returns:
        Proximity category: "VERY_CLOSE", "CLOSE", "MEDIUM", or "FAR"
    """
    import numpy as np
    
    # Compute percentile thresholds from actual depth distribution
    p_very_close = np.percentile(depth_map, DEPTH_PERCENTILE_VERY_CLOSE)
    p_close = np.percentile(depth_map, DEPTH_PERCENTILE_CLOSE)
    p_medium = np.percentile(depth_map, DEPTH_PERCENTILE_MEDIUM)
    
    if depth_value >= p_very_close:
        return "VERY_CLOSE"
    elif depth_value >= p_close:
        return "CLOSE"
    elif depth_value >= p_medium:
        return "MEDIUM"
    else:
        return "FAR"


def get_proximity_description(proximity: str, lang: str = "en") -> str:
    """
    Get human-readable description of proximity category.
    
    Args:
        proximity: Proximity category from classify_depth_proximity()
        lang: Language code ("en" or "hi")
        
    Returns:
        Human-readable description
    """
    descriptions = {
        "en": {
            "VERY_CLOSE": "very close",
            "CLOSE": "nearby",
            "MEDIUM": "at medium distance",
            "FAR": "in the distance"
        },
        "hi": {
            "VERY_CLOSE": "बहुत पास",
            "CLOSE": "पास में",
            "MEDIUM": "मध्यम दूरी पर",
            "FAR": "दूर"
        }
    }
    return descriptions.get(lang, descriptions["en"]).get(proximity, proximity.lower())


if __name__ == "__main__":
    print("SURDAS Configuration")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Device: {get_device()}")
    print(f"YOLO Model: {get_yolo_model_path()}")
    print(f"GPS Enabled: {GPS_ENABLED}")
    print(f"Log Level: {LOG_LEVEL}")
    print(f"Default Test Location: {DEFAULT_TEST_LOCATION['name']}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# XTTS-V2 VOICE CLONING SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
XTTS_ENABLED = os.getenv("XTTS_ENABLED", "true").lower() == "true"
XTTS_MODEL = os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
XTTS_LANGUAGE = os.getenv("XTTS_LANGUAGE", "en")
XTTS_REFERENCE_WAV = os.getenv("XTTS_REFERENCE_WAV", str(PROJECT_ROOT / "voice_models" / "surdas_reference.wav"))
XTTS_WARMUP = os.getenv("XTTS_WARMUP", "true").lower() == "true"
XTTS_USE_GPU = os.getenv("XTTS_USE_GPU", "false").lower() == "true"
