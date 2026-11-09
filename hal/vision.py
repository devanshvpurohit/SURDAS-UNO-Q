"""
HAL Vision Providers for SURDAS on Arduino UNO Q.
Encapsulates Object Detection (YOLO), Relative Depth (MiDaS), and OCR (EasyOCR),
with robust mock fallbacks for testing and low-resource environments.
"""

import abc
import time
import logging
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger("surdas.hal.vision")


# ─────────────────────────────────────────────────────────────────────────────
# 1. OBJECT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class ObjectDetectionProvider(abc.ABC):
    """Abstract provider for object detection."""

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in the given BGR frame.
        Returns:
            List of dicts: [
                {"label": str, "confidence": float, "box": [x1, y1, x2, y2], "class_id": int}
            ]
        """
        pass

    @abc.abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return provider status (model name, latency, device)."""
        pass


class YOLOObjectDetectionProvider(ObjectDetectionProvider):
    """Wraps YOLOv8 model for deterministic object detection."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.25, device: str = "cpu"):
        self.model_path = model_path
        self.conf_threshold = confidence_threshold
        self.device = device
        self.last_latency_ms = 0.0
        self.detection_count = 0
        self.model = None

        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info(f"Loaded YOLO model from {self.model_path} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.model is None or frame is None:
            return []

        t0 = time.time()
        try:
            results = self.model(frame, conf=self.conf_threshold, verbose=False, device=self.device)
            t1 = time.time()
            self.last_latency_ms = round((t1 - t0) * 1000.0, 1)

            detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        label = self.model.names[cls_id] if hasattr(self.model, "names") else str(cls_id)
                        detections.append({
                            "label": label,
                            "confidence": round(conf, 3),
                            "box": xyxy,
                            "class_id": cls_id
                        })
            self.detection_count += len(detections)
            return detections
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return []

    def get_status(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "device": self.device,
            "loaded": self.model is not None,
            "latency_ms": self.last_latency_ms,
            "is_mock": False
        }


class MockObjectDetectionProvider(ObjectDetectionProvider):
    """Mock detection provider returning deterministic synthetic objects for testing."""

    def __init__(self, detections: Optional[List[Dict[str, Any]]] = None):
        self.default_detections = detections if detections is not None else [
            {"label": "chair", "confidence": 0.88, "box": [100, 150, 260, 380], "class_id": 56},
            {"label": "person", "confidence": 0.92, "box": [300, 100, 450, 420], "class_id": 0}
        ]
        self.last_latency_ms = 1.0

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        return list(self.default_detections)

    def set_mock_detections(self, detections: List[Dict[str, Any]]) -> None:
        self.default_detections = detections

    def get_status(self) -> Dict[str, Any]:
        return {
            "model_path": "MOCK_YOLO",
            "device": "cpu",
            "loaded": True,
            "latency_ms": self.last_latency_ms,
            "is_mock": True
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. RELATIVE DEPTH ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

class DepthProvider(abc.ABC):
    """
    Abstract provider for relative depth estimation.
    NOTE: Outputs relative inverse depth (higher = closer), NOT metric distance.
    """

    @abc.abstractmethod
    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Produce a 2D float numpy array of relative inverse depth values.
        """
        pass

    @abc.abstractmethod
    def get_status(self) -> Dict[str, Any]:
        pass


class MiDaSDepthProvider(DepthProvider):
    """Wraps MiDaS Small relative depth model."""

    def __init__(self, cache_dir: str, device: str = "cpu"):
        self.cache_dir = cache_dir
        self.device = device
        self.midas = None
        self.transform = None
        self.last_latency_ms = 0.0

        self._load_model()

    def _load_model(self):
        try:
            import torch
            from pathlib import Path
            p = Path(self.cache_dir)
            if not p.exists():
                logger.warning(f"MiDaS cache not found at {self.cache_dir}")
                return

            midas_transforms = torch.hub.load(str(p), "transforms", source="local")
            self.transform = midas_transforms.small_transform
            self.midas = torch.hub.load(str(p), "MiDaS_small", source="local", pretrained=True).to(self.device).eval()
            logger.info("Loaded MiDaS Small depth model.")
        except Exception as e:
            logger.warning(f"Failed to load MiDaS model: {e}")
            self.midas = None

    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self.midas is None or frame is None:
            return None

        import torch
        t0 = time.time()
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img).to(self.device)
            with torch.no_grad():
                prediction = self.midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                depth_map = prediction.cpu().numpy()
            self.last_latency_ms = round((time.time() - t0) * 1000.0, 1)
            return depth_map
        except Exception as e:
            logger.error(f"Depth inference error: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "model": "MiDaS_small",
            "device": self.device,
            "loaded": self.midas is not None,
            "latency_ms": self.last_latency_ms,
            "is_mock": False
        }


class MockDepthProvider(DepthProvider):
    """Generates synthetic relative depth maps for testing."""

    def __init__(self):
        self.last_latency_ms = 0.5

    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if frame is None:
            return None
        h, w = frame.shape[:2]
        # Linear vertical gradient representing floor receding into distance
        # bottom is close (~500), top is far (~200)
        gradient = np.linspace(200, 500, h, dtype=np.float32)
        depth_map = np.tile(gradient[:, None], (1, w))
        return depth_map

    def get_status(self) -> Dict[str, Any]:
        return {
            "model": "MOCK_DEPTH",
            "device": "cpu",
            "loaded": True,
            "latency_ms": self.last_latency_ms,
            "is_mock": True
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. OPTICAL CHARACTER RECOGNITION (OCR)
# ─────────────────────────────────────────────────────────────────────────────

class OCRProvider(abc.ABC):
    """Abstract OCR provider."""

    @abc.abstractmethod
    def read_text(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract text from image frame.
        Returns: List of {"text": str, "confidence": float, "box": list}
        """
        pass

    @abc.abstractmethod
    def get_status(self) -> Dict[str, Any]:
        pass


class MockOCRProvider(OCRProvider):
    """Mock OCR provider for testing."""

    def __init__(self, sample_text: str = ""):
        self.sample_text = sample_text

    def read_text(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if not self.sample_text:
            return []
        return [{"text": self.sample_text, "confidence": 0.95, "box": [50, 50, 200, 100]}]

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine": "MOCK_OCR",
            "loaded": True,
            "is_mock": True
        }
