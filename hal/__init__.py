"""
Hardware Abstraction Layer (HAL) for SURDAS on Arduino UNO Q.
Provides clean provider interfaces and implementations for cameras, sensors,
AI models, microcontroller bridges, and simulation fallbacks.
"""

from .camera import CameraProvider, V4L2CameraProvider, MockCameraProvider, create_camera_provider
from .vision import (
    ObjectDetectionProvider, YOLOObjectDetectionProvider, MockObjectDetectionProvider,
    DepthProvider, MiDaSDepthProvider, MockDepthProvider,
    OCRProvider, MockOCRProvider
)
from .sensor import IMUReading, IMUProvider, BridgeIMUProvider, SimulatedIMUProvider, create_imu_provider
from .bridge import MicrocontrollerBridge, UnoQBridge, MockMicrocontrollerBridge

__all__ = [
    "CameraProvider", "V4L2CameraProvider", "MockCameraProvider", "create_camera_provider",
    "ObjectDetectionProvider", "YOLOObjectDetectionProvider", "MockObjectDetectionProvider",
    "DepthProvider", "MiDaSDepthProvider", "MockDepthProvider",
    "OCRProvider", "MockOCRProvider",
    "IMUReading", "IMUProvider", "BridgeIMUProvider", "SimulatedIMUProvider", "create_imu_provider",
    "MicrocontrollerBridge", "UnoQBridge", "MockMicrocontrollerBridge"
]
