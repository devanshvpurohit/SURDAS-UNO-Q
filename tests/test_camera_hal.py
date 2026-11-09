"""
Unit tests for Camera Hardware Abstraction Layer (HAL).
Tests auto-discovery, mock fallback, failure handling, and frame acquisition.
"""

import pytest
import numpy as np
from hal.camera import (
    CameraProvider, MockCameraProvider, V4L2CameraProvider,
    create_camera_provider, scan_available_cameras
)


def test_mock_camera_provider_generates_frames():
    cam = MockCameraProvider(width=320, height=240, fps=15.0)
    assert cam.is_connected() is True
    
    frame = cam.get_frame()
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (240, 320, 3)
    
    status = cam.get_status()
    assert status["is_mock"] is True
    assert status["connected"] is True
    assert status["frame_count"] >= 1
    
    cam.release()
    assert cam.is_connected() is False
    assert cam.get_frame() is None


def test_v4l2_invalid_device_handles_gracefully():
    # Attempting to open a non-existent device should NOT crash the process
    cam = V4L2CameraProvider(device="/dev/nonexistent_video_device_xyz", target_fps=15.0)
    assert cam.is_connected() is False
    assert cam.get_frame() is None
    
    status = cam.get_status()
    assert status["connected"] is False
    assert status["is_mock"] is False
    
    cam.release()


def test_create_camera_provider_fallback_to_mock():
    # When requesting mock or when physical camera is missing with allow_mock=True
    cam = create_camera_provider(device="mock", allow_mock=True)
    assert isinstance(cam, MockCameraProvider)
    assert cam.is_connected() is True
    
    frame = cam.get_frame()
    assert frame is not None
    cam.release()


def test_create_camera_provider_strict_mode():
    # If invalid device and allow_mock is False, should raise RuntimeError or fail safely
    with pytest.raises(Exception):
        create_camera_provider(device="/dev/nonexistent_camera_device_123", allow_mock=False)


def test_scan_available_cameras_returns_list():
    devices = scan_available_cameras()
    assert isinstance(devices, list)
    for dev in devices:
        assert "id" in dev
        assert "type" in dev
