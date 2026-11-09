"""
Unit tests for Vision Hardware Abstraction Layer (HAL).
Tests object detection, relative depth estimation, OCR, and output validation.
"""

import numpy as np
from hal.vision import (
    MockObjectDetectionProvider, MockDepthProvider, MockOCRProvider
)


def test_mock_object_detection_validation():
    detector = MockObjectDetectionProvider()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    
    assert isinstance(detections, list)
    assert len(detections) > 0
    for det in detections:
        assert "label" in det
        assert isinstance(det["label"], str)
        assert "confidence" in det
        assert 0.0 <= det["confidence"] <= 1.0
        assert "box" in det
        assert len(det["box"]) == 4
        assert det["box"][0] < det["box"][2]
        assert det["box"][1] < det["box"][3]
        assert "class_id" in det


def test_mock_depth_provider_dimensions():
    depth_estimator = MockDepthProvider()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    depth_map = depth_estimator.estimate_depth(frame)
    
    assert depth_map is not None
    assert isinstance(depth_map, np.ndarray)
    assert depth_map.shape == (240, 320)
    # Bottom should be closer (higher value in inverse depth)
    assert depth_map[-1, 160] > depth_map[0, 160]


def test_mock_ocr_provider():
    ocr = MockOCRProvider(sample_text="ROOM 101")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    results = ocr.read_text(frame)
    
    assert len(results) == 1
    assert results[0]["text"] == "ROOM 101"
    assert results[0]["confidence"] > 0.9
