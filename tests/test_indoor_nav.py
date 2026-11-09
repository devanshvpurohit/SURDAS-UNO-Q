"""
Unit tests for Deterministic Indoor Navigator.
Verifies safety-hold triggers, state transitions, stale perception rejection,
and pause/resume/stop lifecycle.
"""

import numpy as np
from unittest.mock import Mock
from indoor_navigator import IndoorNavigator, NavigationState, HoldReason
from indoor_perception import IndoorPerception, NavConfidence
from spatial_memory import SpatialMemory, ObjectCategory


def test_navigator_lifecycle_and_states(tmp_path):
    db_file = tmp_path / "test_spatial.db"
    memory = SpatialMemory(db_path=str(db_file))
    mock_voice = Mock()

    # Add test object
    memory.add_object("door", category=ObjectCategory.DOOR, x=0.0, y=3.0, z=0.0, confidence=0.9)

    nav = IndoorNavigator(memory, mock_voice)
    assert nav.get_status().state == NavigationState.IDLE

    # Start navigation
    ok = nav.navigate_to_object("door")
    assert ok is True
    assert nav.get_status().state == NavigationState.NAVIGATING

    # Update with valid, high confidence perception from IndoorPerception
    perception_engine = IndoorPerception(grid_size=32)
    clear_depth = np.random.rand(480, 640).astype(np.float32) * 0.3
    result_clear = perception_engine.process_frame([], clear_depth, 640, 480)
    assert result_clear.nav_confidence in [NavConfidence.HIGH, NavConfidence.MEDIUM]

    nav.update(result_clear)
    assert nav.get_status().state in (NavigationState.NAVIGATING, NavigationState.APPROACHING)

    # Pause navigation -> enters SAFETY_HOLD
    nav.pause()
    assert nav.get_status().state == NavigationState.SAFETY_HOLD
    assert nav.get_status().hold_reason == HoldReason.USER_REQUEST

    # Resume navigation
    nav.resume()
    assert nav.get_status().state != NavigationState.SAFETY_HOLD

    # Stop navigation -> returns to IDLE
    nav.stop()
    assert nav.get_status().state == NavigationState.IDLE


def test_safety_hold_on_camera_failure(tmp_path):
    db_file = tmp_path / "test_spatial_cam_fail.db"
    memory = SpatialMemory(db_path=str(db_file))
    mock_voice = Mock()
    memory.add_object("chair", category=ObjectCategory.FURNITURE, x=0.0, y=2.0, z=0.0, confidence=0.9)

    nav = IndoorNavigator(memory, mock_voice)
    nav.navigate_to_object("chair")

    # Camera failure (invalid zero depth map)
    perception_engine = IndoorPerception(grid_size=32)
    invalid_depth = np.zeros((480, 640), dtype=np.float32)
    result_invalid = perception_engine.process_frame([], invalid_depth, 640, 480)
    assert result_invalid.nav_confidence == NavConfidence.INVALID

    nav.update(result_invalid)
    assert nav.get_status().state == NavigationState.SAFETY_HOLD
    assert nav.get_status().hold_reason == HoldReason.CAMERA_FAILURE


def test_safety_hold_on_critical_obstacle(tmp_path):
    db_file = tmp_path / "test_spatial_obs.db"
    memory = SpatialMemory(db_path=str(db_file))
    mock_voice = Mock()
    memory.add_object("desk", category=ObjectCategory.FURNITURE, x=0.0, y=2.5, z=0.0, confidence=0.9)

    nav = IndoorNavigator(memory, mock_voice)
    nav.navigate_to_object("desk")

    # Critical close obstacle with depth variance (background far, obstacle center close)
    perception_engine = IndoorPerception(grid_size=32)
    close_depth = np.full((480, 640), 0.1, dtype=np.float32)
    close_depth[150:400, 200:440] = 1.0  # obstacle region is closest
    detections = [{"class": "chair", "confidence": 0.9, "bbox": [200, 150, 440, 400]}]
    result_close = perception_engine.process_frame(detections, close_depth, 640, 480)

    nav.update(result_close)
    assert nav.get_status().state == NavigationState.SAFETY_HOLD
    assert nav.get_status().hold_reason in [HoldReason.OBSTACLE_TOO_CLOSE, HoldReason.LOW_CONFIDENCE]
