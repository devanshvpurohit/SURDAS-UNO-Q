"""
test_config.py - Tests for configuration system

Tests that configuration loads correctly and provides proper defaults.
"""
import pytest
import os
from pathlib import Path


def test_config_imports():
    """Test that config module imports without errors."""
    import config
    assert hasattr(config, 'PROJECT_ROOT')
    assert hasattr(config, 'GPS_ENABLED')
    assert hasattr(config, 'DEVICE')


def test_project_root_exists():
    """Test that PROJECT_ROOT is a valid directory."""
    from config import PROJECT_ROOT
    assert isinstance(PROJECT_ROOT, Path)
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()


def test_depth_thresholds():
    """Test that depth thresholds are properly defined."""
    from config import (
        DEPTH_VERY_CLOSE_THRESHOLD,
        DEPTH_CLOSE_THRESHOLD,
        DEPTH_MEDIUM_THRESHOLD,
    )
    
    # Thresholds should be in descending order (higher value = closer)
    assert DEPTH_VERY_CLOSE_THRESHOLD > DEPTH_CLOSE_THRESHOLD
    assert DEPTH_CLOSE_THRESHOLD > DEPTH_MEDIUM_THRESHOLD
    assert DEPTH_MEDIUM_THRESHOLD > 0


def test_classify_depth_proximity():
    """Test depth proximity classification."""
    from config import classify_depth_proximity
    
    # Test depth values relative to default thresholds (480, 420, 360)
    assert classify_depth_proximity(500) == "VERY_CLOSE"
    assert classify_depth_proximity(450) == "CLOSE"
    assert classify_depth_proximity(390) == "MEDIUM"
    assert classify_depth_proximity(200) == "FAR"


def test_get_proximity_description():
    """Test proximity description generation."""
    from config import get_proximity_description
    
    # English
    desc_en = get_proximity_description("VERY_CLOSE", "en")
    assert isinstance(desc_en, str)
    assert len(desc_en) > 0
    
    # Hindi
    desc_hi = get_proximity_description("VERY_CLOSE", "hi")
    assert isinstance(desc_hi, str)
    assert len(desc_hi) > 0


def test_gps_config():
    """Test GPS configuration values."""
    from config import (
        GPS_ENABLED,
        GPS_PORT,
        GPS_BAUD,
        GPS_TIMEOUT,
    )
    
    assert isinstance(GPS_ENABLED, bool)
    assert isinstance(GPS_PORT, str)
    assert isinstance(GPS_BAUD, int)
    assert GPS_BAUD > 0
    assert isinstance(GPS_TIMEOUT, float)
    assert GPS_TIMEOUT > 0


def test_navigation_thresholds():
    """Test navigation threshold values."""
    from config import (
        NAV_REROUTE_THRESHOLD_M,
        NAV_ARRIVAL_THRESHOLD_M,
        NAV_TURN_ANNOUNCE_DISTANCE_M,
    )
    
    assert NAV_REROUTE_THRESHOLD_M > 0
    assert NAV_ARRIVAL_THRESHOLD_M > 0
    assert NAV_TURN_ANNOUNCE_DISTANCE_M > 0
    # Turn announce should be greater than arrival threshold
    assert NAV_TURN_ANNOUNCE_DISTANCE_M > NAV_ARRIVAL_THRESHOLD_M


def test_get_device():
    """Test device selection."""
    from config import get_device
    device = get_device()
    assert device in ("cuda", "mps", "cpu")


def test_default_test_location():
    """Test default test location structure."""
    from config import DEFAULT_TEST_LOCATION
    
    assert isinstance(DEFAULT_TEST_LOCATION, dict)
    assert 'lat' in DEFAULT_TEST_LOCATION
    assert 'lon' in DEFAULT_TEST_LOCATION
    assert 'name' in DEFAULT_TEST_LOCATION
    
    # Validate coordinates
    assert -90 <= DEFAULT_TEST_LOCATION['lat'] <= 90
    assert -180 <= DEFAULT_TEST_LOCATION['lon'] <= 180


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
