"""
test_location.py - Tests for GPS location provider

Tests coordinate validation, staleness detection, GPS jumps, etc.
WITHOUT requiring actual GPS hardware.
"""
import pytest
import time
from navigation.location import (
    LocationProvider,
    ManualLocation,
    MockLocationProvider,
    BridgeLocationProvider,
    InvalidCoordinateError,
    LocationUnavailableError,
    StaleLocationError,
)


class TestCoordinateValidation:
    """Test coordinate validation logic."""
    
    def test_valid_coordinates(self):
        """Valid coordinates should pass."""
        provider = LocationProvider()
        assert provider.validate_coordinates(17.3616, 78.4747) is True
        assert provider.validate_coordinates(0, 0) is True
        assert provider.validate_coordinates(-90, -180) is True
        assert provider.validate_coordinates(90, 180) is True
    
    def test_invalid_latitude(self):
        """Invalid latitude should raise error."""
        provider = LocationProvider()
        with pytest.raises(InvalidCoordinateError):
            provider.validate_coordinates(91, 0)  # > 90
        with pytest.raises(InvalidCoordinateError):
            provider.validate_coordinates(-91, 0)  # < -90
    
    def test_invalid_longitude(self):
        """Invalid longitude should raise error."""
        provider = LocationProvider()
        with pytest.raises(InvalidCoordinateError):
            provider.validate_coordinates(0, 181)  # > 180
        with pytest.raises(InvalidCoordinateError):
            provider.validate_coordinates(0, -181)  # < -180
    
    def test_nan_coordinates(self):
        """NaN coordinates should raise error."""
        provider = LocationProvider()
        with pytest.raises(InvalidCoordinateError):
            provider.validate_coordinates(float('nan'), 0)
        with pytest.raises(InvalidCoordinateError):
            provider.validate_coordinates(0, float('nan'))
    
    def test_infinite_coordinates(self):
        """Infinite coordinates should raise error."""
        provider = LocationProvider()
        with pytest.raises(InvalidCoordinateError):
            provider.validate_coordinates(float('inf'), 0)
        with pytest.raises(InvalidCoordinateError):
            provider.validate_coordinates(0, float('-inf'))


class TestManualLocation:
    """Test manual location provider (for testing/dev)."""
    
    def test_manual_location_basic(self):
        """Manual location should return fixed coordinates."""
        provider = ManualLocation(17.3616, 78.4747, "Test Location")
        loc = provider.get_location()
        
        assert loc['lat'] == 17.3616
        assert loc['lon'] == 78.4747
        assert loc['accuracy'] == 0.0
        assert loc['fix_quality'] == 'MANUAL'
        assert 'timestamp' in loc
    
    def test_manual_location_update(self):
        """Manual location should allow updates."""
        provider = ManualLocation(0, 0)
        provider.update(10.5, 20.5)
        loc = provider.get_location()
        
        assert loc['lat'] == 10.5
        assert loc['lon'] == 20.5
    
    def test_manual_location_invalid_coordinates(self):
        """Manual location should reject invalid coordinates."""
        with pytest.raises(InvalidCoordinateError):
            ManualLocation(100, 0)  # Invalid latitude


class TestMockLocationProvider:
    """Test mock location provider (for automated testing)."""
    
    def test_mock_single_waypoint(self):
        """Mock provider with single waypoint."""
        provider = MockLocationProvider([(17.3616, 78.4747)])
        loc = provider.get_location()
        
        assert loc['lat'] == 17.3616
        assert loc['lon'] == 78.4747
        assert loc['fix_quality'] == 'MOCK'
    
    def test_mock_multiple_waypoints(self):
        """Mock provider should advance through waypoints."""
        waypoints = [(0, 0), (1, 1), (2, 2)]
        provider = MockLocationProvider(waypoints)
        
        # First waypoint
        loc = provider.get_location()
        assert (loc['lat'], loc['lon']) == (0, 0)
        
        # Advance
        provider.advance()
        loc = provider.get_location()
        assert (loc['lat'], loc['lon']) == (1, 1)
        
        # Advance again
        provider.advance()
        loc = provider.get_location()
        assert (loc['lat'], loc['lon']) == (2, 2)
    
    def test_mock_reset(self):
        """Mock provider should reset to first waypoint."""
        waypoints = [(0, 0), (1, 1), (2, 2)]
        provider = MockLocationProvider(waypoints)
        
        provider.advance()
        provider.advance()
        provider.reset()
        
        loc = provider.get_location()
        assert (loc['lat'], loc['lon']) == (0, 0)
    
    def test_mock_invalid_waypoints(self):
        """Mock provider should reject invalid waypoints."""
        with pytest.raises(ValueError):
            MockLocationProvider([])  # Empty waypoints
        
        with pytest.raises(InvalidCoordinateError):
            MockLocationProvider([(100, 0)])  # Invalid coordinate


class TestHaversineDistance:
    """Test haversine distance calculation."""
    
    def test_same_point(self):
        """Distance to same point should be 0."""
        provider = LocationProvider()
        dist = provider.haversine_distance(0, 0, 0, 0)
        assert dist == 0.0
    
    def test_known_distance(self):
        """Test against known distance (approximately)."""
        provider = LocationProvider()
        # Distance from equator at 0° to equator at 1° longitude is ~111 km
        dist = provider.haversine_distance(0, 0, 0, 1)
        assert 110_000 < dist < 112_000  # ~111 km with some tolerance
    
    def test_latitude_difference(self):
        """Test latitude difference."""
        provider = LocationProvider()
        # 1 degree of latitude is always ~111 km
        dist = provider.haversine_distance(0, 0, 1, 0)
        assert 110_000 < dist < 112_000


class TestBridgeLocationProvider:
    """Test STM32 Bridge-fed GPS Location Provider."""

    def test_bridge_location_initial_state(self):
        provider = BridgeLocationProvider()
        assert not provider.is_fixed()
        with pytest.raises(LocationUnavailableError):
            provider.get_location()

    def test_bridge_location_valid_update(self):
        provider = BridgeLocationProvider()
        packet = {
            "fix": True,
            "lat": 17.385044,
            "lon": 78.486671,
            "sats": 9,
            "speed": 1.2
        }
        assert provider.update_from_bridge(packet) is True
        assert provider.is_fixed()

        loc = provider.get_location()
        assert loc["lat"] == 17.385044
        assert loc["lon"] == 78.486671
        assert loc["satellites"] == 9
        assert loc["fix_quality"] == "FIXED"
        assert loc["source"] == "STM32_GPS"

    def test_bridge_location_invalid_coords_rejected(self):
        provider = BridgeLocationProvider()
        bad_packet = {
            "fix": True,
            "lat": 95.0,  # invalid latitude > 90
            "lon": 78.486671
        }
        assert provider.update_from_bridge(bad_packet) is False
        assert not provider.is_fixed()

    def test_bridge_location_stale_detection(self):
        provider = BridgeLocationProvider(stale_threshold=0.05)
        packet = {"fix": True, "lat": 12.9716, "lon": 77.5946, "sats": 6}
        provider.update_from_bridge(packet)
        time.sleep(0.08)
        assert not provider.is_fixed()
        with pytest.raises(StaleLocationError):
            provider.get_location()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
