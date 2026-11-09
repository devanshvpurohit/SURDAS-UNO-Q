"""
navigation/location.py — GPS / Location abstraction for SURDAS.

Provides a unified LocationProvider interface that works with:
  1. ManualLocation  — hardcoded lat/lon (dev/testing ONLY)
  2. GPSSerialLocation — NMEA sentences from USB/Bluetooth GPS or ESP32
  3. LocationBridge   — HTTP polling from a phone GPS bridge app
  4. MockLocationProvider — for automated testing with simulated movement

Usage:
    # For production (real GPS):
    provider = GPSSerialLocation(port="/dev/ttyUSB0")
    provider.start()
    
    # For testing (manual coordinates):
    provider = ManualLocation(lat=17.3616, lon=78.4747)
    
    loc = provider.get_location()
    # {"lat": 17.3616, "lon": 78.4747, "accuracy": 0.0, "timestamp": ...}

CLI test:
    python -m navigation.location --lat 17.3616 --lon 78.4747
"""
from __future__ import annotations

import time
import threading
import math
from typing import Optional


class LocationUnavailableError(Exception):
    """Raised when location cannot be determined."""


class InvalidCoordinateError(Exception):
    """Raised when coordinates are invalid or out of range."""


class StaleLocationError(Exception):
    """Raised when location data is too old."""


class LocationProvider:
    """Base class. Override get_location()."""

    def get_location(self) -> dict:
        raise LocationUnavailableError(
            "No location provider configured. "
            "Use ManualLocation(lat, lon) for testing, "
            "or GPSSerialLocation() for a hardware GPS module."
        )

    @staticmethod
    def validate_coordinates(lat: float, lon: float) -> bool:
        """
        Validate that coordinates are within valid ranges.
        
        Returns True if valid, raises InvalidCoordinateError otherwise.
        """
        if not (-90.0 <= lat <= 90.0):
            raise InvalidCoordinateError(
                f"Invalid latitude: {lat}. Must be between -90 and 90 degrees."
            )
        if not (-180.0 <= lon <= 180.0):
            raise InvalidCoordinateError(
                f"Invalid longitude: {lon}. Must be between -180 and 180 degrees."
            )
        if math.isnan(lat) or math.isnan(lon):
            raise InvalidCoordinateError("Coordinates contain NaN values.")
        if math.isinf(lat) or math.isinf(lon):
            raise InvalidCoordinateError("Coordinates contain infinite values.")
        return True

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in meters."""
        R = 6_371_000.0  # Earth radius in meters
        φ1, φ2 = math.radians(lat1), math.radians(lat2)
        Δφ = math.radians(lat2 - lat1)
        Δλ = math.radians(lon2 - lon1)
        a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _make_result(lat: float, lon: float, accuracy: float = 0.0, 
                     fix_quality: str = "UNKNOWN") -> dict:
        """
        Create a standardized location result dict.
        
        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            accuracy: Estimated accuracy in meters (0.0 = exact/manual)
            fix_quality: GPS fix quality ("FIXED", "WEAK", "LOST", "MANUAL", "MOCK", "UNKNOWN")
        """
        return {
            "lat": lat,
            "lon": lon,
            "accuracy": accuracy,
            "timestamp": time.time(),
            "fix_quality": fix_quality,
        }


# ── Bridge / STM32 GPS Location ──────────────────────────────────────────────

class BridgeLocationProvider(LocationProvider):
    """
    LocationProvider that receives real-time GPS fixes from the STM32 microcontroller
    via the UnoQBridge telemetry stream.
    
    Features:
    - Zero serial contention (uses existing Linux-MCU bridge)
    - Validates incoming coordinates and fix flags
    - Detects stale GPS fixes when satellites are lost
    - Reports satellite count and calculated accuracy
    """

    def __init__(self, stale_threshold: float = 5.0):
        self._stale_threshold = stale_threshold
        self._lat: Optional[float] = None
        self._lon: Optional[float] = None
        self._accuracy: float = 99.0
        self._fix_quality: str = "LOST"
        self._satellites: int = 0
        self._speed_knots: float = 0.0
        self._last_fix_time: float = 0.0
        self._lock = threading.Lock()

    def update_from_bridge(self, gps_data: dict) -> bool:
        """
        Process GPS payload received in STM32 telemetry packet.
        Expected format:
            {"fix": true, "lat": 17.385044, "lon": 78.486671, "sats": 8, "speed": 0.0}
        """
        if not isinstance(gps_data, dict):
            return False

        has_fix = bool(gps_data.get("fix", False))
        lat = gps_data.get("lat")
        lon = gps_data.get("lon")
        sats = int(gps_data.get("sats", 0))
        speed = float(gps_data.get("speed", 0.0))

        if has_fix and lat is not None and lon is not None:
            try:
                self.validate_coordinates(float(lat), float(lon))
                with self._lock:
                    self._lat = float(lat)
                    self._lon = float(lon)
                    self._satellites = sats
                    self._speed_knots = speed
                    self._fix_quality = "FIXED" if sats >= 4 else "WEAK"
                    self._accuracy = max(1.5, round(30.0 / max(1, sats), 1))
                    self._last_fix_time = time.time()
                return True
            except (InvalidCoordinateError, ValueError):
                with self._lock:
                    self._fix_quality = "LOST"
                return False
        else:
            with self._lock:
                self._fix_quality = "LOST"
            return False

    def get_location(self) -> dict:
        now = time.time()
        with self._lock:
            if self._lat is None or self._lon is None:
                raise LocationUnavailableError("No GPS fix received from STM32 microcontroller.")
            if now - self._last_fix_time > self._stale_threshold:
                raise StaleLocationError(
                    f"STM32 GPS fix is stale ({now - self._last_fix_time:.1f}s old). Satellite signal lost."
                )
            return {
                "lat": self._lat,
                "lon": self._lon,
                "accuracy": self._accuracy,
                "timestamp": self._last_fix_time,
                "fix_quality": self._fix_quality,
                "satellites": self._satellites,
                "speed_knots": self._speed_knots,
                "source": "STM32_GPS"
            }

    def is_fixed(self) -> bool:
        with self._lock:
            return (
                self._fix_quality in ("FIXED", "WEAK")
                and (time.time() - self._last_fix_time <= self._stale_threshold)
            )


# ── Manual / Test location ────────────────────────────────────────────────────

class ManualLocation(LocationProvider):
    """
    Fixed coordinates for testing without GPS hardware.
    
    ⚠️ WARNING: This is for TESTING ONLY. Do NOT use in production navigation.
    
    The accuracy is reported as 0.0 (exact) and fix_quality as "MANUAL"
    so the navigation system knows this is not real GPS data.
    
    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        name: Optional human-readable name for logging
    """

    def __init__(self, lat: float, lon: float, name: str = "Manual Test Location"):
        self.validate_coordinates(lat, lon)
        self._lat = lat
        self._lon = lon
        self._name = name
        print(f"[LOCATION] ⚠️  Using MANUAL location: {name} ({lat:.4f}, {lon:.4f})")
        print(f"[LOCATION] ⚠️  This is for TESTING ONLY. Real navigation requires GPS.")

    def get_location(self) -> dict:
        return self._make_result(self._lat, self._lon, accuracy=0.0, fix_quality="MANUAL")

    def update(self, lat: float, lon: float):
        """Allow live update (e.g. from test script simulating movement)."""
        self.validate_coordinates(lat, lon)
        self._lat = lat
        self._lon = lon


# ── Serial GPS (NMEA) ─────────────────────────────────────────────────────────

class GPSSerialLocation(LocationProvider):
    """
    Reads NMEA sentences from a serial GPS module (USB/Bluetooth/ESP32).
    Parses $GPRMC and $GPGGA sentences with validation.

    Features:
    - Coordinate validation
    - Stale fix detection
    - GPS jump protection
    - Fix quality tracking
    - Accuracy estimation from HDOP

    Args:
        port:     Serial port path, e.g. "/dev/ttyUSB0" or "/dev/cu.usbserial-0001"
        baudrate: GPS baud rate (most modules: 9600)
        timeout:  Seconds to wait for a valid fix before raising LocationUnavailableError
        stale_threshold: Seconds before a fix is considered stale
        max_jump_distance: Maximum allowed position jump in meters (protects against GPS glitches)
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        timeout: float = 10.0,
        stale_threshold: float = 5.0,
        max_jump_distance: float = 500.0,
    ):
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._stale_threshold = stale_threshold
        self._max_jump_distance = max_jump_distance
        
        self._lat: Optional[float] = None
        self._lon: Optional[float] = None
        self._accuracy: float = 99.0
        self._fix_quality: str = "LOST"
        self._ts: float = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Track previous position for jump detection
        self._prev_lat: Optional[float] = None
        self._prev_lon: Optional[float] = None

    def start(self):
        """Start background NMEA reader thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="GPSReader")
        self._thread.start()
        print(f"[GPS] Started GPS reader on {self._port}")

    def stop(self):
        """Stop background GPS reader."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[GPS] GPS reader stopped")

    def _reader_loop(self):
        try:
            import serial  # pyserial — optional dep
            ser = serial.Serial(self._port, self._baudrate, timeout=1)
            print(f"[GPS] Serial port opened: {self._port} @ {self._baudrate} baud")
            
            while self._running:
                try:
                    line = ser.readline().decode("ascii", errors="replace").strip()
                    if line:
                        self._parse_nmea(line)
                except Exception as e:
                    # Don't crash on individual parse errors
                    pass
                    
            ser.close()
            
        except ImportError:
            print("[GPS] ❌ pyserial not installed. Install with: pip install pyserial")
            self._running = False
        except serial.SerialException as e:
            print(f"[GPS] ❌ Serial port error: {e}")
            print(f"[GPS] Check that {self._port} exists and is not in use by another program.")
            self._running = False
        except Exception as e:
            print(f"[GPS] ❌ GPS reader error: {e}")
            self._running = False

    def _parse_nmea(self, sentence: str):
        """Parse $GPRMC or $GPGGA sentences with validation."""
        try:
            if sentence.startswith("$GPRMC"):
                parts = sentence.split(",")
                if len(parts) < 7 or parts[2] != "A":  # A = valid fix
                    self._update_fix_quality("LOST")
                    return
                    
                lat = self._nmea_coord(parts[3], parts[4])
                lon = self._nmea_coord(parts[5], parts[6])
                
                if lat is not None and lon is not None:
                    self._update_position(lat, lon, accuracy=15.0, fix_quality="FIXED")

            elif sentence.startswith("$GPGGA"):
                parts = sentence.split(",")
                if len(parts) < 10 or parts[6] == "0":  # fix quality 0 = invalid
                    self._update_fix_quality("LOST")
                    return
                    
                lat = self._nmea_coord(parts[2], parts[3])
                lon = self._nmea_coord(parts[4], parts[5])
                hdop = float(parts[8]) if parts[8] else 99.0
                
                if lat is not None and lon is not None:
                    # Estimate accuracy from HDOP (horizontal dilution of precision)
                    # HDOP < 2 = excellent, 2-5 = good, 5-10 = moderate, >10 = poor
                    accuracy = hdop * 5.0  # rough estimate in meters
                    fix_quality = "FIXED" if hdop < 5.0 else "WEAK"
                    self._update_position(lat, lon, accuracy=accuracy, fix_quality=fix_quality)
                    
        except Exception as e:
            # Ignore malformed sentences
            pass

    def _update_position(self, lat: float, lon: float, accuracy: float, fix_quality: str):
        """Update position with validation and jump detection."""
        try:
            # Validate coordinates
            self.validate_coordinates(lat, lon)
            
            # Check for unrealistic GPS jumps (protects against glitches)
            if self._prev_lat is not None and self._prev_lon is not None:
                distance = self.haversine_distance(self._prev_lat, self._prev_lon, lat, lon)
                if distance > self._max_jump_distance:
                    print(f"[GPS] ⚠️  Rejected position jump of {distance:.0f}m (max: {self._max_jump_distance:.0f}m)")
                    return
            
            # Valid position - update state
            with self._lock:
                self._lat, self._lon = lat, lon
                self._accuracy = accuracy
                self._fix_quality = fix_quality
                self._ts = time.time()
                self._prev_lat, self._prev_lon = lat, lon
                
        except InvalidCoordinateError as e:
            print(f"[GPS] ⚠️  Invalid coordinates rejected: {e}")
            self._update_fix_quality("INVALID")

    def _update_fix_quality(self, quality: str):
        """Update fix quality without changing position."""
        with self._lock:
            self._fix_quality = quality

    @staticmethod
    def _nmea_coord(raw: str, direction: str) -> Optional[float]:
        """Convert NMEA ddmm.mmmm format to decimal degrees."""
        if not raw or not direction:
            return None
        try:
            dot = raw.index(".")
            degrees = float(raw[:dot - 2])
            minutes = float(raw[dot - 2:])
            decimal = degrees + minutes / 60.0
            if direction in ("S", "W"):
                decimal = -decimal
            return decimal
        except Exception:
            return None

    def get_location(self) -> dict:
        """
        Get current GPS location with staleness and quality checks.
        
        Raises:
            LocationUnavailableError: If no fix available within timeout
            StaleLocationError: If fix is too old
        """
        deadline = time.time() + self._timeout
        
        while time.time() < deadline:
            with self._lock:
                # Check if we have a position
                if self._lat is not None and self._lon is not None:
                    # Check staleness
                    age = time.time() - self._ts
                    if age > self._stale_threshold:
                        raise StaleLocationError(
                            f"GPS fix is {age:.1f}s old (stale threshold: {self._stale_threshold}s). "
                            "GPS may have lost signal."
                        )
                    
                    # Check fix quality
                    if self._fix_quality in ("LOST", "INVALID"):
                        continue  # Keep waiting for valid fix
                    
                    return self._make_result(
                        self._lat, self._lon, 
                        self._accuracy, 
                        self._fix_quality
                    )
            
            time.sleep(0.2)
        
        # Timeout reached
        with self._lock:
            quality = self._fix_quality
            
        raise LocationUnavailableError(
            f"GPS serial port {self._port}: no valid fix within {self._timeout}s. "
            f"Current status: {quality}. "
            "Check cable/port and that the GPS module has clear sky visibility."
        )

    def get_status(self) -> dict:
        """Get current GPS status without raising exceptions."""
        with self._lock:
            age = time.time() - self._ts if self._ts > 0 else float('inf')
            return {
                "has_fix": self._lat is not None and self._lon is not None,
                "fix_quality": self._fix_quality,
                "accuracy": self._accuracy,
                "age_seconds": age,
                "is_stale": age > self._stale_threshold,
            }


# ── HTTP Bridge (phone GPS) ───────────────────────────────────────────────────

class LocationBridge(LocationProvider):
    """
    Poll a local HTTP endpoint for GPS data.
    Useful when a phone runs a GPS-to-HTTP bridge app on the same Wi-Fi.

    The endpoint must return JSON: {"lat": ..., "lon": ..., "accuracy": ...}
    
    Args:
        url: HTTP endpoint URL
        timeout: Request timeout in seconds
    """

    def __init__(self, url: str = "http://192.168.4.1:8080/gps", timeout: float = 3.0):
        self._url = url
        self._timeout = timeout

    def get_location(self) -> dict:
        import urllib.request
        import json
        try:
            with urllib.request.urlopen(self._url, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode())
                lat = float(data["lat"])
                lon = float(data["lon"])
                
                # Validate coordinates
                self.validate_coordinates(lat, lon)
                
                return self._make_result(
                    lat, lon,
                    float(data.get("accuracy", 10.0)),
                    fix_quality="BRIDGE"
                )
        except InvalidCoordinateError as e:
            raise LocationUnavailableError(f"Invalid coordinates from GPS bridge: {e}")
        except Exception as e:
            raise LocationUnavailableError(f"GPS bridge at {self._url} unreachable: {e}")


# ── Mock Location Provider (for automated testing) ────────────────────────────

class MockLocationProvider(LocationProvider):
    """
    Mock location provider for automated testing.
    Can simulate movement along a predefined path.
    
    Args:
        waypoints: List of (lat, lon) tuples representing a path
        speed_mps: Simulated movement speed in meters per second
        loop: Whether to loop back to start after reaching the end
    """
    
    def __init__(self, waypoints: list[tuple[float, float]], speed_mps: float = 1.4, loop: bool = False):
        if not waypoints:
            raise ValueError("MockLocationProvider requires at least one waypoint")
        
        # Validate all waypoints
        for lat, lon in waypoints:
            self.validate_coordinates(lat, lon)
        
        self._waypoints = waypoints
        self._speed_mps = speed_mps
        self._loop = loop
        self._current_index = 0
        self._start_time = time.time()
        print(f"[LOCATION] 🧪 Using MOCK location provider with {len(waypoints)} waypoints")
    
    def get_location(self) -> dict:
        """Get current mock location based on elapsed time and speed."""
        lat, lon = self._waypoints[self._current_index]
        return self._make_result(lat, lon, accuracy=5.0, fix_quality="MOCK")
    
    def advance(self):
        """Advance to next waypoint."""
        self._current_index += 1
        if self._current_index >= len(self._waypoints):
            if self._loop:
                self._current_index = 0
            else:
                self._current_index = len(self._waypoints) - 1
    
    def reset(self):
        """Reset to first waypoint."""
        self._current_index = 0
        self._start_time = time.time()


# ── CLI test mode ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="SURDAS Location Provider Test")
    parser.add_argument("--lat", type=float, required=True, help="Latitude (decimal degrees)")
    parser.add_argument("--lon", type=float, required=True, help="Longitude (decimal degrees)")
    parser.add_argument("--serial", type=str, default=None, help="Serial port for GPS module")
    args = parser.parse_args()

    if args.serial:
        provider = GPSSerialLocation(args.serial)
        provider.start()
    else:
        provider = ManualLocation(args.lat, args.lon)

    loc = provider.get_location()
    print(json.dumps(loc, indent=2))
