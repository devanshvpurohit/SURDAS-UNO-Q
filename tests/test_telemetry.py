"""
Unit tests for Unified Telemetry and Caregiver API Service.
Tests endpoints, standardized IMU payloads, simulated vs real tagging,
and absence of ESP32 hardcoded dependencies.
"""

from fastapi.testclient import TestClient
import telemetry
from hal.sensor import SimulatedIMUProvider, BridgeIMUProvider
from hal.bridge import MockMicrocontrollerBridge


def test_telemetry_health():
    client = TestClient(telemetry.app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["platform"] == "Arduino UNO Q"


def test_telemetry_imu_simulated():
    client = TestClient(telemetry.app)
    sim_imu = SimulatedIMUProvider()
    telemetry.set_imu_provider(sim_imu)

    response = client.get("/imu")
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "SIMULATION"
    assert data["state"] == "NORMAL"
    assert "acceleration" in data
    assert "pitch" in data
    assert "roll" in data
    assert "falls" in data


def test_telemetry_imu_hardware():
    client = TestClient(telemetry.app)
    hw_imu = BridgeIMUProvider()
    hw_imu.update_from_bridge_data({
        "ax": 0.01, "ay": 0.02, "az": 0.99,
        "acceleration": 0.99, "pitch": 0.5, "roll": 0.2,
        "falls": 0, "state": "NORMAL"
    })
    telemetry.set_imu_provider(hw_imu)

    response = client.get("/imu")
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "HARDWARE"
    assert data["state"] == "NORMAL"


def test_telemetry_emergency_stop():
    client = TestClient(telemetry.app)
    bridge = MockMicrocontrollerBridge()
    telemetry.set_bridge(bridge)

    response = client.post("/emergency_stop")
    assert response.status_code == 200
    data = response.json()
    assert data["emergency_stop"] is True
    assert bridge.emergency_stopped is True


def test_telemetry_test_fall():
    client = TestClient(telemetry.app)
    sim_imu = SimulatedIMUProvider()
    telemetry.set_imu_provider(sim_imu)

    response = client.post("/test_fall")
    assert response.status_code == 200
    assert sim_imu.fall_count == 1


def test_telemetry_serves_static_dashboard():
    client = TestClient(telemetry.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "html" in response.text.lower()
