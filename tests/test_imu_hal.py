"""
Unit tests for Sensor and IMU Hardware Abstraction Layer.
Validates Simulated vs Real telemetry tagging, fall event states,
missing IMU handling, and data integrity.
"""

import time
from hal.sensor import (
    IMUReading, SimulatedIMUProvider, BridgeIMUProvider, create_imu_provider
)


def test_simulated_imu_provider_output():
    sim = SimulatedIMUProvider()
    assert sim.is_connected() is False
    assert sim.is_simulated() is True

    reading = sim.get_reading()
    assert isinstance(reading, IMUReading)
    assert reading.source == "SIMULATION"
    assert reading.state == "NORMAL"
    assert reading.falls == 0
    assert 0.8 <= reading.acceleration <= 1.2


def test_simulated_fall_sequence():
    sim = SimulatedIMUProvider()
    sim.trigger_simulated_fall()
    assert sim.fall_count == 1

    # Immediately after trigger: freefall phase
    reading = sim.get_reading()
    assert reading.state in ("FREEFALL", "IMPACT")
    assert reading.source == "SIMULATION"
    assert reading.falls == 1

    sim.reset_falls()
    assert sim.fall_count == 0


def test_bridge_imu_hardware_tagging():
    bridge_imu = BridgeIMUProvider()
    assert bridge_imu.is_connected() is False
    assert bridge_imu.is_simulated() is False

    # Simulate packet arrival from STM32 MCU
    packet = {
        "ax": 0.05,
        "ay": -0.02,
        "az": 0.99,
        "acceleration": 0.99,
        "pitch": 2.5,
        "roll": -1.2,
        "falls": 0,
        "state": "NORMAL"
    }
    bridge_imu.update_from_bridge_data(packet)

    assert bridge_imu.is_connected() is True
    reading = bridge_imu.get_reading()
    assert reading is not None
    assert reading.source == "HARDWARE"
    assert reading.state == "NORMAL"


def test_create_imu_provider_factory():
    sim_provider = create_imu_provider(enable_simulation=True)
    assert sim_provider.is_simulated() is True

    real_provider = create_imu_provider(enable_simulation=False)
    assert real_provider.is_simulated() is False
