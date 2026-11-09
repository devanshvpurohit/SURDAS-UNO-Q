"""
Unit tests for Arduino UNO Q Microcontroller Bridge HAL.
Verifies command dispatch, deterministic emergency stop, and status polling.
"""

from hal.bridge import MockMicrocontrollerBridge


def test_mock_bridge_command_dispatch():
    bridge = MockMicrocontrollerBridge()
    assert bridge.is_connected() is True

    # Send LED command
    ok = bridge.send_command("SET_LED", state=True)
    assert ok is True
    telemetry = bridge.read_telemetry()
    assert telemetry["led"] is True

    # Send Buzzer command
    ok = bridge.send_command("SET_BUZZER", state=True)
    assert ok is True
    telemetry = bridge.read_telemetry()
    assert telemetry["buzzer"] is True


def test_mock_bridge_emergency_stop():
    bridge = MockMicrocontrollerBridge()
    assert bridge.emergency_stopped is False

    # Trigger emergency stop
    ok = bridge.trigger_emergency_stop()
    assert ok is True
    assert bridge.emergency_stopped is True
    
    telemetry = bridge.read_telemetry()
    assert telemetry["e_stop"] is True

    # Reset
    bridge.send_command("RESET_STOP")
    assert bridge.emergency_stopped is False
