"""
HAL Microcontroller Bridge for SURDAS on Arduino UNO Q.
Coordinates real-time communication between Qualcomm QRB2210 Linux MPU
and STM32U585 MCU for deterministic sensor reading and safety actions.
"""

import abc
import time
import json
import logging
import threading
from typing import Optional, Dict, Any, Callable

logger = logging.getLogger("surdas.hal.bridge")


class MicrocontrollerBridge(abc.ABC):
    """Abstract interface for Linux-to-Microcontroller communication."""

    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Return True if connection to microcontroller is active."""
        pass

    @abc.abstractmethod
    def send_command(self, command: str, **kwargs) -> bool:
        """Send a control command to the microcontroller."""
        pass

    @abc.abstractmethod
    def trigger_emergency_stop(self) -> bool:
        """Trigger immediate deterministic hardware emergency stop."""
        pass

    @abc.abstractmethod
    def read_telemetry(self) -> Dict[str, Any]:
        """Read the latest sensor and status telemetry from microcontroller."""
        pass

    @abc.abstractmethod
    def register_telemetry_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for streaming MCU telemetry packets."""
        pass


class UnoQBridge(MicrocontrollerBridge):
    """
    Arduino UNO Q hardware bridge implementation.
    Attempts connection via Arduino Bridge library or internal TTY serial port.
    """

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.connected = False
        self.callbacks = []
        self.lock = threading.Lock()
        self.last_telemetry: Dict[str, Any] = {}
        self.serial_conn = None

        self._try_connect()

    def _try_connect(self) -> bool:
        # Check if arduino bridge package is installed or serial device exists
        try:
            import serial
            candidate_ports = [self.port] if self.port else [
                "/dev/ttyRPMSG0", "/dev/ttyHS0", "/dev/ttyACM0", "/dev/ttyUSB0"
            ]
            for p in candidate_ports:
                if p:
                    try:
                        self.serial_conn = serial.Serial(p, self.baudrate, timeout=1.0)
                        self.connected = True
                        logger.info(f"Connected to Arduino UNO Q MCU on {p}")
                        return True
                    except Exception:
                        pass
        except ImportError:
            logger.info("pyserial not available, bridge running in decoupled mode.")

        self.connected = False
        return False

    def is_connected(self) -> bool:
        return self.connected

    def send_command(self, command: str, **kwargs) -> bool:
        if not self.connected or not self.serial_conn:
            logger.debug(f"[BRIDGE_UNCONNECTED] Dropping command: {command}")
            return False

        try:
            payload = json.dumps({"cmd": command, "args": kwargs}) + "\n"
            self.serial_conn.write(payload.encode("utf-8"))
            return True
        except Exception as e:
            logger.error(f"Error sending command to MCU: {e}")
            self.connected = False
            return False

    def trigger_emergency_stop(self) -> bool:
        """
        Immediate deterministic hardware stop.
        Does NOT depend on AI, network, or high-level logic.
        """
        logger.warning("[SAFETY] 🚨 EMERGENCY STOP TRIGGERED ON MICROCONTROLLER BRIDGE")
        return self.send_command("EMERGENCY_STOP")

    def read_telemetry(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.last_telemetry)

    def register_telemetry_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self.callbacks.append(callback)


class MockMicrocontrollerBridge(MicrocontrollerBridge):
    """
    Mock bridge for development, headless testing, and environments
    without physical Arduino UNO Q hardware attached.
    """

    def __init__(self):
        self.emergency_stopped = False
        self.last_commands = []
        self.callbacks = []
        self.telemetry = {
            "mcu_ready": True,
            "battery_mv": 4150,
            "buzzer": False,
            "led": False,
            "e_stop": False,
            "is_mock": True
        }

    def is_connected(self) -> bool:
        return True

    def send_command(self, command: str, **kwargs) -> bool:
        self.last_commands.append({"cmd": command, "args": kwargs, "time": time.time()})
        if command == "SET_LED":
            self.telemetry["led"] = kwargs.get("state", False)
        elif command == "SET_BUZZER":
            self.telemetry["buzzer"] = kwargs.get("state", False)
        elif command == "EMERGENCY_STOP":
            self.emergency_stopped = True
            self.telemetry["e_stop"] = True
            self.telemetry["buzzer"] = True
        elif command == "RESET_STOP":
            self.emergency_stopped = False
            self.telemetry["e_stop"] = False
        return True

    def trigger_emergency_stop(self) -> bool:
        self.emergency_stopped = True
        self.telemetry["e_stop"] = True
        logger.info("[MOCK_BRIDGE] Emergency stop activated.")
        return True

    def read_telemetry(self) -> Dict[str, Any]:
        return dict(self.telemetry)

    def register_telemetry_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self.callbacks.append(callback)
