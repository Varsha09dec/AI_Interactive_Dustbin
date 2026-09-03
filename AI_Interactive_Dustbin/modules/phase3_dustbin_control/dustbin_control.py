"""
=============================================================================
Project: AI Interactive Rejecting Dustbin
Module: Phase 3 - Dustbin Lid / Servo Control
File: dustbin_control.py
=============================================================================

This module provides the independent, modular hardware-control engine for the
AI Interactive Rejecting Dustbin project.

Hardware Architecture:
Host Computer (Python) -> USB Serial -> Arduino / ESP32 -> Servo Motor -> Dustbin Lid

Key Features:
- Clean, high-level lid control interface (open_lid, close_lid, open_and_wait, reject_action).
- Robust serial communication protocol over USB with ASCII commands and acknowledgments.
- Built-in Mock / Hardware Emulation Mode for testing when physical Arduino/ESP32 is not connected.
- Auto-discovery of connected Arduino / ESP32 USB serial ports.
- Safe shutdown and exception handling to prevent servo jamming.
- Completely decoupled from YOLO, MediaPipe, audio, and main.py.
=============================================================================
"""

import enum
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import serial
    import serial.tools.list_ports as list_ports
except ImportError:
    serial = None
    list_ports = None


class LidState(enum.Enum):
    """Represents the physical/operational position of the dustbin lid."""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    REJECTING = "REJECTING"
    UNKNOWN = "UNKNOWN"


@dataclass
class DustbinHardwareConfig:
    """Hardware configuration for Arduino/ESP32 and servo actuator."""
    port: Optional[str] = None          # e.g., 'COM3', '/dev/ttyUSB0', or None for auto-detect
    baudrate: int = 9600               # Standard baudrate (9600 for Arduino, 115200 for ESP32)
    timeout: float = 2.0               # Serial read timeout in seconds
    open_angle: int = 90               # Servo position when lid is fully open (0-180 deg)
    closed_angle: int = 0              # Servo position when lid is closed (0-180 deg)
    default_open_duration: float = 3.0 # Default hold time before auto-closing lid
    auto_reconnect: bool = True        # Automatically attempt reconnect on serial failure


# =============================================================================
# SERIAL PROTOCOL CONSTANTS
# =============================================================================

CMD_OPEN = "OPEN\n"
CMD_CLOSE = "CLOSE\n"
CMD_REJECT = "REJECT\n"
CMD_STATUS = "STATUS\n"
CMD_PING = "PING\n"

RESP_ACK_OPEN = "ACK:OPEN"
RESP_ACK_CLOSE = "ACK:CLOSE"
RESP_ACK_REJECT = "ACK:REJECT"
RESP_PONG = "PONG"


def find_microcontroller_port() -> Optional[str]:
    """
    Scans system USB serial ports and identifies connected Arduino or ESP32 boards.
    Returns the port name if found, else None.
    """
    if list_ports is None:
        return None

    known_keywords = ["arduino", "ch340", "cp210", "ftdi", "usb serial", "esp32"]
    try:
        ports = list(list_ports.comports())
        for p in ports:
            desc = (p.description or "").lower()
            mfg = (p.manufacturer or "").lower()
            for kw in known_keywords:
                if kw in desc or kw in mfg:
                    return p.device
    except Exception:
        pass

    return None


# =============================================================================
# MOCK SERIAL CONNECTION (For Simulation & CI/CD Testing)
# =============================================================================

class MockSerialConnection:
    """
    Simulates an Arduino/ESP32 microcontroller running the dustbin servo firmware.
    Used for automated verification when physical hardware is not connected.
    """

    def __init__(self, port: str = "MOCK_PORT", baudrate: int = 9600, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.is_open: bool = True

        self.virtual_angle: int = 0
        self.virtual_state: LidState = LidState.CLOSED
        self._rx_buffer: List[str] = []
        self.command_history: List[str] = []
        self.simulate_hardware_fault: bool = False

    def write(self, data: bytes) -> int:
        if not self.is_open:
            raise RuntimeError("Cannot write to closed mock serial port.")

        if self.simulate_hardware_fault:
            return 0

        cmd_str = data.decode("ascii", errors="replace").strip()
        self.command_history.append(cmd_str)

        if cmd_str == "OPEN":
            self.virtual_state = LidState.OPEN
            self.virtual_angle = 90
            self._rx_buffer.append(f"{RESP_ACK_OPEN}\n")
        elif cmd_str == "CLOSE":
            self.virtual_state = LidState.CLOSED
            self.virtual_angle = 0
            self._rx_buffer.append(f"{RESP_ACK_CLOSE}\n")
        elif cmd_str == "REJECT":
            self.virtual_state = LidState.CLOSED
            self.virtual_angle = 0
            self._rx_buffer.append(f"{RESP_ACK_REJECT}\n")
        elif cmd_str == "PING":
            self._rx_buffer.append(f"{RESP_PONG}\n")
        elif cmd_str == "STATUS":
            self._rx_buffer.append(f"STATE:{self.virtual_state.value}\n")
        elif cmd_str.startswith("ANGLE"):
            try:
                deg = int(cmd_str.split()[1])
                self.virtual_angle = max(0, min(180, deg))
                self._rx_buffer.append(f"ACK:ANGLE {self.virtual_angle}\n")
            except (IndexError, ValueError):
                self._rx_buffer.append("ERR:INVALID_ANGLE\n")
        else:
            self._rx_buffer.append("ERR:UNKNOWN_COMMAND\n")

        return len(data)

    def readline(self) -> bytes:
        if not self.is_open or not self._rx_buffer:
            return b""
        line = self._rx_buffer.pop(0)
        return line.encode("ascii")

    def reset_input_buffer(self) -> None:
        self._rx_buffer.clear()

    def close(self) -> None:
        self.is_open = False


# =============================================================================
# DUSTBIN HARDWARE CONTROLLER
# =============================================================================

class DustbinController:
    """
    Main hardware control interface for the dustbin lid servo actuator.
    Provides synchronous and timed lid operations over serial or mock mode.
    """

    def __init__(
        self,
        config: Optional[DustbinHardwareConfig] = None,
        mock_mode: bool = False,
    ):
        self.config = config or DustbinHardwareConfig()
        self.mock_mode = mock_mode
        self._serial = None
        self._current_state = LidState.CLOSED
        self._last_command_time = 0.0

    @property
    def is_connected(self) -> bool:
        if self.mock_mode:
            return self._serial is not None and self._serial.is_open
        return self._serial is not None and getattr(self._serial, "is_open", False)

    @property
    def state(self) -> LidState:
        return self._current_state

    def connect(self) -> bool:
        if self.is_connected:
            return True

        if self.mock_mode:
            self._serial = MockSerialConnection(
                port="VIRTUAL_COM_PORT",
                baudrate=self.config.baudrate,
                timeout=self.config.timeout,
            )
            self._current_state = LidState.CLOSED
            return True

        target_port = self.config.port or find_microcontroller_port()
        if not target_port:
            return False

        if serial is None:
            raise ImportError("pyserial is required for hardware serial communication.")

        try:
            self._serial = serial.Serial(
                port=target_port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout,
            )
            time.sleep(1.5)
            self._serial.reset_input_buffer()

            if self._send_raw_command(CMD_PING, expected_ack=RESP_PONG):
                self._current_state = LidState.CLOSED
                return True
            else:
                self.disconnect()
                return False

        except Exception as e:
            print(f"[ERROR] Failed to connect to serial port '{target_port}': {e}")
            self.disconnect()
            return False

    def disconnect(self) -> None:
        if self.is_connected:
            try:
                self._send_raw_command(CMD_CLOSE, expected_ack=RESP_ACK_CLOSE)
            except Exception:
                pass
            try:
                self._serial.close()
            except Exception:
                pass
        self._serial = None
        self._current_state = LidState.UNKNOWN

    def _send_raw_command(self, command: str, expected_ack: Optional[str] = None) -> bool:
        if not self.is_connected:
            return False

        try:
            self._serial.write(command.encode("ascii"))
            self._last_command_time = time.time()

            if expected_ack is not None:
                resp = self._serial.readline().decode("ascii", errors="replace").strip()
                return expected_ack in resp
            return True

        except Exception as e:
            print(f"[WARN] Serial communication error: {e}")
            return False

    def open_lid(self) -> bool:
        success = self._send_raw_command(CMD_OPEN, expected_ack=RESP_ACK_OPEN)
        if success:
            self._current_state = LidState.OPEN
        return success

    def close_lid(self) -> bool:
        success = self._send_raw_command(CMD_CLOSE, expected_ack=RESP_ACK_CLOSE)
        if success:
            self._current_state = LidState.CLOSED
        return success

    def open_and_wait(self, duration_sec: Optional[float] = None) -> bool:
        hold_time = duration_sec if duration_sec is not None else self.config.default_open_duration
        if not self.open_lid():
            return False

        time.sleep(hold_time)
        return self.close_lid()

    def reject_action(self) -> bool:
        success = self._send_raw_command(CMD_REJECT, expected_ack=RESP_ACK_REJECT)
        if success:
            self._current_state = LidState.CLOSED
        return success

    def set_angle(self, angle_degrees: int) -> bool:
        clamped_angle = max(0, min(180, int(angle_degrees)))
        cmd = f"ANGLE {clamped_angle}\n"
        expected = f"ACK:ANGLE {clamped_angle}"
        return self._send_raw_command(cmd, expected_ack=expected)

    def query_status(self) -> LidState:
        if not self.is_connected:
            return LidState.UNKNOWN

        self._send_raw_command(CMD_STATUS)
        resp = self._serial.readline().decode("ascii", errors="replace").strip()
        if "STATE:OPEN" in resp:
            self._current_state = LidState.OPEN
        elif "STATE:CLOSED" in resp:
            self._current_state = LidState.CLOSED
        elif "STATE:REJECTING" in resp:
            self._current_state = LidState.REJECTING
        return self._current_state

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
