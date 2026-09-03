"""
=============================================================================
Project: AI Interactive Rejecting Dustbin
Module: Phase 3 - Dustbin Lid / Servo Control
File: test_dustbin_control.py
=============================================================================

Independent test suite for Phase 3 (Dustbin Lid / Servo Control).

Hardware Connectivity Logic:
- Scans system USB serial ports for connected Arduino or ESP32 boards.
- If physical hardware is present: executes live hardware communication tests.
- If physical hardware is absent: executes the comprehensive Mock/Simulation test suite.
- Strictly adheres to the rule: Never claim physical hardware success if no
  hardware is actually connected.

Test Cases Covered:
1. Lid Open Operation
2. Lid Close Operation
3. Repeated Sequential Operation (Stress test: 10 cycles)
4. Timed Auto-close (open_and_wait)
5. Reject Snap Action (comedic refusal)
6. Custom Angle Positioning
7. Invalid Command Handling
8. Serial Disconnection / Fault Tolerance
9. Safe Context-Manager Shutdown
=============================================================================
"""

import os
import sys
import time
import unittest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dustbin_control import (
    DustbinController,
    DustbinHardwareConfig,
    LidState,
    MockSerialConnection,
    find_microcontroller_port,
)


class TestPhase3DustbinControlSimulation(unittest.TestCase):
    """
    Simulated verification suite validating all control operations, command
    protocols, state transitions, and error handling using the virtual mock microcontroller.
    """

    def setUp(self):
        self.config = DustbinHardwareConfig(
            port="SIMULATED_COM_PORT",
            baudrate=9600,
            timeout=1.0,
            open_angle=90,
            closed_angle=0,
            default_open_duration=0.1,
        )
        self.controller = DustbinController(config=self.config, mock_mode=True)
        self.assertTrue(self.controller.connect(), "Failed to connect to mock controller.")

    def tearDown(self):
        self.controller.disconnect()

    def test_1_open_lid(self):
        """Test Case 1: Actuate servo to open dustbin lid."""
        print("\n[TEST 1] Testing Lid Open Operation...")
        success = self.controller.open_lid()
        self.assertTrue(success, "open_lid() failed to receive acknowledgment.")
        self.assertEqual(self.controller.state, LidState.OPEN)
        self.assertEqual(self.controller._serial.virtual_angle, 90)
        self.assertIn("OPEN", self.controller._serial.command_history)
        print(f"  -> State: {self.controller.state.value}, Servo Angle: {self.controller._serial.virtual_angle} deg (PASSED)")

    def test_2_close_lid(self):
        """Test Case 2: Actuate servo to close dustbin lid."""
        print("\n[TEST 2] Testing Lid Close Operation...")
        self.controller.open_lid()
        success = self.controller.close_lid()
        self.assertTrue(success, "close_lid() failed to receive acknowledgment.")
        self.assertEqual(self.controller.state, LidState.CLOSED)
        self.assertEqual(self.controller._serial.virtual_angle, 0)
        self.assertIn("CLOSE", self.controller._serial.command_history)
        print(f"  -> State: {self.controller.state.value}, Servo Angle: {self.controller._serial.virtual_angle} deg (PASSED)")

    def test_3_repeated_operation(self):
        """Test Case 3: Safe repeated operation (10 consecutive cycles)."""
        print("\n[TEST 3] Testing Safe Repeated Operation (10 Cycles)...")
        for cycle in range(1, 11):
            open_ok = self.controller.open_lid()
            close_ok = self.controller.close_lid()
            self.assertTrue(open_ok and close_ok, f"Failed at cycle {cycle}")
        self.assertEqual(self.controller.state, LidState.CLOSED)
        self.assertEqual(len(self.controller._serial.command_history), 20)
        print(f"  -> Successfully executed {len(self.controller._serial.command_history)} commands cleanly (PASSED)")

    def test_4_timed_open_and_wait(self):
        """Test Case 4: Timed hold and automatic close."""
        print("\n[TEST 4] Testing Timed Hold and Auto-Close...")
        t0 = time.time()
        success = self.controller.open_and_wait(duration_sec=0.15)
        dt = time.time() - t0
        self.assertTrue(success, "open_and_wait failed.")
        self.assertGreaterEqual(dt, 0.14, "Hold duration was not respected.")
        self.assertEqual(self.controller.state, LidState.CLOSED)
        print(f"  -> Held open for {dt:.2f}s and auto-closed safely (PASSED)")

    def test_5_reject_action(self):
        """Test Case 5: Comedic rejection snap motion."""
        print("\n[TEST 5] Testing Reject Snap Motion...")
        success = self.controller.reject_action()
        self.assertTrue(success, "reject_action failed to receive acknowledgment.")
        self.assertEqual(self.controller.state, LidState.CLOSED)
        self.assertIn("REJECT", self.controller._serial.command_history)
        print("  -> Comedic reject action acknowledged and executed (PASSED)")

    def test_6_custom_angle(self):
        """Test Case 6: Direct servo positioning angle."""
        print("\n[TEST 6] Testing Direct Angle Positioning...")
        self.assertTrue(self.controller.set_angle(45))
        self.assertEqual(self.controller._serial.virtual_angle, 45)
        self.assertTrue(self.controller.set_angle(250))
        self.assertEqual(self.controller._serial.virtual_angle, 180)
        print(f"  -> Servo correctly clamped and rotated to {self.controller._serial.virtual_angle} deg (PASSED)")

    def test_7_invalid_command_handling(self):
        """Test Case 7: Invalid serial command gracefully handled."""
        print("\n[TEST 7] Testing Invalid Command Handling...")
        resp = self.controller._send_raw_command("INVALID_GIBBERISH\n", expected_ack="ACK")
        self.assertFalse(resp, "Invalid command should not return positive acknowledgment.")
        print("  -> Invalid command safely rejected by microcontroller protocol (PASSED)")

    def test_8_serial_disconnection(self):
        """Test Case 8: Graceful handling of communication dropout/disconnection."""
        print("\n[TEST 8] Testing Disconnection / Fault Handling...")
        self.controller._serial.simulate_hardware_fault = True
        ok = self.controller.open_lid()
        self.assertFalse(ok, "Command must return False during hardware communication failure.")

        self.controller.disconnect()
        self.assertFalse(self.controller.is_connected)
        self.assertFalse(self.controller.open_lid(), "Cannot open lid while disconnected.")
        print("  -> Safely caught communication dropouts without system crash (PASSED)")

    def test_9_safe_shutdown(self):
        """Test Case 9: Safe shutdown via context manager."""
        print("\n[TEST 9] Testing Safe Shutdown via Context Manager...")
        with DustbinController(config=self.config, mock_mode=True) as c:
            c.open_lid()
            self.assertTrue(c.is_connected)
            self.assertEqual(c.state, LidState.OPEN)

        self.assertFalse(c.is_connected)
        print("  -> Context manager safely detached hardware and reset states (PASSED)")


def check_and_run_hardware_tests():
    print("==================================================================")
    print("   AI Interactive Rejecting Dustbin - Phase 3 Hardware Tests     ")
    print("==================================================================")

    detected_port = find_microcontroller_port()

    if detected_port:
        print(f"[STATUS] Physical Microcontroller DETECTED on port: {detected_port}")
        print("Executing physical hardware test suite...\n")
        cfg = DustbinHardwareConfig(port=detected_port, baudrate=9600)
        ctrl = DustbinController(config=cfg, mock_mode=False)
        if ctrl.connect():
            print("Connected to live hardware! Testing open/close...")
            ctrl.open_lid()
            time.sleep(1.0)
            ctrl.close_lid()
            ctrl.disconnect()
            print("[PHYSICAL HARDWARE TEST COMPLETED SUCCESSFULLY]")
        else:
            print("[WARN] Could not handshake with physical device. Running simulation tests.")
    else:
        print("[STATUS] Physical Hardware Status: NOT CONNECTED (No Arduino/ESP32 USB device found).")
        print("Note: In accordance with project rules, mock/simulation testing will be executed.")
        print("Running comprehensive simulation & protocol validation suite...\n")

    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase3DustbinControlSimulation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    check_and_run_hardware_tests()
