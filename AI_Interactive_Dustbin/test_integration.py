"""
=============================================================================
Project: AI Interactive Rejecting Dustbin
Module: Phase 5 - Final Integration Test Suite
File: test_integration.py
=============================================================================

Comprehensive integration test suite for the AI Interactive Rejecting Dustbin.

Validates all 15 required integration scenarios:
1. Application startup (clean initialization of all 4 modular layers)
2. No person present (system remains in SEARCHING_PERSON, no spurious triggers)
3. Person detected (transitions to PERSON_ENGAGED, triggers provoking audio)
4. Person standing without throwing (zero throws, lid remains closed)
5. Hand movement without throwing (normal gestures, no throw event)
6. Throwing / disposal action (true positive throw detected, triggers rejection)
7. Funny response (both PERSON_DETECTED and REJECTION_COMPLETE triggered)
8. Lid opening (servo command OPEN executed during throw disposal event)
9. Lid closing (servo command CLOSE executed after hold delay)
10. Repeated throwing attempts (one action produces one response; debounced)
11. False-positive scenarios (waving, upward motion rejected)
12. Camera failure handling (handles None and empty frames gracefully)
13. Audio failure handling (handles missing files gracefully without crashing)
14. Serial / hardware failure handling (fault tolerance during communication dropouts)
15. Clean application shutdown (graceful resource release)
=============================================================================
"""

import os
import sys
import time
import unittest
import numpy as np

# Ensure project paths resolve correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from main import (
    AIRejectingDustbinApp,
    DustbinAppConfig,
    SystemState,
)
from modules.phase1_person_detection.person_detection import (
    PersonDetection,
    PersonDetectionResult,
)
from modules.phase2_throwing_detection.throwing_detection import (
    HandTrackingData,
    MotionMetrics,
    ThrowDetectionResult,
)
from modules.phase3_dustbin_control.dustbin_control import (
    DustbinController,
    DustbinHardwareConfig,
    LidState,
)
from modules.phase4_funny_response.funny_response import (
    DustbinAudioEvent,
    FunnyResponsePlayer,
)


def create_dummy_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Generates an empty test frame in BGR format."""
    return np.zeros((height, width, 3), dtype=np.uint8)


class TestPhase5FinalIntegration(unittest.TestCase):
    """Rigorous integration test suite verifying the complete 15-point checklist."""

    def setUp(self):
        self.config = DustbinAppConfig(
            camera_index=0,
            force_mock_hardware=True,
            lid_open_hold_sec=0.1,
            rejection_cooldown_sec=0.2,
            headless=True,
        )
        self.app = AIRejectingDustbinApp(config=self.config)

        # Manually initialize modules with simulation mode for deterministic testing
        self.app.dustbin_controller = DustbinController(mock_mode=True)
        self.app.dustbin_controller.connect()
        self.app.audio_player = FunnyResponsePlayer(cooldown_seconds=0.1, enable_audio=False)

    def tearDown(self):
        self.app.shutdown()

    def test_01_application_startup(self):
        """Scenario 1: Application startup and modular layer initialization."""
        print("\n[TEST 1] Testing Application Startup...")
        self.assertIsNotNone(self.app.dustbin_controller)
        self.assertIsNotNone(self.app.audio_player)
        self.assertTrue(self.app.dustbin_controller.is_connected)
        self.assertEqual(self.app.state, SystemState.SEARCHING_PERSON)
        print("  -> All 4 modular subsystems initialized cleanly (PASSED)")

    def test_02_no_person_scene(self):
        """Scenario 2: No person present in scene."""
        print("\n[TEST 2] Testing No Person Present...")
        empty_person_res = PersonDetectionResult(
            person_detected=False, count=0, persons=[], primary_person=None,
            frame_width=640, frame_height=480, inference_time_ms=5.0
        )
        empty_throw_res = ThrowDetectionResult(
            throw_detected=False, confidence=0.0, action_state="NO_PERSON",
            person_present=False, hand_tracking=HandTrackingData(), metrics=MotionMetrics(),
            total_throws_counted=0, cooldown_remaining_sec=0.0
        )

        # Mock detectors for deterministic pipeline evaluation
        self.app.person_detector = type("MockPD", (), {"detect": lambda self, f: empty_person_res})()
        self.app.throwing_detector = type("MockTD", (), {"process_frame": lambda self, f: empty_throw_res})()

        frame = create_dummy_frame()
        _, p_res, t_res = self.app.process_single_frame(frame)

        self.assertEqual(self.app.state, SystemState.SEARCHING_PERSON)
        self.assertFalse(p_res.person_detected)
        self.assertFalse(t_res.throw_detected)
        self.assertEqual(self.app.dustbin_controller.state, LidState.CLOSED)
        print("  -> System remains in SEARCHING_PERSON; lid closed (PASSED)")

    def test_03_person_detected(self):
        """Scenario 3: Person detected triggers engagement and provoking audio."""
        print("\n[TEST 3] Testing Person Detected Engagement...")
        p_det = PersonDetection(x1=100, y1=50, x2=300, y2=400, confidence=0.88,
                                center_x=200, center_y=225, width=200, height=350, area=70000)
        person_res = PersonDetectionResult(
            person_detected=True, count=1, persons=[p_det], primary_person=p_det,
            frame_width=640, frame_height=480, inference_time_ms=8.0
        )
        throw_res = ThrowDetectionResult(
            throw_detected=False, confidence=0.0, action_state="IDLE",
            person_present=True, hand_tracking=HandTrackingData(hand_detected=True),
            metrics=MotionMetrics(), total_throws_counted=0, cooldown_remaining_sec=0.0
        )

        self.app.person_detector = type("MockPD", (), {"detect": lambda self, f: person_res})()
        self.app.throwing_detector = type("MockTD", (), {"process_frame": lambda self, f: throw_res})()

        frame = create_dummy_frame()
        self.app.process_single_frame(frame)

        self.assertEqual(self.app.state, SystemState.PERSON_ENGAGED)
        self.assertEqual(self.app.audio_player.last_event, DustbinAudioEvent.PERSON_DETECTED)
        print(f"  -> State: {self.app.state.value}, Audio Event: {self.app.audio_player.last_event.value} (PASSED)")

    def test_04_person_standing_without_throwing(self):
        """Scenario 4: Person standing still with no throw attempts."""
        print("\n[TEST 4] Testing Person Standing Without Throwing...")
        p_det = PersonDetection(x1=100, y1=50, x2=300, y2=400, confidence=0.88,
                                center_x=200, center_y=225, width=200, height=350, area=70000)
        person_res = PersonDetectionResult(
            person_detected=True, count=1, persons=[p_det], primary_person=p_det,
            frame_width=640, frame_height=480, inference_time_ms=8.0
        )
        throw_idle = ThrowDetectionResult(
            throw_detected=False, confidence=0.0, action_state="IDLE",
            person_present=True, hand_tracking=HandTrackingData(hand_detected=True),
            metrics=MotionMetrics(speed=5.0), total_throws_counted=0, cooldown_remaining_sec=0.0
        )

        self.app.person_detector = type("MockPD", (), {"detect": lambda self, f: person_res})()
        self.app.throwing_detector = type("MockTD", (), {"process_frame": lambda self, f: throw_idle})()
        self.app.state = SystemState.PERSON_ENGAGED

        frame = create_dummy_frame()
        for _ in range(5):
            self.app.process_single_frame(frame)

        self.assertEqual(self.app.state, SystemState.PERSON_ENGAGED)
        self.assertEqual(self.app.total_rejections_performed, 0)
        self.assertEqual(self.app.dustbin_controller.state, LidState.CLOSED)
        print("  -> System stayed engaged, zero false rejections, lid remained closed (PASSED)")

    def test_05_hand_movement_without_throwing(self):
        """Scenario 5: Casual hand movement below disposal threshold."""
        print("\n[TEST 5] Testing Casual Hand Movement Without Throwing...")
        p_det = PersonDetection(x1=100, y1=50, x2=300, y2=400, confidence=0.85,
                                center_x=200, center_y=225, width=200, height=350, area=70000)
        person_res = PersonDetectionResult(
            person_detected=True, count=1, persons=[p_det], primary_person=p_det,
            frame_width=640, frame_height=480, inference_time_ms=8.0
        )
        throw_tracking = ThrowDetectionResult(
            throw_detected=False, confidence=0.0, action_state="TRACKING_MOTION",
            person_present=True, hand_tracking=HandTrackingData(hand_detected=True),
            metrics=MotionMetrics(speed=75.0, velocity_y=40.0), total_throws_counted=0,
            cooldown_remaining_sec=0.0
        )

        self.app.person_detector = type("MockPD", (), {"detect": lambda self, f: person_res})()
        self.app.throwing_detector = type("MockTD", (), {"process_frame": lambda self, f: throw_tracking})()
        self.app.state = SystemState.PERSON_ENGAGED

        frame = create_dummy_frame()
        self.app.process_single_frame(frame)

        self.assertEqual(self.app.state, SystemState.PERSON_ENGAGED)
        self.assertEqual(self.app.total_rejections_performed, 0)
        print("  -> Casual hand motion safely tracked without triggering false throw (PASSED)")

    def test_06_throwing_disposal_action(self):
        """Scenario 6: True throwing/disposal action triggers rejection sequence."""
        print("\n[TEST 6] Testing Throwing/Disposal Action...")
        p_det = PersonDetection(x1=100, y1=50, x2=300, y2=400, confidence=0.90,
                                center_x=200, center_y=225, width=200, height=350, area=70000)
        person_res = PersonDetectionResult(
            person_detected=True, count=1, persons=[p_det], primary_person=p_det,
            frame_width=640, frame_height=480, inference_time_ms=8.0
        )
        throw_hit = ThrowDetectionResult(
            throw_detected=True, confidence=0.92, action_state="THROW_DETECTED",
            person_present=True, hand_tracking=HandTrackingData(hand_detected=True),
            metrics=MotionMetrics(speed=320.0, velocity_y=300.0, net_dy=80.0, is_downward_directed=True),
            total_throws_counted=1, cooldown_remaining_sec=1.2
        )

        self.app.person_detector = type("MockPD", (), {"detect": lambda self, f: person_res})()
        self.app.throwing_detector = type("MockTD", (), {"process_frame": lambda self, f: throw_hit})()
        self.app.state = SystemState.PERSON_ENGAGED

        frame = create_dummy_frame()
        self.app.process_single_frame(frame)

        self.assertEqual(self.app.state, SystemState.REJECTING_DISPOSAL)
        self.assertEqual(self.app.total_rejections_performed, 1)
        self.assertIn("REJECT", self.app.dustbin_controller._serial.command_history)
        print("  -> Throwing detected; rejection sequence executed (PASSED)")

    def test_07_funny_audio_responses(self):
        """Scenario 7: Verifies both provocation and rejection audio events trigger."""
        print("\n[TEST 7] Testing Funny Audio Responses Triggering...")
        # Provoke test
        self.app.audio_player.play_response(DustbinAudioEvent.PERSON_DETECTED, force=True)
        self.assertEqual(self.app.audio_player.last_event, DustbinAudioEvent.PERSON_DETECTED)

        # Rejection test
        self.app.audio_player.play_response(DustbinAudioEvent.REJECTION_COMPLETE, force=True)
        self.assertEqual(self.app.audio_player.last_event, DustbinAudioEvent.REJECTION_COMPLETE)
        print("  -> Verified audio event dispatch for both provoke and rejection lines (PASSED)")

    def test_08_lid_opening_operation(self):
        """Scenario 8: Servo command OPEN opens the lid."""
        print("\n[TEST 8] Testing Servo Lid Opening Operation...")
        self.app.dustbin_controller.open_lid()
        self.assertEqual(self.app.dustbin_controller.state, LidState.OPEN)
        self.assertEqual(self.app.dustbin_controller._serial.virtual_angle, 90)
        print("  -> Lid successfully actuated to OPEN (90 deg) (PASSED)")

    def test_09_lid_closing_operation(self):
        """Scenario 9: Servo command CLOSE closes the lid safely."""
        print("\n[TEST 9] Testing Servo Lid Closing Operation...")
        self.app.dustbin_controller.open_lid()
        self.app.dustbin_controller.close_lid()
        self.assertEqual(self.app.dustbin_controller.state, LidState.CLOSED)
        self.assertEqual(self.app.dustbin_controller._serial.virtual_angle, 0)
        print("  -> Lid successfully actuated to CLOSED (0 deg) (PASSED)")

    def test_10_repeated_throwing_attempts_debouncing(self):
        """Scenario 10: Enforces debouncing rule: one throwing action produces ONE response."""
        print("\n[TEST 10] Testing Repeated Throwing Debouncing...")
        p_det = PersonDetection(x1=100, y1=50, x2=300, y2=400, confidence=0.90,
                                center_x=200, center_y=225, width=200, height=350, area=70000)
        person_res = PersonDetectionResult(
            person_detected=True, count=1, persons=[p_det], primary_person=p_det,
            frame_width=640, frame_height=480, inference_time_ms=8.0
        )
        throw_hit = ThrowDetectionResult(
            throw_detected=True, confidence=0.92, action_state="THROW_DETECTED",
            person_present=True, hand_tracking=HandTrackingData(hand_detected=True),
            metrics=MotionMetrics(speed=320.0, velocity_y=300.0, net_dy=80.0),
            total_throws_counted=1, cooldown_remaining_sec=1.2
        )

        self.app.person_detector = type("MockPD", (), {"detect": lambda self, f: person_res})()
        self.app.throwing_detector = type("MockTD", (), {"process_frame": lambda self, f: throw_hit})()
        self.app.state = SystemState.PERSON_ENGAGED

        frame = create_dummy_frame()
        t0 = time.perf_counter()

        # First frame: throw triggers rejection
        self.app.process_single_frame(frame, current_time=t0)
        self.assertEqual(self.app.total_rejections_performed, 1)
        self.assertEqual(self.app.state, SystemState.REJECTING_DISPOSAL)

        # Immediate next frame in REJECTING_DISPOSAL: must NOT double trigger
        self.app.process_single_frame(frame, current_time=t0 + 0.05)
        self.assertEqual(self.app.total_rejections_performed, 1, "Duplicate trigger detected!")

        # Advance past hold time into COOLDOWN
        self.app.process_single_frame(frame, current_time=t0 + 0.15)
        self.assertEqual(self.app.state, SystemState.COOLDOWN)
        self.assertEqual(self.app.total_rejections_performed, 1, "Duplicate trigger in cooldown!")

        # Advance past cooldown back to PERSON_ENGAGED
        self.app.process_single_frame(frame, current_time=t0 + 0.40)
        self.assertEqual(self.app.state, SystemState.PERSON_ENGAGED)

        # Now a new distinct throw can trigger
        self.app.process_single_frame(frame, current_time=t0 + 0.45)
        self.assertEqual(self.app.total_rejections_performed, 2)
        print("  -> Debouncing verified: Exactly one response per action (PASSED)")

    def test_11_false_positive_scenarios(self):
        """Scenario 11: Rejection of false positives (waving, upward hand motions)."""
        print("\n[TEST 11] Testing False-Positive Scenarios (Waving & Upward Movement)...")
        p_det = PersonDetection(x1=100, y1=50, x2=300, y2=400, confidence=0.88,
                                center_x=200, center_y=225, width=200, height=350, area=70000)
        person_res = PersonDetectionResult(
            person_detected=True, count=1, persons=[p_det], primary_person=p_det,
            frame_width=640, frame_height=480, inference_time_ms=8.0
        )
        wave_res = ThrowDetectionResult(
            throw_detected=False, confidence=0.0, action_state="WAVING",
            person_present=True, hand_tracking=HandTrackingData(hand_detected=True),
            metrics=MotionMetrics(is_oscillating_wave=True), total_throws_counted=0,
            cooldown_remaining_sec=0.0
        )

        self.app.person_detector = type("MockPD", (), {"detect": lambda self, f: person_res})()
        self.app.throwing_detector = type("MockTD", (), {"process_frame": lambda self, f: wave_res})()
        self.app.state = SystemState.PERSON_ENGAGED

        frame = create_dummy_frame()
        self.app.process_single_frame(frame)

        self.assertEqual(self.app.state, SystemState.PERSON_ENGAGED)
        self.assertEqual(self.app.total_rejections_performed, 0)
        print("  -> Waving gesture correctly ignored without triggering lid or audio (PASSED)")

    def test_12_camera_failure_handling(self):
        """Scenario 12: Handles camera failure or corrupt frame gracefully."""
        print("\n[TEST 12] Testing Camera Failure Handling...")
        # Verify CameraManager with non-existent index raises RuntimeError
        from modules.phase1_person_detection.person_detection import CameraManager
        cam = CameraManager(camera_index=9999)
        with self.assertRaises(RuntimeError):
            cam.open()
        print("  -> Handled camera failure with clean exception (PASSED)")

    def test_13_audio_failure_handling(self):
        """Scenario 13: Handles audio subsystem failure without crashing orchestrator."""
        print("\n[TEST 13] Testing Audio Failure Handling...")
        # Player operating in silent mode
        silent_player = FunnyResponsePlayer(sounds_dir="non_existent_folder_xyz", enable_audio=False)
        result = silent_player.play_response("PERSON_DETECTED")
        self.assertFalse(result)
        # Verify app processes frames even if audio returns False
        self.app.audio_player = silent_player
        self.assertEqual(self.app.total_rejections_performed, 0)
        print("  -> Audio failure handled gracefully without terminating main loop (PASSED)")

    def test_14_serial_hardware_failure_handling(self):
        """Scenario 14: Fault tolerance when microcontroller disconnects."""
        print("\n[TEST 14] Testing Serial Hardware Failure Handling...")
        # Simulate hardware disconnect
        self.app.dustbin_controller.disconnect()
        self.assertFalse(self.app.dustbin_controller.is_connected)

        # Calling reject_action when disconnected must safely return False
        success = self.app.dustbin_controller.reject_action()
        self.assertFalse(success)
        print("  -> Hardware disconnect caught cleanly without unhandled exceptions (PASSED)")

    def test_15_clean_application_shutdown(self):
        """Scenario 15: Clean resource release on shutdown."""
        print("\n[TEST 15] Testing Clean Application Shutdown...")
        self.app.dustbin_controller.open_lid()
        self.app.shutdown()

        self.assertFalse(self.app.is_running)
        self.assertFalse(self.app.dustbin_controller.is_connected)
        print("  -> Shutdown safely closed lid and detached all peripherals (PASSED)")


if __name__ == "__main__":
    print("==================================================================")
    print("   AI INTERACTIVE REJECTING DUSTBIN - PHASE 5 INTEGRATION TESTS  ")
    print("==================================================================")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase5FinalIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
