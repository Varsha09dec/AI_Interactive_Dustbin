"""
=============================================================================
Project: AI Interactive Rejecting Dustbin
Module: Phase 2 - Waste / Throwing Action Detection
File: test_throwing_detection.py
=============================================================================

Comprehensive test suite for Phase 2 (Throwing / Waste Action Detection).

Validates all 10 required test scenarios:
1. No person present
2. Person standing still (static posture)
3. Normal casual hand movement
4. Waving (horizontal oscillatory gesture rejection)
5. Moving hand without throwing (upward gesture / head scratch)
6. Throwing / disposal action (true positive detection)
7. Multiple throwing attempts (cooldown recovery and counter increment)
8. Fast non-throwing movement (erratic jitter rejection)
9. Person entering and leaving camera view (boundary transitions)
10. Camera failure handling (None and empty frame validation)

Also provides an interactive live webcam mode:
    python test_throwing_detection.py --live
=============================================================================
"""

import argparse
import os
import sys
import time
import unittest
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
phase1_dir = os.path.abspath(os.path.join(current_dir, "..", "phase1_person_detection"))

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if phase1_dir not in sys.path:
    sys.path.insert(0, phase1_dir)

from throwing_detection import (
    HandTrackingData,
    MotionMetrics,
    MotionTracker,
    ThrowDecisionEngine,
    ThrowDetectionResult,
    ThrowingDetector,
)


class LocalCameraManager:
    """Robust local camera interface fallback for standalone testing."""
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera at index {self.camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


try:
    from person_detection import CameraManager
except Exception:
    CameraManager = LocalCameraManager

if CameraManager is None:
    CameraManager = LocalCameraManager


class TestPhase2ThrowingDetection(unittest.TestCase):
    """Rigorous independent test suite covering all 10 Phase 2 test cases."""

    def setUp(self):
        self.tracker = MotionTracker(window_size=12)
        self.decision_engine = ThrowDecisionEngine(
            min_throw_speed=240.0,
            min_downward_dy=50.0,
            cooldown_duration=1.0,
        )

    def test_1_no_person(self):
        """Test Case 1: No person present -> strictly no throw detected."""
        print("\n[TEST 1] Testing No Person Present...")
        metrics = MotionMetrics(velocity_y=400.0, net_dy=100.0, is_downward_directed=True)
        throw_det, conf, state, _ = self.decision_engine.evaluate(
            person_present=False,
            hand_detected=True,
            metrics=metrics,
            current_time=1.0,
        )
        self.assertFalse(throw_det, "Throw must NOT be detected when no person is present.")
        self.assertEqual(state, "NO_PERSON")
        print(f"  -> State: {state}, Throw: {throw_det} (PASSED)")

    def test_2_person_standing_still(self):
        """Test Case 2: Person standing still with stationary hand."""
        print("\n[TEST 2] Testing Person Standing Still...")
        for i in range(10):
            self.tracker.add_point(x=320, y=240, timestamp=i * 0.033)

        metrics = self.tracker.compute_metrics()
        throw_det, conf, state, _ = self.decision_engine.evaluate(
            person_present=True,
            hand_detected=True,
            metrics=metrics,
            current_time=0.33,
        )
        self.assertFalse(throw_det, "Stationary posture must not trigger a throw.")
        self.assertEqual(state, "IDLE")
        self.assertLess(metrics.speed, 20.0)
        print(f"  -> Speed: {metrics.speed:.1f} px/s, State: {state} (PASSED)")

    def test_3_normal_hand_movement(self):
        """Test Case 3: Casual hand movement below throwing speed threshold."""
        print("\n[TEST 3] Testing Normal Hand Movement...")
        for i in range(10):
            self.tracker.add_point(x=300 + i * 2, y=200 + i * 3, timestamp=i * 0.05)

        metrics = self.tracker.compute_metrics()
        throw_det, conf, state, _ = self.decision_engine.evaluate(
            person_present=True,
            hand_detected=True,
            metrics=metrics,
            current_time=0.5,
        )
        self.assertFalse(throw_det, "Slow casual hand movement must not trigger a throw.")
        self.assertIn(state, ["IDLE", "TRACKING_MOTION"])
        self.assertLess(metrics.velocity_y, 240.0)
        print(f"  -> Vy: {metrics.velocity_y:.1f} px/s (Threshold 240), State: {state} (PASSED)")

    def test_4_waving(self):
        """Test Case 4: Waving rejection (horizontal left-right oscillations)."""
        print("\n[TEST 4] Testing Waving Rejection...")
        coords = [200, 240, 280, 230, 190, 240, 290, 250, 200]
        for i, x in enumerate(coords):
            self.tracker.add_point(x=x, y=220, timestamp=i * 0.04)

        metrics = self.tracker.compute_metrics()
        self.assertTrue(metrics.is_oscillating_wave, "Failed to classify horizontal oscillation as waving.")

        throw_det, conf, state, _ = self.decision_engine.evaluate(
            person_present=True,
            hand_detected=True,
            metrics=metrics,
            current_time=0.4,
        )
        self.assertFalse(throw_det, "Waving must be explicitly rejected.")
        self.assertEqual(state, "WAVING")
        print(f"  -> is_oscillating_wave: {metrics.is_oscillating_wave}, State: {state} (PASSED)")

    def test_5_moving_hand_without_throwing(self):
        """Test Case 5: Moving hand upward (scratching head, lifting coffee)."""
        print("\n[TEST 5] Testing Upward Hand Movement Without Throwing...")
        for i in range(8):
            self.tracker.add_point(x=300, y=350 - i * 25, timestamp=i * 0.04)

        metrics = self.tracker.compute_metrics()
        self.assertLess(metrics.velocity_y, 0, "Upward motion must produce negative velocity_y.")

        throw_det, conf, state, _ = self.decision_engine.evaluate(
            person_present=True,
            hand_detected=True,
            metrics=metrics,
            current_time=0.32,
        )
        self.assertFalse(throw_det, "Upward movement must not trigger a throw.")
        self.assertEqual(state, "MOVING_UPWARD")
        print(f"  -> Vy: {metrics.velocity_y:.1f} px/s, State: {state} (PASSED)")

    def test_6_throwing_disposal_action(self):
        """Test Case 6: True throwing / disposal action toward dustbin."""
        print("\n[TEST 6] Testing Throwing/Disposal Action...")
        y_seq = [150, 160, 180, 220, 270, 330, 390, 420]
        for i, y in enumerate(y_seq):
            self.tracker.add_point(x=320 + i * 5, y=y, timestamp=i * 0.04)

        metrics = self.tracker.compute_metrics()
        self.assertTrue(metrics.is_downward_directed, "Kinematics should register as downward directed.")

        throw_det, conf, state, cooldown = self.decision_engine.evaluate(
            person_present=True,
            hand_detected=True,
            metrics=metrics,
            object_detected=True,
            current_time=0.35,
        )
        self.assertTrue(throw_det, "Deliberate downward throw must be detected.")
        self.assertEqual(state, "THROW_DETECTED")
        self.assertGreaterEqual(conf, 0.60, "Confidence should exceed 0.60.")
        self.assertEqual(self.decision_engine.total_throws_counted, 1)
        print(f"  -> Throw: {throw_det}, Conf: {conf*100:.0f}%, State: {state} (PASSED)")

    def test_7_multiple_throwing_attempts(self):
        """Test Case 7: Multiple distinct throws with cooldown verification."""
        print("\n[TEST 7] Testing Multiple Throwing Attempts...")
        for i, y in enumerate([150, 190, 250, 330, 400]):
            self.tracker.add_point(x=320, y=y, timestamp=i * 0.04)

        m1 = self.tracker.compute_metrics()
        t1, _, s1, c1 = self.decision_engine.evaluate(
            person_present=True, hand_detected=True, metrics=m1, current_time=0.20
        )
        self.assertTrue(t1, "Throw 1 must be detected.")
        self.assertEqual(self.decision_engine.total_throws_counted, 1)

        t_blocked, _, s_blocked, _ = self.decision_engine.evaluate(
            person_present=True, hand_detected=True, metrics=m1, current_time=0.40
        )
        self.assertFalse(t_blocked, "Duplicate trigger must be suppressed during cooldown.")
        self.assertEqual(s_blocked, "COOLDOWN")

        self.tracker.reset()
        for i, y in enumerate([140, 180, 240, 320, 390]):
            self.tracker.add_point(x=320, y=y, timestamp=1.5 + i * 0.04)

        m2 = self.tracker.compute_metrics()
        t2, _, s2, _ = self.decision_engine.evaluate(
            person_present=True, hand_detected=True, metrics=m2, current_time=1.70
        )
        self.assertTrue(t2, "Throw 2 must be detected after cooldown expires.")
        self.assertEqual(self.decision_engine.total_throws_counted, 2)
        print(f"  -> Throws Counted: {self.decision_engine.total_throws_counted} (PASSED)")

    def test_8_fast_erratic_movement(self):
        """Test Case 8: High-speed erratic non-throwing motion (low straightness efficiency)."""
        print("\n[TEST 8] Testing Fast Erratic Movement...")
        flail = [(200, 200), (350, 220), (180, 210), (340, 190), (190, 220), (320, 200)]
        for i, (x, y) in enumerate(flail):
            self.tracker.add_point(x=x, y=y, timestamp=i * 0.02)

        metrics = self.tracker.compute_metrics()
        throw_det, conf, state, _ = self.decision_engine.evaluate(
            person_present=True,
            hand_detected=True,
            metrics=metrics,
            current_time=0.15,
        )
        self.assertFalse(throw_det, "Erratic motion must not trigger a throw.")
        self.assertIn(state, ["ERRATIC_MOTION", "WAVING", "TRACKING_MOTION"])
        print(f"  -> State: {state}, Throw: {throw_det} (PASSED)")

    def test_9_person_entering_and_leaving(self):
        """Test Case 9: Boundary state transitions when person enters and leaves."""
        print("\n[TEST 9] Testing Person Entering and Leaving...")
        t_a, _, s_a, _ = self.decision_engine.evaluate(
            person_present=False, hand_detected=False, metrics=MotionMetrics(), current_time=1.0
        )
        self.assertEqual(s_a, "NO_PERSON")

        self.tracker.add_point(300, 200, timestamp=1.05)
        self.tracker.add_point(300, 205, timestamp=1.10)
        t_b, _, s_b, _ = self.decision_engine.evaluate(
            person_present=True, hand_detected=True, metrics=self.tracker.compute_metrics(), current_time=1.10
        )
        self.assertFalse(t_b)
        self.assertIn(s_b, ["IDLE", "TRACKING_MOTION"])

        self.tracker.reset()
        t_c, _, s_c, _ = self.decision_engine.evaluate(
            person_present=False, hand_detected=False, metrics=MotionMetrics(), current_time=1.20
        )
        self.assertEqual(s_c, "NO_PERSON")
        self.assertFalse(t_c)
        print(f"  -> Transitions: {s_a} -> {s_b} -> {s_c} (PASSED)")

    def test_10_camera_failure(self):
        """Test Case 10: Robustness against invalid/empty frames."""
        print("\n[TEST 10] Testing Camera Failure Handling...")
        detector = ThrowingDetector(check_objects=False)

        with self.assertRaises(ValueError):
            detector.process_frame(None)

        empty_frame = np.array([], dtype=np.uint8)
        with self.assertRaises(ValueError):
            detector.process_frame(empty_frame)

        print("  -> Correctly raised ValueError on None and empty frames (PASSED)")


def run_live_webcam(duration_seconds: int = 0):
    """Launches live interactive webcam session for Phase 2."""
    print("==================================================================")
    print("  AI Interactive Rejecting Dustbin - Phase 2 Live Throwing Test   ")
    print("==================================================================")
    print("Initializing ThrowingDetector and CameraManager...")

    detector = ThrowingDetector(check_objects=True)
    cam = CameraManager(camera_index=0, width=640, height=480)

    try:
        cam.open()
    except Exception as e:
        print(f"[ERROR] Could not open webcam: {e}")
        return

    window_name = "AI Rejecting Dustbin - Phase 2: Throwing Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    print("\nLive camera running. Controls:")
    print(" - Press 'q' or 'ESC' to exit")
    print(" - Stand in front of camera and perform a downward disposal action toward bin")
    print(" - Try waving or lifting hand to observe false-positive rejection\n")

    t_start = time.time()
    try:
        while True:
            ret, frame = cam.read_frame()
            if not ret or frame is None:
                break

            frame = cv2.flip(frame, 1)
            result = detector.process_frame(frame)
            annotated = detector.annotate(frame, result, show_hud=True)

            if result.throw_detected:
                print(f"[ALERT] ★ THROW DETECTED! Total: {result.total_throws_counted} (Conf: {result.confidence * 100:.0f}%)")

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break

            if duration_seconds > 0 and (time.time() - t_start) >= duration_seconds:
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Throwing Detection Tests")
    parser.add_argument("--live", action="store_true", help="Run live webcam test with GUI")
    parser.add_argument("--duration", type=int, default=0, help="Live webcam test duration (seconds)")
    args = parser.parse_args()

    if args.live:
        run_live_webcam(duration_seconds=args.duration)
    else:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase2ThrowingDetection)
        runner = unittest.TextTestRunner(verbosity=2)
        test_result = runner.run(suite)
        if not test_result.wasSuccessful():
            sys.exit(1)
