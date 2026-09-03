"""
=============================================================================
Project: AI Interactive Rejecting Dustbin
Module: Phase 1 - Person Detection
File: test_person_detection.py
=============================================================================

Comprehensive test suite for Phase 1 (Person Detection).

Validates all 7 required test scenarios:
1. Empty scene (no person present)
2. Single person present
3. Multiple people present
4. Person entering the camera view (state transition: False -> True)
5. Person leaving the camera view (state transition: True -> False)
6. Camera failure handling (invalid camera index, disconnected feed)
7. Invalid or missing model path handling

Also supports an interactive live webcam mode:
    python test_person_detection.py --live
=============================================================================
"""

import os
import sys
import argparse
import unittest
import time
import cv2
import numpy as np
import ultralytics

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from person_detection import (
    PersonDetector,
    CameraManager,
    PersonDetectionResult,
    get_default_model_path,
)


def get_sample_person_image() -> np.ndarray:
    """Retrieves a standard reference image containing people from ultralytics assets."""
    asset_dir = os.path.join(os.path.dirname(ultralytics.__file__), "assets")
    zidane_path = os.path.join(asset_dir, "zidane.jpg")
    if os.path.exists(zidane_path):
        img = cv2.imread(zidane_path)
        if img is not None:
            return img
    bus_path = os.path.join(asset_dir, "bus.jpg")
    if os.path.exists(bus_path):
        img = cv2.imread(bus_path)
        if img is not None:
            return img
    raise FileNotFoundError("Standard reference sample image not found.")


def generate_empty_scene(width: int = 640, height: int = 480) -> np.ndarray:
    """Generates an empty room-like background scene with no humans."""
    scene = np.full((height, width, 3), (180, 190, 200), dtype=np.uint8)
    cv2.line(scene, (0, int(height * 0.7)), (width, int(height * 0.7)), (120, 120, 130), 3)
    cv2.rectangle(scene, (50, 50), (200, 250), (160, 170, 175), -1)
    return scene


class TestPhase1PersonDetection(unittest.TestCase):
    """Test suite covering the 7 mandatory Phase 1 test cases."""

    @classmethod
    def setUpClass(cls):
        model_path = get_default_model_path()
        cls.detector = PersonDetector(model_path=model_path, conf_threshold=0.45)
        cls.multi_person_img = get_sample_person_image()

        h, w = cls.multi_person_img.shape[:2]
        cls.single_person_img = cls.multi_person_img[:, : int(w * 0.55)].copy()

        cls.empty_scene = generate_empty_scene(width=640, height=480)

    def test_1_empty_scene(self):
        """Test Case 1: Detects absence of people in an empty scene."""
        print("\n[TEST 1] Testing Empty Scene...")
        result = self.detector.detect(self.empty_scene)

        self.assertFalse(result.person_detected, "Person was falsely detected in an empty scene.")
        self.assertEqual(result.count, 0, "Person count must be 0 in empty scene.")
        self.assertIsNone(result.primary_person, "primary_person must be None in empty scene.")
        self.assertEqual(len(result.persons), 0)
        print(f"  -> Result: person_detected={result.person_detected}, count={result.count} (PASSED)")

    def test_2_one_person(self):
        """Test Case 2: Detects a single person accurately."""
        print("\n[TEST 2] Testing Single Person Detection...")
        result = self.detector.detect(self.single_person_img)

        self.assertTrue(result.person_detected, "Failed to detect person in single-person frame.")
        self.assertGreaterEqual(result.count, 1, "Expected at least 1 person detected.")
        self.assertIsNotNone(result.primary_person, "primary_person should not be None.")
        self.assertGreater(result.primary_person.confidence, 0.45, "Confidence should exceed threshold.")
        self.assertGreater(result.primary_person.area, 0, "Bounding box area must be positive.")

        annotated = self.detector.annotate(self.single_person_img, result)
        self.assertEqual(annotated.shape, self.single_person_img.shape)
        print(
            f"  -> Result: person_detected={result.person_detected}, "
            f"count={result.count}, primary_conf={result.primary_person.confidence * 100:.1f}% (PASSED)"
        )

    def test_3_multiple_people(self):
        """Test Case 3: Detects multiple people and identifies primary individual."""
        print("\n[TEST 3] Testing Multiple People Detection...")
        result = self.detector.detect(self.multi_person_img)

        self.assertTrue(result.person_detected, "Failed to detect people in multi-person frame.")
        self.assertGreaterEqual(result.count, 2, f"Expected >= 2 people, detected {result.count}.")
        self.assertIsNotNone(result.primary_person, "primary_person must be identified.")
        for person in result.persons:
            self.assertGreaterEqual(result.primary_person.area, person.area)
        print(
            f"  -> Result: person_detected={result.person_detected}, "
            f"count={result.count}, primary_area={result.primary_person.area} px (PASSED)"
        )

    def test_4_person_entering(self):
        """Test Case 4: Simulates a person entering camera view (state: False -> True)."""
        print("\n[TEST 4] Testing Person Entering Scene...")
        h, w = 480, 640
        frame_0 = generate_empty_scene(w, h)
        res_0 = self.detector.detect(frame_0)
        self.assertFalse(res_0.person_detected, "Frame 0 should have no person.")

        person_crop = cv2.resize(self.single_person_img, (200, 360))
        frame_1 = frame_0.copy()
        frame_1[60:420, 220:420] = person_crop

        res_1 = self.detector.detect(frame_1)
        self.assertTrue(res_1.person_detected, "Person should be detected after entering.")
        print(
            f"  -> Transition: Frame 0 (detected={res_0.person_detected}) "
            f"-> Frame 1 (detected={res_1.person_detected}) (PASSED)"
        )

    def test_5_person_leaving(self):
        """Test Case 5: Simulates a person leaving camera view (state: True -> False)."""
        print("\n[TEST 5] Testing Person Leaving Scene...")
        h, w = 480, 640

        person_crop = cv2.resize(self.single_person_img, (200, 360))
        frame_0 = generate_empty_scene(w, h)
        frame_0[60:420, 220:420] = person_crop
        res_0 = self.detector.detect(frame_0)
        self.assertTrue(res_0.person_detected, "Person should be detected initially.")

        frame_1 = generate_empty_scene(w, h)
        res_1 = self.detector.detect(frame_1)
        self.assertFalse(res_1.person_detected, "Person should no longer be detected after leaving.")
        print(
            f"  -> Transition: Frame 0 (detected={res_0.person_detected}) "
            f"-> Frame 1 (detected={res_1.person_detected}) (PASSED)"
        )

    def test_6_camera_failure(self):
        """Test Case 6: Handles camera failure and invalid frames gracefully."""
        print("\n[TEST 6] Testing Camera Failure Handling...")
        cam = CameraManager(camera_index=9999)
        with self.assertRaises(RuntimeError) as ctx:
            cam.open()
        self.assertIn("Camera failure", str(ctx.exception))
        print("  -> Subtest 6a: Handled invalid camera index 9999 with RuntimeError (PASSED)")

        with self.assertRaises(ValueError):
            self.detector.detect(None)

        empty_frame = np.array([], dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.detector.detect(empty_frame)
        print("  -> Subtest 6b: Handled None and empty frame inputs with ValueError (PASSED)")

    def test_7_invalid_missing_model(self):
        """Test Case 7: Handles invalid or missing YOLO model path."""
        print("\n[TEST 7] Testing Invalid/Missing Model Handling...")
        non_existent_path = "non_existent_weights_path_987654321.pt"
        with self.assertRaises(FileNotFoundError) as ctx:
            PersonDetector(model_path=non_existent_path)
        self.assertIn("YOLO model weights not found", str(ctx.exception))
        print("  -> Handled missing model file with FileNotFoundError (PASSED)")


def run_live_webcam(duration_seconds: int = 0):
    """Executes real-time person detection on the laptop's built-in webcam."""
    print("==================================================================")
    print("   AI Interactive Rejecting Dustbin - Phase 1 Live Webcam Test   ")
    print("==================================================================")
    print("Initializing camera feed and YOLO person detector...")

    detector = PersonDetector(conf_threshold=0.5)
    cam = CameraManager(camera_index=0, width=640, height=480)

    try:
        cam.open()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    window_name = "AI Rejecting Dustbin - Phase 1: Person Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    start_time = time.time()
    try:
        while True:
            ret, frame = cam.read_frame()
            if not ret or frame is None:
                break

            frame = cv2.flip(frame, 1)
            result = detector.detect(frame)
            annotated = detector.annotate(frame, result, show_banner=True)

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break

            if duration_seconds > 0 and (time.time() - start_time) >= duration_seconds:
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Person Detection Tests")
    parser.add_argument("--live", action="store_true", help="Launch interactive live webcam test")
    parser.add_argument("--duration", type=int, default=0, help="Auto-close live webcam after N seconds")
    args = parser.parse_args()

    if args.live:
        run_live_webcam(duration_seconds=args.duration)
    else:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase1PersonDetection)
        runner = unittest.TextTestRunner(verbosity=2)
        test_result = runner.run(suite)
        if not test_result.wasSuccessful():
            sys.exit(1)
