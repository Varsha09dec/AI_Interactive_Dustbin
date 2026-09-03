"""
=============================================================================
Project: AI Interactive Rejecting Dustbin
Module: Phase 2 - Waste / Throwing Action Detection
File: throwing_detection.py
=============================================================================

This module provides the independent, modular throwing/waste-disposal action
detection engine for the AI Interactive Rejecting Dustbin project.

Logical Separation of Components:
1. Person / Hand Detection Sub-layer
2. Landmark Extraction Sub-layer (MediaPipe)
3. Object Detection Sub-layer (YOLO)
4. Motion Tracking Sub-layer (Temporal trajectory & kinematics)
5. Throwing Decision Logic (Kinematic state machine & false-positive filters)

Key Features:
- Multi-frame kinematic analysis (velocity, acceleration, directional vector).
- False positive rejection (waving, static posture, upward motion, erratic jitter).
- Cooldown / debouncing mechanism to avoid duplicate triggers.
- Visual HUD annotations (trajectory trail, velocity vector, status badges).
- Fully decoupled from future phases and main.py; preserves Phase 1 integrity.
=============================================================================
"""

import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
phase1_dir = os.path.abspath(os.path.join(current_dir, "..", "phase1_person_detection"))

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if phase1_dir not in sys.path:
    sys.path.insert(0, phase1_dir)

try:
    from person_detection import PersonDetector, PersonDetectionResult
except ImportError:
    PersonDetector = None
    PersonDetectionResult = None


@dataclass
class TrajectoryPoint:
    x: int
    y: int
    timestamp: float


@dataclass
class MotionMetrics:
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    speed: float = 0.0
    net_dx: float = 0.0
    net_dy: float = 0.0
    path_length: float = 0.0
    direction_deg: float = 0.0
    is_oscillating_wave: bool = False
    is_downward_directed: bool = False
    peak_velocity: float = 0.0


@dataclass
class HandTrackingData:
    hand_detected: bool = False
    wrist_pos: Optional[Tuple[int, int]] = None
    palm_center: Optional[Tuple[int, int]] = None
    landmarks_px: List[Tuple[int, int]] = field(default_factory=list)
    object_near_hand: bool = False
    object_label: Optional[str] = None


@dataclass
class ThrowDetectionResult:
    throw_detected: bool
    confidence: float
    action_state: str
    person_present: bool
    hand_tracking: HandTrackingData
    metrics: MotionMetrics
    total_throws_counted: int
    cooldown_remaining_sec: float
    inference_time_ms: float = 0.0


class PersonDetectorAdapter:
    def __init__(self, yolo_model_path: Optional[str] = None, conf_threshold: float = 0.45):
        self.yolo_model_path = yolo_model_path
        self.conf_threshold = conf_threshold
        self.detector = None

        if PersonDetector is not None:
            try:
                self.detector = PersonDetector(
                    model_path=yolo_model_path,
                    conf_threshold=conf_threshold,
                )
            except Exception as e:
                print(f"[WARN] Failed to initialize Phase 1 PersonDetector: {e}")

        if self.detector is None:
            model_file = yolo_model_path or os.path.join(project_root, "assets", "models", "yolov8n.pt")
            if not os.path.isfile(model_file):
                model_file = "yolov8n.pt"
            self.model = YOLO(model_file)
        else:
            self.model = None

    def is_person_present(self, frame: np.ndarray) -> bool:
        if self.detector is not None:
            res = self.detector.detect(frame)
            return res.person_detected
        else:
            res = self.model.predict(frame, classes=[0], conf=self.conf_threshold, verbose=False)
            return len(res) > 0 and len(res[0].boxes) > 0


class HandLandmarkerExtractor:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            candidate = os.path.join(project_root, "assets", "models", "hand_landmarker.task")
            model_path = candidate if os.path.isfile(candidate) else "hand_landmarker.task"

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"MediaPipe hand landmarker model not found at: {model_path}")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def extract(self, frame: np.ndarray) -> HandTrackingData:
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect(mp_image)

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return HandTrackingData(hand_detected=False)

        landmarks = result.hand_landmarks[0]
        px_coords: List[Tuple[int, int]] = []
        for lm in landmarks:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            px_coords.append((cx, cy))

        wrist = px_coords[0]
        palm_cx = (px_coords[0][0] + px_coords[5][0] + px_coords[17][0]) // 3
        palm_cy = (px_coords[0][1] + px_coords[5][1] + px_coords[17][1]) // 3

        return HandTrackingData(
            hand_detected=True,
            wrist_pos=wrist,
            palm_center=(palm_cx, palm_cy),
            landmarks_px=px_coords,
        )


class ObjectDetector:
    DISPOSABLE_CLASSES = {
        39: "bottle",
        41: "cup",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        64: "mouse",
        67: "cell phone",
        73: "book",
    }

    def __init__(self, yolo_model_path: Optional[str] = None):
        model_file = yolo_model_path or os.path.join(project_root, "assets", "models", "yolov8n.pt")
        if not os.path.isfile(model_file):
            model_file = "yolov8n.pt"
        self.model = YOLO(model_file)

    def check_object_near_hand(
        self,
        frame: np.ndarray,
        hand_pos: Optional[Tuple[int, int]],
        max_dist_px: int = 150,
    ) -> Tuple[bool, Optional[str]]:
        if hand_pos is None:
            return False, None

        target_classes = list(self.DISPOSABLE_CLASSES.keys())
        results = self.model.predict(
            source=frame,
            classes=target_classes,
            conf=0.35,
            imgsz=480,
            verbose=False,
        )

        if not results or len(results) == 0 or results[0].boxes is None:
            return False, None

        hx, hy = hand_pos
        for box in results[0].boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy()
            cx = int((xyxy[0] + xyxy[2]) / 2)
            cy = int((xyxy[1] + xyxy[3]) / 2)

            dist = math.hypot(cx - hx, cy - hy)
            if dist <= max_dist_px:
                label = self.DISPOSABLE_CLASSES.get(cls_id, "object")
                return True, label

        return False, None


class MotionTracker:
    def __init__(self, window_size: int = 12, max_time_gap: float = 0.5):
        self.window_size = window_size
        self.max_time_gap = max_time_gap
        self.history: Deque[TrajectoryPoint] = deque(maxlen=window_size)

    def add_point(self, x: int, y: int, timestamp: Optional[float] = None) -> None:
        t = timestamp if timestamp is not None else time.perf_counter()
        if len(self.history) > 0 and (t - self.history[-1].timestamp) > self.max_time_gap:
            self.history.clear()
        self.history.append(TrajectoryPoint(x=x, y=y, timestamp=t))

    def reset(self) -> None:
        self.history.clear()

    def compute_metrics(self) -> MotionMetrics:
        if len(self.history) < 3:
            return MotionMetrics()

        p_first = self.history[0]
        p_last = self.history[-1]
        dt = p_last.timestamp - p_first.timestamp

        if dt <= 0.001:
            return MotionMetrics()

        net_dx = p_last.x - p_first.x
        net_dy = p_last.y - p_first.y

        vx = net_dx / dt
        vy = net_dy / dt
        net_speed = math.hypot(vx, vy)

        path_length = 0.0
        peak_v = 0.0
        sign_changes_x = 0
        prev_dx_sign = 0

        for i in range(1, len(self.history)):
            seg_dx = self.history[i].x - self.history[i - 1].x
            seg_dy = self.history[i].y - self.history[i - 1].y
            seg_dt = self.history[i].timestamp - self.history[i - 1].timestamp
            path_length += math.hypot(seg_dx, seg_dy)

            if seg_dt > 0:
                inst_v = math.hypot(seg_dx, seg_dy) / seg_dt
                if inst_v > peak_v:
                    peak_v = inst_v

            if abs(seg_dx) > 4:
                curr_sign = 1 if seg_dx > 0 else -1
                if prev_dx_sign != 0 and curr_sign != prev_dx_sign:
                    sign_changes_x += 1
                prev_dx_sign = curr_sign

        path_speed = path_length / dt if dt > 0 else 0.0
        speed = max(net_speed, path_speed)

        angle_rad = math.atan2(net_dy, net_dx)
        angle_deg = math.degrees(angle_rad)

        is_waving = (sign_changes_x >= 2) and (net_dy < max(40.0, abs(net_dx) * 0.8))
        is_downward = (net_dy > 45.0) and (vy > 180.0) and (angle_deg > 30.0 and angle_deg < 150.0)

        return MotionMetrics(
            velocity_x=vx,
            velocity_y=vy,
            speed=speed,
            net_dx=net_dx,
            net_dy=net_dy,
            path_length=path_length,
            direction_deg=angle_deg,
            is_oscillating_wave=is_waving,
            is_downward_directed=is_downward,
            peak_velocity=peak_v,
        )


class ThrowDecisionEngine:
    def __init__(
        self,
        min_throw_speed: float = 240.0,
        min_downward_dy: float = 50.0,
        cooldown_duration: float = 1.2,
    ):
        self.min_throw_speed = min_throw_speed
        self.min_downward_dy = min_downward_dy
        self.cooldown_duration = cooldown_duration

        self.last_throw_timestamp: float = -9999.0
        self.total_throws_counted: int = 0
        self.current_state: str = "IDLE"

    def evaluate(
        self,
        person_present: bool,
        hand_detected: bool,
        metrics: MotionMetrics,
        object_detected: bool = False,
        current_time: Optional[float] = None,
    ) -> Tuple[bool, float, str, float]:
        now = current_time if current_time is not None else time.perf_counter()
        cooldown_elapsed = now - self.last_throw_timestamp
        cooldown_remaining = max(0.0, self.cooldown_duration - cooldown_elapsed)

        if not person_present:
            self.current_state = "NO_PERSON"
            return False, 0.0, self.current_state, cooldown_remaining

        if cooldown_remaining > 0:
            self.current_state = "COOLDOWN"
            return False, 0.0, self.current_state, cooldown_remaining

        if metrics.is_oscillating_wave:
            self.current_state = "WAVING"
            return False, 0.0, self.current_state, 0.0

        if not hand_detected or (metrics.speed < 40.0 and metrics.path_length < 25.0):
            self.current_state = "IDLE"
            return False, 0.0, self.current_state, 0.0

        if metrics.velocity_y < -50.0 or metrics.net_dy < -20.0:
            self.current_state = "MOVING_UPWARD"
            return False, 0.0, self.current_state, 0.0

        if metrics.path_length > 100.0:
            direct_dist = math.hypot(metrics.net_dx, metrics.net_dy)
            efficiency = direct_dist / metrics.path_length
            if efficiency < 0.35:
                self.current_state = "ERRATIC_MOTION"
                return False, 0.0, self.current_state, 0.0

        if (
            metrics.is_downward_directed
            and metrics.velocity_y >= self.min_throw_speed
            and metrics.net_dy >= self.min_downward_dy
        ):
            base_score = min(0.85, 0.5 + (metrics.velocity_y / (self.min_throw_speed * 2.5)))

            if object_detected:
                base_score = min(1.0, base_score + 0.15)

            if metrics.peak_velocity > self.min_throw_speed * 1.5:
                base_score = min(1.0, base_score + 0.10)

            confidence = round(base_score, 2)

            self.last_throw_timestamp = now
            self.total_throws_counted += 1
            self.current_state = "THROW_DETECTED"
            return True, confidence, self.current_state, self.cooldown_duration

        self.current_state = "TRACKING_MOTION"
        return False, 0.0, self.current_state, 0.0


class ThrowingDetector:
    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        hand_model_path: Optional[str] = None,
        cooldown_sec: float = 1.2,
        check_objects: bool = True,
    ):
        self.person_adapter = PersonDetectorAdapter(yolo_model_path=yolo_model_path)
        self.hand_extractor = HandLandmarkerExtractor(model_path=hand_model_path)
        self.motion_tracker = MotionTracker(window_size=12)
        self.decision_engine = ThrowDecisionEngine(cooldown_duration=cooldown_sec)
        self.check_objects = check_objects

        if self.check_objects:
            try:
                self.object_detector = ObjectDetector(yolo_model_path=yolo_model_path)
            except Exception:
                self.object_detector = None
        else:
            self.object_detector = None

    def process_frame(self, frame: np.ndarray) -> ThrowDetectionResult:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame: Frame must be a non-empty numpy array.")

        t_start = time.perf_counter()

        person_present = self.person_adapter.is_person_present(frame)

        if not person_present:
            self.motion_tracker.reset()
            t_end = time.perf_counter()
            return ThrowDetectionResult(
                throw_detected=False,
                confidence=0.0,
                action_state="NO_PERSON",
                person_present=False,
                hand_tracking=HandTrackingData(),
                metrics=MotionMetrics(),
                total_throws_counted=self.decision_engine.total_throws_counted,
                cooldown_remaining_sec=0.0,
                inference_time_ms=(t_end - t_start) * 1000.0,
            )

        hand_data = self.hand_extractor.extract(frame)

        if hand_data.hand_detected and self.object_detector is not None:
            obj_found, obj_label = self.object_detector.check_object_near_hand(
                frame, hand_data.wrist_pos, max_dist_px=140
            )
            hand_data.object_near_hand = obj_found
            hand_data.object_label = obj_label

        if hand_data.hand_detected and hand_data.wrist_pos is not None:
            wx, wy = hand_data.wrist_pos
            self.motion_tracker.add_point(wx, wy, timestamp=t_start)
            metrics = self.motion_tracker.compute_metrics()
        else:
            metrics = MotionMetrics()

        throw_detected, conf, state, cooldown_rem = self.decision_engine.evaluate(
            person_present=person_present,
            hand_detected=hand_data.hand_detected,
            metrics=metrics,
            object_detected=hand_data.object_near_hand,
            current_time=t_start,
        )

        t_end = time.perf_counter()

        return ThrowDetectionResult(
            throw_detected=throw_detected,
            confidence=conf,
            action_state=state,
            person_present=person_present,
            hand_tracking=hand_data,
            metrics=metrics,
            total_throws_counted=self.decision_engine.total_throws_counted,
            cooldown_remaining_sec=cooldown_rem,
            inference_time_ms=(t_end - t_start) * 1000.0,
        )

    def annotate(
        self,
        frame: np.ndarray,
        result: ThrowDetectionResult,
        show_hud: bool = True,
    ) -> np.ndarray:
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        if result.hand_tracking.hand_detected and result.hand_tracking.landmarks_px:
            pts = result.hand_tracking.landmarks_px
            for pt in pts:
                cv2.circle(annotated, pt, 3, (0, 255, 255), -1)

            if result.hand_tracking.wrist_pos:
                cv2.circle(annotated, result.hand_tracking.wrist_pos, 6, (0, 165, 255), -1)

        hist = list(self.motion_tracker.history)
        if len(hist) > 1:
            for i in range(1, len(hist)):
                p1 = (hist[i - 1].x, hist[i - 1].y)
                p2 = (hist[i].x, hist[i].y)
                thickness = max(1, int(i * 3 / len(hist)))
                cv2.line(annotated, p1, p2, (255, 100, 0), thickness, cv2.LINE_AA)

            if result.hand_tracking.wrist_pos and result.metrics.speed > 80.0:
                wx, wy = result.hand_tracking.wrist_pos
                vx_norm = int(result.metrics.velocity_x * 0.15)
                vy_norm = int(result.metrics.velocity_y * 0.15)
                arrow_tip = (wx + vx_norm, wy + vy_norm)
                cv2.arrowedLine(annotated, (wx, wy), arrow_tip, (0, 255, 0), 2, tipLength=0.3)

        if show_hud:
            banner_h = 50
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

            if result.action_state == "THROW_DETECTED":
                status_text = f"★ THROW DETECTED! (Conf: {int(result.confidence * 100)}%)"
                status_color = (255, 0, 255)
            elif result.action_state == "WAVING":
                status_text = "✋ WAVING DETECTED (Ignored)"
                status_color = (0, 165, 255)
            elif result.action_state == "COOLDOWN":
                status_text = f"⏱ COOLDOWN ({result.cooldown_remaining_sec:.1f}s)"
                status_color = (200, 200, 0)
            elif result.action_state == "NO_PERSON":
                status_text = "○ NO PERSON PRESENT"
                status_color = (100, 100, 100)
            else:
                status_text = f"● READY / {result.action_state}"
                status_color = (0, 255, 128)

            cv2.putText(
                annotated,
                status_text,
                (15, 32),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                status_color,
                1,
                cv2.LINE_AA,
            )

            info_text = f"Throws: {result.total_throws_counted}  |  {result.inference_time_ms:.1f}ms"
            (tw, _), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(
                annotated,
                info_text,
                (w - tw - 15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def reset(self) -> None:
        self.motion_tracker.reset()
        self.decision_engine.last_throw_timestamp = -9999.0
        self.decision_engine.total_throws_counted = 0
        self.decision_engine.current_state = "IDLE"
