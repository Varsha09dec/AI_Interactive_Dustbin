"""
=============================================================================
Project: AI Interactive Rejecting Dustbin
Module: Phase 1 - Person Detection
File: person_detection.py
=============================================================================

This module provides the independent, modular person detection engine for the
AI Interactive Rejecting Dustbin project.

Key Features:
- YOLO-based person identification (class 0 only) for high-speed CPU execution.
- Extraction of bounding boxes, confidence scores, centroids, and primary person.
- Frame annotation utilities with modern HUD visuals (status banner, bounding boxes).
- Resilient camera abstraction (CameraManager) with robust failure handling.
- Completely decoupled from future phases and main.py.
=============================================================================
"""

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class PersonDetection:
    """Represents a single detected person within a frame."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    center_x: int
    center_y: int
    width: int
    height: int
    area: int

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.center_x, self.center_y)


@dataclass
class PersonDetectionResult:
    """Encapsulates the complete detection outcome for a single video frame."""
    person_detected: bool
    count: int
    persons: List[PersonDetection]
    primary_person: Optional[PersonDetection]
    frame_width: int
    frame_height: int
    inference_time_ms: float


def get_default_model_path() -> str:
    """Locates the default YOLO model weight file across standard project directories."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    candidate_paths = [
        os.path.join(project_root, "assets", "models", "yolov8n.pt"),
        os.path.join(project_root, "yolov8n.pt"),
        os.path.join(current_dir, "yolov8n.pt"),
        "yolov8n.pt",
    ]

    for path in candidate_paths:
        if os.path.isfile(path):
            return path

    return "yolov8n.pt"


class PersonDetector:
    """
    Modular YOLO-based person detector optimized for real-time edge/laptop inference.
    Exclusively filters for COCO class 0 ('person') to maximize FPS and avoid
    unnecessary bounding box parsing.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        imgsz: int = 640,
        device: str = "cpu",
    ):
        self.model_path = model_path if model_path else get_default_model_path()
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device

        if not os.path.isfile(self.model_path) and (
            os.path.isabs(self.model_path)
            or os.sep in self.model_path
            or "/" in self.model_path
            or not self.model_path.startswith("yolo")
            or "non_existent" in self.model_path
        ):
            raise FileNotFoundError(
                f"YOLO model weights not found at path: {self.model_path}"
            )

        try:
            self.model = YOLO(self.model_path)
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model from '{self.model_path}': {e}"
            )

    def detect(self, frame: np.ndarray) -> PersonDetectionResult:
        """Runs person detection on a single image or video frame."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid input frame: Frame must be a non-empty numpy array.")

        height, width = frame.shape[:2]
        start_time = time.perf_counter()

        results = self.model.predict(
            source=frame,
            classes=[0],
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        detections: List[PersonDetection] = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                x1, y1, x2, y2 = [int(v) for v in xyxy]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width - 1, x2), min(height - 1, y2)

                box_w = max(0, x2 - x1)
                box_h = max(0, y2 - y1)
                area = box_w * box_h
                cx = x1 + (box_w // 2)
                cy = y1 + (box_h // 2)

                detections.append(
                    PersonDetection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=conf,
                        center_x=cx,
                        center_y=cy,
                        width=box_w,
                        height=box_h,
                        area=area,
                    )
                )

        detections.sort(key=lambda d: d.area, reverse=True)
        primary = detections[0] if len(detections) > 0 else None

        return PersonDetectionResult(
            person_detected=len(detections) > 0,
            count=len(detections),
            persons=detections,
            primary_person=primary,
            frame_width=width,
            frame_height=height,
            inference_time_ms=inference_time_ms,
        )

    def annotate(
        self,
        frame: np.ndarray,
        result: PersonDetectionResult,
        show_banner: bool = True,
    ) -> np.ndarray:
        """Draws visual annotations onto a copy of the frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for idx, person in enumerate(result.persons):
            is_primary = (person is result.primary_person)
            box_color = (0, 255, 128) if is_primary else (0, 220, 255)
            thickness = 2 if is_primary else 1

            cv2.rectangle(
                annotated,
                (person.x1, person.y1),
                (person.x2, person.y2),
                box_color,
                thickness,
            )

            corner_len = min(20, person.width // 4, person.height // 4)
            if corner_len > 5:
                cv2.line(annotated, (person.x1, person.y1), (person.x1 + corner_len, person.y1), (255, 255, 255), 3)
                cv2.line(annotated, (person.x1, person.y1), (person.x1, person.y1 + corner_len), (255, 255, 255), 3)
                cv2.line(annotated, (person.x2, person.y2), (person.x2 - corner_len, person.y2), (255, 255, 255), 3)
                cv2.line(annotated, (person.x2, person.y2), (person.x2, person.y2 - corner_len), (255, 255, 255), 3)

            cv2.circle(annotated, person.center, 4, box_color, -1)

            tag = f"Person #{idx + 1}" + (" [PRIMARY]" if is_primary else "")
            conf_str = f" {person.confidence * 100:.1f}%"
            label = tag + conf_str

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

            tag_y1 = max(0, person.y1 - th - 8)
            tag_y2 = person.y1
            tag_x2 = min(w, person.x1 + tw + 10)

            cv2.rectangle(annotated, (person.x1, tag_y1), (tag_x2, tag_y2), box_color, -1)
            cv2.putText(
                annotated,
                label,
                (person.x1 + 5, tag_y2 - baseline),
                font,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        if show_banner:
            banner_height = 45
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

            if result.person_detected:
                status_text = f"● PERSON DETECTED  |  Count: {result.count}"
                status_color = (0, 255, 128)
            else:
                status_text = "○ NO PERSON DETECTED"
                status_color = (80, 80, 240)

            cv2.putText(
                annotated,
                status_text,
                (15, 28),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                status_color,
                1,
                cv2.LINE_AA,
            )

            latency_text = f"Latency: {result.inference_time_ms:.1f}ms"
            (lw, _), _ = cv2.getTextSize(latency_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(
                annotated,
                latency_text,
                (w - lw - 15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        return annotated


class CameraManager:
    """Robust camera capture interface for laptop webcam access with graceful failure handling."""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Camera failure: Unable to open webcam at index {self.camera_index}. "
                "Ensure the camera is connected and not currently used by another application."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None or not self.cap.isOpened():
            return False, None
        ret, frame = self.cap.read()
        return ret, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
