"""
=============================================================================
Project: AI Interactive Rejecting Dustbin
Module: Phase 5 - Final Application Integration
File: main.py
=============================================================================

Central Application Orchestrator for the AI Interactive Rejecting Dustbin.

Architecture:
main.py (Central Orchestrator)
  ├── Phase 1: modules.phase1_person_detection.person_detection
  ├── Phase 2: modules.phase2_throwing_detection.throwing_detection
  ├── Phase 3: modules.phase3_dustbin_control.dustbin_control
  └── Phase 4: modules.phase4_funny_response.funny_response

Responsibilities:
1. Coordinates the real-time computer vision pipeline (single camera capture per frame).
2. Manages system event state machine:
   - SEARCHING_PERSON -> PERSON_ENGAGED -> REJECTING_DISPOSAL -> COOLDOWN
3. Triggers synchronized physical lid actuation (Phase 3) and audio dialogue (Phase 4).
4. Enforces debouncing: One throwing action produces exactly ONE rejection event.
5. Seamlessly switches between live physical microcontroller hardware and simulation mode.
6. Robust exception handling and safe shutdown (safely closing lid and releasing camera).
=============================================================================
"""

import argparse
import enum
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# Configure system paths to ensure modular imports resolve seamlessly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# =============================================================================
# MODULAR IMPORTS (Phase 1 to Phase 4 Public Interfaces)
# =============================================================================
try:
    from modules.phase1_person_detection.person_detection import (
        CameraManager,
        PersonDetectionResult,
        PersonDetector,
    )
    from modules.phase2_throwing_detection.throwing_detection import (
        ThrowDetectionResult,
        ThrowingDetector,
    )
    from modules.phase3_dustbin_control.dustbin_control import (
        DustbinController,
        DustbinHardwareConfig,
        LidState,
        find_microcontroller_port,
    )
    from modules.phase4_funny_response.funny_response import (
        DustbinAudioEvent,
        FunnyResponsePlayer,
    )
except ImportError as e:
    print(f"[FATAL] Failed to import modular subsystem: {e}")
    print("Ensure you are executing main.py from the AI_Interactive_Dustbin directory.")
    sys.exit(1)


class SystemState(enum.Enum):
    """Real-time operational state of the rejecting dustbin system."""
    SEARCHING_PERSON = "SEARCHING_PERSON"     # Scanning scene for approaching individuals
    PERSON_ENGAGED = "PERSON_ENGAGED"         # Person present; provoking dialogue active
    REJECTING_DISPOSAL = "REJECTING_DISPOSAL" # Throw detected; servo snap & rejection line playing
    COOLDOWN = "COOLDOWN"                     # Post-rejection cooldown suppressing repeat triggers


@dataclass
class DustbinAppConfig:
    """Master configuration parameters for the integrated dustbin application."""
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    force_mock_hardware: bool = False
    baudrate: int = 9600
    lid_open_hold_sec: float = 1.2
    rejection_cooldown_sec: float = 2.5
    headless: bool = False
    max_runtime_sec: float = 0.0


class AIRejectingDustbinApp:
    """
    Central orchestrator coordinating vision, kinematics, servo lid hardware,
    and comedic audio commentary into a unified real-time loop.
    """

    def __init__(self, config: Optional[DustbinAppConfig] = None):
        self.config = config or DustbinAppConfig()
        self.state = SystemState.SEARCHING_PERSON

        # Subsystems
        self.camera: Optional[CameraManager] = None
        self.person_detector: Optional[PersonDetector] = None
        self.throwing_detector: Optional[ThrowingDetector] = None
        self.dustbin_controller: Optional[DustbinController] = None
        self.audio_player: Optional[FunnyResponsePlayer] = None

        # State tracking & telemetry
        self.is_running = False
        self.hardware_is_simulated = False
        self.detected_hardware_port: Optional[str] = None
        self.total_rejections_performed = 0
        self.state_enter_timestamp = time.perf_counter()
        self.last_throw_event_time = -9999.0
        self.fps = 0.0
        self._frame_count = 0
        self._fps_start_time = time.perf_counter()

    def initialize(self) -> bool:
        """
        Initializes all subsystems, detects hardware availability, loads AI models,
        and prepares the audio engine.
        """
        print("==================================================================")
        print("     AI INTERACTIVE REJECTING DUSTBIN - SYSTEM INITIALIZATION     ")
        print("==================================================================")

        # 1. Initialize Audio Response Engine (Phase 4)
        print("[INIT 1/4] Initializing Audio Personality Engine...")
        try:
            self.audio_player = FunnyResponsePlayer(cooldown_seconds=2.0, default_volume=1.0)
            avail_pro = len(self.audio_player.get_available_responses(DustbinAudioEvent.PERSON_DETECTED))
            avail_rej = len(self.audio_player.get_available_responses(DustbinAudioEvent.REJECTION_COMPLETE))
            print(f"  -> Audio Engine Ready. Provoke Tracks: {avail_pro}, Rejection Tracks: {avail_rej}")
        except Exception as e:
            print(f"[WARN] Audio engine initialization failed: {e}. Running in silent mode.")
            self.audio_player = FunnyResponsePlayer(enable_audio=False)

        # 2. Initialize Hardware / Servo Controller (Phase 3)
        print("[INIT 2/4] Initializing Dustbin Lid Servo Controller...")
        self.detected_hardware_port = find_microcontroller_port()

        if self.detected_hardware_port and not self.config.force_mock_hardware:
            print(f"  -> Physical Microcontroller detected on port: {self.detected_hardware_port}")
            hw_cfg = DustbinHardwareConfig(
                port=self.detected_hardware_port,
                baudrate=self.config.baudrate,
                default_open_duration=self.config.lid_open_hold_sec,
            )
            self.dustbin_controller = DustbinController(config=hw_cfg, mock_mode=False)
            if self.dustbin_controller.connect():
                self.hardware_is_simulated = False
                print("  -> Physical Microcontroller Handshake SUCCESSFUL.")
            else:
                print("  -> [WARN] Could not handshake with physical device. Falling back to Simulation Mode.")
                self.dustbin_controller = DustbinController(mock_mode=True)
                self.dustbin_controller.connect()
                self.hardware_is_simulated = True
        else:
            print("  -> Physical Hardware: NOT CONNECTED (or mock forced).")
            print("  -> Operating safely in SIMULATION / VIRTUAL SERVO MODE.")
            self.dustbin_controller = DustbinController(mock_mode=True)
            self.dustbin_controller.connect()
            self.hardware_is_simulated = True

        # 3. Initialize Vision Models (Phase 1 & Phase 2)
        print("[INIT 3/4] Initializing YOLO & MediaPipe Computer Vision Models...")
        try:
            self.person_detector = PersonDetector(conf_threshold=0.45)
            self.throwing_detector = ThrowingDetector(cooldown_sec=self.config.rejection_cooldown_sec)
            print("  -> Computer Vision & Kinematic Tracking Engines Loaded.")
        except Exception as e:
            print(f"[FATAL] Failed to initialize computer vision models: {e}")
            return False

        # 4. Initialize Camera Manager (Phase 1)
        print("[INIT 4/4] Connecting to Webcam Capture Feed...")
        try:
            self.camera = CameraManager(
                camera_index=self.config.camera_index,
                width=self.config.frame_width,
                height=self.config.frame_height,
            )
            self.camera.open()
            print(f"  -> Camera Stream Opened (Index: {self.config.camera_index}).")
        except RuntimeError as e:
            print(f"[FATAL] Camera capture failure: {e}")
            return False

        print("==================================================================")
        print("            ALL SUBSYSTEMS INITIALIZED & OPERATIONAL              ")
        print("==================================================================")
        return True

    def _execute_rejection_sequence(self) -> None:
        """
        Coordinates the funny rejection event:
        1. Actuates servo to perform the rejection movement.
        2. Plays comedic rejection audio dialogue.
        3. Enforces single-response debouncing.
        """
        print("\n[EVENT] ★ THROWING ACTION DETECTED! Executing Comedic Rejection...")

        # 1. Trigger comedic voice line
        if self.audio_player is not None:
            self.audio_player.play_response(DustbinAudioEvent.REJECTION_COMPLETE, force=True)

        # 2. Trigger hardware lid actuation (Rejection snap)
        if self.dustbin_controller is not None:
            self.dustbin_controller.reject_action()

        self.total_rejections_performed += 1
        self.last_throw_event_time = time.perf_counter()
        self.state = SystemState.REJECTING_DISPOSAL
        self.state_enter_timestamp = time.perf_counter()

    def process_single_frame(
        self, frame: np.ndarray, current_time: Optional[float] = None
    ) -> Tuple[np.ndarray, PersonDetectionResult, ThrowDetectionResult]:
        """
        Executes one complete pass of the coordinated detection pipeline:
        Frame -> Person Detection -> Throwing Kinematics -> State Machine -> Annotated Output
        """
        now = current_time if current_time is not None else time.perf_counter()

        # Step 1: Detect Person (Phase 1)
        person_result = self.person_detector.detect(frame)

        # Step 2: Detect Throwing Kinematics (Phase 2)
        throw_result = self.throwing_detector.process_frame(frame)

        # Step 3: Event Coordination & State Machine
        if not person_result.person_detected:
            if self.state != SystemState.SEARCHING_PERSON:
                self.state = SystemState.SEARCHING_PERSON
                self.state_enter_timestamp = now
        else:
            # Person is present in front of the dustbin
            if self.state == SystemState.SEARCHING_PERSON:
                self.state = SystemState.PERSON_ENGAGED
                self.state_enter_timestamp = now
                # Trigger welcoming/provoking voice line
                if self.audio_player is not None:
                    self.audio_player.play_response(DustbinAudioEvent.PERSON_DETECTED, force=False)

            elif self.state == SystemState.PERSON_ENGAGED:
                # Watch for disposal/throwing motion
                if throw_result.throw_detected:
                    self._execute_rejection_sequence()

            elif self.state == SystemState.REJECTING_DISPOSAL:
                # Hold rejection state briefly before moving to debouncing cooldown
                if (now - self.state_enter_timestamp) >= self.config.lid_open_hold_sec:
                    self.state = SystemState.COOLDOWN
                    self.state_enter_timestamp = now
                    if self.dustbin_controller is not None:
                        self.dustbin_controller.close_lid()

            elif self.state == SystemState.COOLDOWN:
                # Cooldown period to guarantee ONE response per throw
                if (now - self.state_enter_timestamp) >= self.config.rejection_cooldown_sec:
                    self.state = SystemState.PERSON_ENGAGED
                    self.state_enter_timestamp = now

        # Step 4: Render Unified Master HUD
        annotated_frame = self._render_master_hud(frame, person_result, throw_result)
        return annotated_frame, person_result, throw_result

    def _render_master_hud(
        self,
        frame: np.ndarray,
        person_res: PersonDetectionResult,
        throw_res: ThrowDetectionResult,
    ) -> np.ndarray:
        """Renders an informative top status banner and visual detection indicators."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw Person bounding boxes from Phase 1
        if person_res.primary_person is not None:
            p = person_res.primary_person
            cv2.rectangle(annotated, (p.x1, p.y1), (p.x2, p.y2), (0, 255, 128), 2)
            cv2.putText(
                annotated,
                f"User ({p.confidence * 100:.0f}%)",
                (p.x1, max(20, p.y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 128),
                1,
                cv2.LINE_AA,
            )

        # Draw MediaPipe hand landmarks and motion trails from Phase 2
        if throw_res.hand_tracking.hand_detected and throw_res.hand_tracking.landmarks_px:
            for pt in throw_res.hand_tracking.landmarks_px:
                cv2.circle(annotated, pt, 3, (0, 255, 255), -1)
            if throw_res.hand_tracking.wrist_pos:
                cv2.circle(annotated, throw_res.hand_tracking.wrist_pos, 5, (0, 165, 255), -1)

        # Master HUD Top Bar
        hud_height = 55
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_height), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        # State badge text & color
        state_colors = {
            SystemState.SEARCHING_PERSON: (120, 120, 120),
            SystemState.PERSON_ENGAGED: (0, 255, 128),
            SystemState.REJECTING_DISPOSAL: (0, 0, 255),
            SystemState.COOLDOWN: (0, 200, 255),
        }
        badge_color = state_colors.get(self.state, (200, 200, 200))
        badge_text = f"STATE: {self.state.value}"

        cv2.putText(
            annotated,
            badge_text,
            (15, 24),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            badge_color,
            1,
            cv2.LINE_AA,
        )

        # Hardware & Servo status text
        hw_mode = "SIMULATION" if self.hardware_is_simulated else f"LIVE ({self.detected_hardware_port})"
        lid_st = self.dustbin_controller.state.value if self.dustbin_controller else "UNKNOWN"
        hw_info = f"Hardware: {hw_mode}  |  Lid: {lid_st}  |  Rejections: {self.total_rejections_performed}"

        cv2.putText(
            annotated,
            hw_info,
            (15, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # FPS / Telemetry right-aligned
        perf_text = f"FPS: {self.fps:.1f}  |  Inf: {throw_res.inference_time_ms:.0f}ms"
        (tw, _), _ = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(
            annotated,
            perf_text,
            (w - tw - 15, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        return annotated

    def run(self) -> None:
        """Executes the central application loop until interrupted."""
        if not self.initialize():
            print("[ERROR] Application initialization failed.")
            self.shutdown()
            return

        window_name = "AI Interactive Rejecting Dustbin - Final Integrated System"
        if not self.config.headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)

        print("\n[RUNNING] Interactive Rejecting Dustbin Loop Started.")
        print("Controls: Press 'q' or 'ESC' to safely stop.\n")

        self.is_running = True
        loop_start_time = time.perf_counter()
        self._fps_start_time = time.perf_counter()
        self._frame_count = 0

        try:
            while self.is_running:
                ret, frame = self.camera.read_frame()
                if not ret or frame is None:
                    print("[WARN] Camera frame dropped or disconnected.")
                    break

                # Flip horizontally for natural mirror view
                frame = cv2.flip(frame, 1)

                # Process unified detection pipeline
                annotated, person_res, throw_res = self.process_single_frame(frame)

                # Calculate smoothed FPS
                self._frame_count += 1
                fps_dt = time.perf_counter() - self._fps_start_time
                if fps_dt >= 1.0:
                    self.fps = self._frame_count / fps_dt
                    self._frame_count = 0
                    self._fps_start_time = time.perf_counter()

                # Display GUI feed
                if not self.config.headless:
                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("q"), 27]:
                        print("\n[USER] Exit requested via keyboard.")
                        break

                # Check runtime limit (for automated testing / timed runs)
                if self.config.max_runtime_sec > 0:
                    if (time.perf_counter() - loop_start_time) >= self.config.max_runtime_sec:
                        break

        except KeyboardInterrupt:
            print("\n[INTERRUPT] KeyboardInterrupt received.")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Ensures safe, graceful termination of camera, hardware, and audio engines."""
        print("\n==================================================================")
        print("          SHUTTING DOWN AI REJECTING DUSTBIN SAFELY...           ")
        print("==================================================================")

        # 1. Safely close dustbin lid and disconnect serial
        if self.dustbin_controller is not None:
            try:
                print("  -> Closing dustbin lid and disconnecting serial...")
                self.dustbin_controller.close_lid()
                self.dustbin_controller.disconnect()
            except Exception as e:
                print(f"  -> Notice during hardware disconnect: {e}")

        # 2. Stop audio engine
        if self.audio_player is not None:
            try:
                print("  -> Stopping audio channels...")
                self.audio_player.stop()
            except Exception as e:
                print(f"  -> Notice during audio stop: {e}")

        # 3. Release camera feed and close GUI windows
        if self.camera is not None:
            try:
                print("  -> Releasing webcam handle...")
                self.camera.release()
            except Exception as e:
                print(f"  -> Notice during camera release: {e}")

        cv2.destroyAllWindows()
        self.is_running = False
        print("[SUCCESS] Application cleanly shut down.\n")


def parse_arguments() -> DustbinAppConfig:
    """Parses CLI flags for the application."""
    parser = argparse.ArgumentParser(
        description="AI Interactive Rejecting Dustbin - Final Integrated Application"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Webcam device index (default: 0)"
    )
    parser.add_argument(
        "--mock-hardware",
        action="store_true",
        help="Force simulation/mock mode for microcontroller servo control",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying OpenCV GUI window (ideal for CI/CD)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Auto-close application after N seconds (0 = run indefinitely)",
    )
    args = parser.parse_args()

    return DustbinAppConfig(
        camera_index=args.camera,
        force_mock_hardware=args.mock_hardware,
        headless=args.headless,
        max_runtime_sec=args.duration,
    )


if __name__ == "__main__":
    config = parse_arguments()
    app = AIRejectingDustbinApp(config=config)
    app.run()
