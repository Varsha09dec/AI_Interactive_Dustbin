"""
=============================================================================
Project: AI Interactive Rejecting Dustbin
Module: Phase 4 - Funny Personality / Audio Response
File: funny_response.py
=============================================================================

This module provides the independent, modular funny personality and audio-response
playback engine for the AI Interactive Rejecting Dustbin project.

Supported Dialogue Scenarios:
1. PERSON_DETECTED:
   Provoking / teasing audio lines encouraging the approaching user to dispose
   of their waste (assets/sounds/provoke/ or assets/sounds/provocate/).
2. REJECTION_COMPLETE / LID_CLOSED:
   Funny rejection lines played after the dustbin lid snaps shut and rejects the
   user's waste disposal attempt (assets/sounds/rejection/).

Key Features:
- Non-blocking audio playback using pygame.mixer so computer vision loops never stutter.
- Dynamic audio discovery: automatically recognizes all audio tracks (.mp3, .wav, .ogg, etc.)
  without requiring code modifications when new tracks are added.
- Anti-repetition algorithm: prevents playing the exact same track twice consecutively.
- Repeated event protection (cooldown / debouncing) to suppress spam triggers.
- Robust error handling: missing sounds, empty folders, or playback errors fail gracefully.
- Completely decoupled from YOLO, MediaPipe, OpenCV, Arduino, and main.py.
=============================================================================
"""

import enum
import os
import random
import sys
import time
from typing import Dict, List, Optional, Set, Union

try:
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    import pygame
except ImportError:
    pygame = None


class DustbinAudioEvent(enum.Enum):
    """Supported event categories that trigger funny audio dialogue."""
    PERSON_DETECTED = "PERSON_DETECTED"
    REJECTION_COMPLETE = "REJECTION_COMPLETE"
    LID_CLOSED = "LID_CLOSED"


SUPPORTED_AUDIO_EXTENSIONS = (".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac")


def get_default_sounds_dir() -> str:
    """Resolves the default project assets/sounds directory relative to module location."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    return os.path.join(project_root, "assets", "sounds")


class FunnyResponsePlayer:
    """
    Modular audio response controller driving the rejecting dustbin's personality.
    Selects, randomizes, and plays contextual voice tracks through the laptop's speakers.
    """

    def __init__(
        self,
        sounds_dir: Optional[str] = None,
        cooldown_seconds: float = 3.5,
        default_volume: float = 1.0,
        enable_audio: bool = True,
    ):
        self.sounds_dir = sounds_dir or get_default_sounds_dir()
        self.cooldown_seconds = max(0.1, float(cooldown_seconds))
        self.default_volume = max(0.0, min(1.0, float(default_volume)))
        self.enable_audio = enable_audio

        self._is_initialized = False
        self._current_channel = None
        self._current_sound = None
        self._last_played_file: Optional[str] = None
        self._last_event: Optional[DustbinAudioEvent] = None
        self._last_play_timestamp: float = -9999.0

        self._last_file_by_event: Dict[str, str] = {}

        if self.enable_audio:
            self._init_mixer()

    def _init_mixer(self) -> bool:
        if pygame is None:
            print("[WARN] pygame is not installed. Audio playback will operate in mock/silent mode.")
            return False

        if pygame.mixer.get_init():
            self._is_initialized = True
            return True

        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            self._is_initialized = True
            return True
        except Exception as e:
            print(f"[WARN] Failed to initialize audio mixer: {e}. Operating in silent mode.")
            self._is_initialized = False
            return False

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def last_played_file(self) -> Optional[str]:
        return self._last_played_file

    @property
    def last_event(self) -> Optional[DustbinAudioEvent]:
        return self._last_event

    def set_cooldown(self, seconds: float) -> None:
        self.cooldown_seconds = max(0.0, float(seconds))

    def set_volume(self, volume: float) -> None:
        self.default_volume = max(0.0, min(1.0, float(volume)))
        if self._current_channel is not None and pygame is not None:
            try:
                self._current_channel.set_volume(self.default_volume)
            except Exception:
                pass

    def is_playing(self) -> bool:
        if not self.enable_audio or not self._is_initialized or pygame is None:
            return False

        try:
            if self._current_channel is not None and self._current_channel.get_busy():
                return True
            if pygame.mixer.get_busy():
                return True
        except Exception:
            pass

        return False

    def stop(self) -> None:
        if not self.enable_audio or not self._is_initialized or pygame is None:
            return

        try:
            if self._current_channel is not None:
                self._current_channel.stop()
            pygame.mixer.stop()
        except Exception as e:
            print(f"[DEBUG] Notice while stopping audio: {e}")

    def _normalize_event(self, event: Union[str, DustbinAudioEvent]) -> Optional[DustbinAudioEvent]:
        if isinstance(event, DustbinAudioEvent):
            return event

        if not isinstance(event, str):
            return None

        clean_str = event.strip().upper()
        if clean_str == "PERSON_DETECTED":
            return DustbinAudioEvent.PERSON_DETECTED
        elif clean_str in ["REJECTION_COMPLETE", "LID_CLOSED"]:
            return DustbinAudioEvent.REJECTION_COMPLETE

        return None

    def _get_event_directory(self, event: DustbinAudioEvent) -> Optional[str]:
        if not os.path.isdir(self.sounds_dir):
            return None

        if event == DustbinAudioEvent.PERSON_DETECTED:
            for folder in ["provoke", "provocate"]:
                cand = os.path.join(self.sounds_dir, folder)
                if os.path.isdir(cand):
                    return cand
        elif event in [DustbinAudioEvent.REJECTION_COMPLETE, DustbinAudioEvent.LID_CLOSED]:
            for folder in ["rejection", "reject"]:
                cand = os.path.join(self.sounds_dir, folder)
                if os.path.isdir(cand):
                    return cand

        return None

    def get_available_responses(self, event: Union[str, DustbinAudioEvent]) -> List[str]:
        norm_event = self._normalize_event(event)
        if norm_event is None:
            return []

        dir_path = self._get_event_directory(norm_event)
        if dir_path is None or not os.path.isdir(dir_path):
            return []

        try:
            files = [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if f.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS)
            ]
            return sorted(files)
        except Exception as e:
            print(f"[WARN] Error reading directory '{dir_path}': {e}")
            return []

    def select_response_file(self, event: Union[str, DustbinAudioEvent]) -> Optional[str]:
        norm_event = self._normalize_event(event)
        if norm_event is None:
            return None

        available_files = self.get_available_responses(norm_event)
        if not available_files:
            return None

        if len(available_files) == 1:
            chosen = available_files[0]
            self._last_file_by_event[norm_event.value] = chosen
            return chosen

        event_key = norm_event.value
        last_played = self._last_file_by_event.get(event_key)

        candidates = [f for f in available_files if f != last_played]
        if not candidates:
            candidates = available_files

        chosen = random.choice(candidates)
        self._last_file_by_event[event_key] = chosen
        return chosen

    def play_response(
        self,
        event: Union[str, DustbinAudioEvent],
        force: bool = False,
        blocking: bool = False,
    ) -> bool:
        norm_event = self._normalize_event(event)
        if norm_event is None:
            print(f"[WARN] Invalid audio event supplied: '{event}'. Ignored.")
            return False

        now = time.perf_counter()
        elapsed_since_last_play = now - self._last_play_timestamp

        if not force and elapsed_since_last_play < self.cooldown_seconds:
            return False

        if not force and self.is_playing():
            return False

        file_to_play = self.select_response_file(norm_event)
        if file_to_play is None:
            print(f"[WARN] No valid audio files found for event: {norm_event.value}")
            return False

        if not os.path.isfile(file_to_play):
            print(f"[WARN] Audio file does not exist on disk: {file_to_play}")
            return False

        if not self.enable_audio or not self._is_initialized or pygame is None:
            self._last_played_file = file_to_play
            self._last_event = norm_event
            self._last_play_timestamp = now
            return True

        try:
            self.stop()

            sound = pygame.mixer.Sound(file_to_play)
            sound.set_volume(self.default_volume)
            channel = sound.play()

            if channel is not None:
                self._current_channel = channel
            self._current_sound = sound

            self._last_played_file = file_to_play
            self._last_event = norm_event
            self._last_play_timestamp = now

            if blocking:
                while self.is_playing():
                    time.sleep(0.05)

            return True

        except Exception as e:
            print(f"[ERROR] Failed to play audio track '{file_to_play}': {e}")
            return False
