"""
=============================================================================
Project: AI Interactive Rejecting Dustbin
Module: Phase 4 - Funny Personality / Audio Response
File: test_funny_response.py
=============================================================================

Independent test suite for Phase 4 (Funny Personality / Audio Response).

Test Cases Covered:
1. Valid Sound Selection (PERSON_DETECTED loads and selects audio file)
2. Multiple Sound Discovery (Detects all files in provoke and rejection directories)
3. Random Response Selection & Anti-Repetition (Avoids identical consecutive responses)
4. Missing Sound / Empty Directory Resilience (Fails gracefully without crashing)
5. Invalid Event Handling (Gracefully rejects unknown events)
6. Repeated Playback Protection / Cooldown Throttling (Suppresses rapid triggers)
7. Safe Playback Interruption (Calling stop() when idle and when active)

Interactive Live Speaker Test:
    python test_funny_response.py --live
=============================================================================
"""

import argparse
import os
import sys
import tempfile
import time
import unittest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from funny_response import (
    DustbinAudioEvent,
    FunnyResponsePlayer,
    get_default_sounds_dir,
)


class TestPhase4FunnyResponse(unittest.TestCase):
    """Rigorous unit test suite validating the Phase 4 FunnyResponsePlayer module."""

    def setUp(self):
        self.sounds_dir = get_default_sounds_dir()
        self.player = FunnyResponsePlayer(
            sounds_dir=self.sounds_dir,
            cooldown_seconds=0.5,
            default_volume=0.8,
            enable_audio=True,
        )

    def tearDown(self):
        self.player.stop()

    def test_1_valid_sound_loading(self):
        """Test Case 1: Valid PERSON_DETECTED response can be selected and loaded."""
        print("\n[TEST 1] Testing Valid Sound Loading...")
        provoke_file = self.player.select_response_file("PERSON_DETECTED")
        self.assertIsNotNone(provoke_file, "Failed to select a PERSON_DETECTED audio file.")
        self.assertTrue(os.path.isfile(provoke_file), f"Selected file does not exist on disk: {provoke_file}")
        self.assertTrue(provoke_file.lower().endswith(".mp3"), "Expected .mp3 audio format.")

        success = self.player.play_response("PERSON_DETECTED", force=True, blocking=False)
        self.assertTrue(success, "play_response('PERSON_DETECTED') returned False.")
        self.assertIsNotNone(self.player.last_played_file)
        self.assertEqual(self.player.last_event, DustbinAudioEvent.PERSON_DETECTED)
        print(f"  -> Successfully loaded & initiated: {os.path.basename(provoke_file)} (PASSED)")

    def test_2_multiple_sounds_discovery(self):
        """Test Case 2: Verifies dynamic discovery of multiple audio files in each event folder."""
        print("\n[TEST 2] Testing Multiple Sound Discovery...")
        provoke_files = self.player.get_available_responses("PERSON_DETECTED")
        rejection_files = self.player.get_available_responses("REJECTION_COMPLETE")

        self.assertGreaterEqual(len(provoke_files), 3, "Expected at least 3 provoke audio tracks.")
        self.assertGreaterEqual(len(rejection_files), 3, "Expected at least 3 rejection audio tracks.")

        for path in provoke_files + rejection_files:
            self.assertTrue(os.path.isfile(path), f"Discovered file not found: {path}")

        print(f"  -> Discovered {len(provoke_files)} provoke tracks: {[os.path.basename(f) for f in provoke_files]}")
        print(f"  -> Discovered {len(rejection_files)} rejection tracks: {[os.path.basename(f) for f in rejection_files]} (PASSED)")

    def test_3_random_selection_and_anti_repetition(self):
        """Test Case 3: Response selection variety and consecutive anti-repetition check."""
        print("\n[TEST 3] Testing Random Selection & Anti-Repetition...")
        available = self.player.get_available_responses("PERSON_DETECTED")
        self.assertGreater(len(available), 1, "Need at least 2 files to test anti-repetition.")

        selected_sequence = []
        for i in range(10):
            chosen = self.player.select_response_file("PERSON_DETECTED")
            selected_sequence.append(chosen)

        for i in range(1, len(selected_sequence)):
            prev_file = selected_sequence[i - 1]
            curr_file = selected_sequence[i]
            self.assertNotEqual(
                prev_file,
                curr_file,
                f"Consecutive repetition detected at step {i}: {os.path.basename(curr_file)} was picked twice in a row.",
            )

        unique_picks = set(selected_sequence)
        self.assertGreaterEqual(len(unique_picks), 2, "Expected variety in selected audio files.")
        print(f"  -> Selection Sequence (10 trials, 0 consecutive repeats):")
        for idx, f in enumerate(selected_sequence[:5]):
            print(f"     Step {idx + 1}: {os.path.basename(f)}")
        print("  -> Anti-repetition confirmed (PASSED)")

    def test_4_missing_sound_handling(self):
        """Test Case 4: Graceful handling of missing files and empty directories."""
        print("\n[TEST 4] Testing Missing Sound / Empty Directory Resilience...")
        with tempfile.TemporaryDirectory() as empty_dir:
            temp_player = FunnyResponsePlayer(sounds_dir=empty_dir, enable_audio=False)

            files = temp_player.get_available_responses("PERSON_DETECTED")
            self.assertEqual(files, [], "Empty folder should return empty file list.")

            result = temp_player.play_response("PERSON_DETECTED")
            self.assertFalse(result, "Playing from empty directory must safely return False without exception.")

        print("  -> Safely caught missing directory/files without application crash (PASSED)")

    def test_5_invalid_event_handling(self):
        """Test Case 5: Graceful rejection of unrecognized events."""
        print("\n[TEST 5] Testing Invalid Event Handling...")
        result_invalid_str = self.player.play_response("FLYING_SAUCER_DETECTED")
        self.assertFalse(result_invalid_str, "Unrecognized event string must return False.")

        result_empty_str = self.player.play_response("")
        self.assertFalse(result_empty_str, "Empty event string must return False.")

        responses = self.player.get_available_responses("NOT_AN_EVENT")
        self.assertEqual(responses, [])
        print("  -> Handled unrecognized events cleanly (PASSED)")

    def test_6_repeated_playback_protection(self):
        """Test Case 6: Cooldown mechanism suppresses spamming repeated events."""
        print("\n[TEST 6] Testing Repeated Playback Protection (Cooldown)...")
        self.player.set_cooldown(2.0)

        t1 = self.player.play_response("REJECTION_COMPLETE", force=True)
        self.assertTrue(t1, "Initial trigger should succeed.")

        t2 = self.player.play_response("REJECTION_COMPLETE", force=False)
        self.assertFalse(t2, "Immediate repeated trigger must be blocked by cooldown.")

        t3 = self.player.play_response("REJECTION_COMPLETE", force=False)
        self.assertFalse(t3, "Spam trigger must be blocked by cooldown.")

        t4_forced = self.player.play_response("REJECTION_COMPLETE", force=True)
        self.assertTrue(t4_forced, "Forced trigger should bypass cooldown.")
        print("  -> Rapid event spam successfully throttled by cooldown (PASSED)")

    def test_7_safe_stop(self):
        """Test Case 7: stop() method works safely both when idle and active."""
        print("\n[TEST 7] Testing Safe Stop Operation...")
        self.player.stop()

        self.player.play_response("PERSON_DETECTED", force=True)
        time.sleep(0.05)
        self.player.stop()
        self.assertFalse(self.player.is_playing())
        print("  -> stop() executed safely in both idle and active states (PASSED)")


def run_live_audio_test():
    print("==================================================================")
    print("   AI Interactive Rejecting Dustbin - Phase 4 Live Audio Test    ")
    print("==================================================================")
    print("Initializing FunnyResponsePlayer...")

    player = FunnyResponsePlayer(cooldown_seconds=1.0, default_volume=1.0)

    print("\n1. Testing PERSON_DETECTED (Provoking Dialogue)...")
    success1 = player.play_response("PERSON_DETECTED", force=True, blocking=False)
    if success1:
        print(f"  -> Now playing: {os.path.basename(player.last_played_file)}")
        print("  -> Speaking through laptop speaker for 4 seconds...")
        time.sleep(4.0)
        player.stop()
    else:
        print("  -> [FAILED] Could not play PERSON_DETECTED audio.")

    time.sleep(1.0)

    print("\n2. Testing REJECTION_COMPLETE (Funny Rejection Dialogue)...")
    success2 = player.play_response("REJECTION_COMPLETE", force=True, blocking=False)
    if success2:
        print(f"  -> Now playing: {os.path.basename(player.last_played_file)}")
        print("  -> Speaking through laptop speaker for 4 seconds...")
        time.sleep(4.0)
        player.stop()
    else:
        print("  -> [FAILED] Could not play REJECTION_COMPLETE audio.")

    print("\n[LIVE AUDIO TEST COMPLETE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 Funny Personality / Audio Tests")
    parser.add_argument("--live", action="store_true", help="Execute live audio playback through speakers")
    args = parser.parse_args()

    if args.live:
        run_live_audio_test()
    else:
        print("==================================================================")
        print("   AI Interactive Rejecting Dustbin - Phase 4 Automated Tests    ")
        print("==================================================================")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase4FunnyResponse)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        if not result.wasSuccessful():
            sys.exit(1)
