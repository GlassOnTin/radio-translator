# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the streaming silero VAD adapter: state continuity across feed
blockings, and speech-vs-music discrimination on the recorded fixtures.
CPU only (silero ONNX), no GPU."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asr  # noqa: E402
import chunker  # noqa: E402
import vad  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def feed_in_blocks(v, audio, block):
    pcm = (audio * 32767).astype(np.int16).tobytes()
    decisions = []
    for i in range(0, len(pcm), block):
        decisions.append(v.is_speech(pcm[i:i + block], 16000))
    return decisions


class TestStreamingState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audio = asr.load_wav(os.path.join(HERE, "samples", "lang_zh.wav"))[:16000 * 5]

    def test_block_invariant_probabilities(self):
        a, b, c = vad.SileroVad(), vad.SileroVad(), vad.SileroVad()
        feed_in_blocks(a, self.audio, 480 * 2)   # one shot
        feed_in_blocks(b, self.audio, 960)       # 30 ms frames
        feed_in_blocks(c, self.audio, 1570)      # odd blocks
        np.testing.assert_array_equal(a.probs, b.probs)
        np.testing.assert_array_equal(a.probs, c.probs)

    def test_reset_restores_initial_state(self):
        a = vad.SileroVad()
        feed_in_blocks(a, self.audio, 960)
        first = list(a.probs)
        a.reset()
        a.probs.clear()
        feed_in_blocks(a, self.audio, 960)
        np.testing.assert_array_equal(first, a.probs)

    def test_silence_low_probability(self):
        v = vad.SileroVad()
        feed_in_blocks(v, np.zeros(16000, dtype=np.float32), 960)
        self.assertLess(max(v.probs), 0.2)


class TestMusicDiscrimination(unittest.TestCase):
    """The empirical gate for swapping webrtcvad -> silero in the chunker:
    clean speech must score high, the music-bed fixture must score low."""

    @classmethod
    def setUpClass(cls):
        cls.speech = asr.load_wav(os.path.join(HERE, "samples", "lang_zh.wav"))
        music_path = os.path.join(HERE, "captures", "zh_radio_180s.wav")
        cls.music = asr.load_wav(music_path) if os.path.isfile(music_path) else None

    def speech_fraction(self, audio):
        v = vad.SileroVad()
        d = feed_in_blocks(v, audio, 960)
        return sum(d) / len(d)

    def test_clean_speech_high_fraction(self):
        frac = self.speech_fraction(self.speech)
        print(f"\nsilero speech fraction lang_zh.wav: {frac:.2f}")
        self.assertGreater(frac, 0.5)

    def test_music_bed_low_fraction(self):
        if self.music is None:
            self.skipTest("captures/zh_radio_180s.wav not present")
        frac = self.speech_fraction(self.music)
        print(f"silero speech fraction zh_radio_180s.wav (music): {frac:.2f}")
        self.assertLess(frac, 0.35)

    def test_chunker_emits_fewer_chunks_on_music(self):
        if self.music is None:
            self.skipTest("captures/zh_radio_180s.wav not present")
        cs = chunker.Chunker(vad=vad.SileroVad())
        cs.feed(self.speech)
        cs.flush()
        cm = chunker.Chunker(vad=vad.SileroVad())
        cm.feed(self.music)
        cm.flush()
        print(f"silero chunker: speech {cs.stats} | music {cm.stats}")
        self.assertGreater(cs.stats["emitted"], cm.stats["emitted"])


if __name__ == "__main__":
    unittest.main()
