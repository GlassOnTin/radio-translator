# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for the live chunker (fake VAD, no GPU, no audio hardware).
Frame i of the synthetic stream has a unique amplitude so tests can assert
exactly which stream frames landed in a chunk (pre-roll, carry, caps)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chunker  # noqa: E402

FRAME_N = 480  # 30 ms @ 16 kHz


class FakeVad:
    """Scripted VAD: frame index in `speech` -> True. Counts calls."""

    def __init__(self, speech):
        self.speech = set(speech)
        self.n = 0

    def is_speech(self, pcm, sr):
        i = self.n
        self.n += 1
        return i in self.speech


def stream(n):
    """n frames of 480 samples; frame i amplitude = 0.001 + 1e-5*i."""
    amps = (0.001 + 1e-5 * np.arange(n)).astype(np.float32)
    return np.repeat(amps, FRAME_N)


def amp_of(audio, frame_idx):
    return float(audio[frame_idx * FRAME_N])


def logic_chunker(speech, **kw):
    """AGC disabled (gain clamped to 1) so frame amplitudes pass through."""
    kw.setdefault("agc_max_gain", 1.0)
    return chunker.Chunker(vad=FakeVad(speech), **kw)


class TestChunkBoundaries(unittest.TestCase):
    def test_silence_only_emits_nothing(self):
        c = logic_chunker([])
        self.assertEqual(c.feed(stream(100)), [])
        self.assertEqual(c.flush(), [])
        self.assertEqual(c.stats["emitted"], 0)

    def test_pause_closes_chunk(self):
        c = logic_chunker(range(10, 90))
        out = c.feed(stream(110))
        self.assertEqual(len(out), 1)
        audio, meta = out[0]
        self.assertEqual(meta["reason"], "pause")
        # 10 pre-roll + 80 speech + 10 trailing silence frames
        self.assertEqual(len(audio), 100 * FRAME_N)
        self.assertAlmostEqual(amp_of(audio, 0), 0.001, places=6)
        self.assertIsNotNone(meta["start"])
        self.assertEqual(c.stats["emitted"], 1)

    def test_preroll_starts_before_speech(self):
        c = logic_chunker(range(20, 100))
        out = c.feed(stream(120))
        audio, _ = out[0]
        # speech at frame 20, pre-roll 10 -> chunk starts at stream frame 10
        self.assertAlmostEqual(amp_of(audio, 0), 0.001 + 1e-5 * 10, places=6)
        self.assertEqual(len(audio), 100 * FRAME_N)

    def test_short_burst_discarded(self):
        c = logic_chunker(range(10, 30))
        out = c.feed(stream(60))
        self.assertEqual(out, [])
        self.assertEqual(c.stats["discarded"], 1)
        self.assertEqual(c.stats["emitted"], 0)

    def test_flush_emits_partial(self):
        c = logic_chunker(range(100))
        self.assertEqual(c.feed(stream(100)), [])
        out = c.flush()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1]["reason"], "end")
        self.assertEqual(len(out[0][0]), 100 * FRAME_N)


class TestCaps(unittest.TestCase):
    def test_hard_cap(self):
        c = logic_chunker(range(400))
        out = c.feed(stream(300))
        self.assertEqual(len(out), 1)
        audio, meta = out[0]
        self.assertEqual(meta["reason"], "maxlen")
        self.assertEqual(len(audio), 250 * FRAME_N)  # 7.5 s / 30 ms
        self.assertEqual(c.stats["maxlen"], 1)
        self.assertEqual(c.flush(), [])  # 50 leftover frames < 2 s minimum
        self.assertEqual(c.stats["discarded"], 1)

    def test_soft_cap_flushes_on_short_pause_near_cap(self):
        # 205 speech frames (age 205 >= soft start 200), then 5 silence frames
        c = logic_chunker(range(205))
        out = c.feed(stream(220))
        self.assertEqual(len(out), 1)
        audio, meta = out[0]
        self.assertEqual(meta["reason"], "softcap")
        self.assertEqual(len(audio), 210 * FRAME_N)
        self.assertEqual(c.stats["softcap"], 1)

    def test_short_pause_early_does_not_close(self):
        speech = list(range(100)) + list(range(105, 150))
        c = logic_chunker(speech)
        self.assertEqual(c.feed(stream(150)), [])  # 5-frame pause at age 105
        out = c.flush()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1]["reason"], "end")
        self.assertEqual(len(out[0][0]), 150 * FRAME_N)


class TestCarry(unittest.TestCase):
    def test_carry_overlaps_last_speech_not_silence(self):
        speech = list(range(10, 90)) + list(range(120, 200))
        c = logic_chunker(speech, carry_s=0.3)  # 10 frames
        out = c.feed(stream(240))
        self.assertEqual(len(out), 2)
        first, second = out[0][0], out[1][0]
        self.assertEqual(len(first), 100 * FRAME_N)
        # chunk 1 last speech = stream frames 80..89 -> carried into chunk 2
        self.assertAlmostEqual(amp_of(second, 0), 0.001 + 1e-5 * 80, places=6)
        self.assertAlmostEqual(amp_of(second, 9), 0.001 + 1e-5 * 89, places=6)
        # then the 10-frame pre-roll ring (stream frames 110..119)
        self.assertAlmostEqual(amp_of(second, 10), 0.001 + 1e-5 * 110, places=6)

    def test_no_carry_by_default(self):
        speech = list(range(10, 90)) + list(range(120, 200))
        c = logic_chunker(speech)
        out = c.feed(stream(240))
        self.assertEqual(len(out), 2)
        # chunk 2 starts at pre-roll (stream frame 110), no carried speech
        self.assertAlmostEqual(amp_of(out[1][0], 0), 0.001 + 1e-5 * 110, places=6)


class TestAgc(unittest.TestCase):
    def test_boosts_quiet_toward_target(self):
        c = chunker.Chunker(vad=FakeVad(range(200)), agc_target=0.1, agc_max_gain=40)
        c.feed(np.full(200 * FRAME_N, 0.005, dtype=np.float32))
        out = c.flush()
        rms = float(np.sqrt(np.mean(out[0][0] ** 2)))
        self.assertGreater(rms, 0.05)
        self.assertLess(rms, 0.15)

    def test_gain_clamped(self):
        c = chunker.Chunker(vad=FakeVad(range(200)), agc_target=0.1, agc_max_gain=40)
        c.feed(np.full(200 * FRAME_N, 1e-5, dtype=np.float32))
        out = c.flush()
        audio = out[0][0]
        rms = float(np.sqrt(np.mean(audio ** 2)))
        self.assertLess(rms, 0.01)  # 40x max gain cannot reach 0.1 from 1e-5
        self.assertLessEqual(float(np.abs(audio).max()), 1.0)

    def test_loud_audio_not_attenuated(self):
        c = chunker.Chunker(vad=FakeVad(range(200)), agc_target=0.1, agc_max_gain=40)
        c.feed(np.full(200 * FRAME_N, 0.5, dtype=np.float32))
        out = c.flush()
        rms = float(np.sqrt(np.mean(out[0][0] ** 2)))
        self.assertAlmostEqual(rms, 0.5, delta=0.01)


class TestFeedBuffering(unittest.TestCase):
    def test_arbitrary_block_lengths_match_frame_by_frame(self):
        speech = range(10, 90)
        a = logic_chunker(speech)
        b = logic_chunker(speech)
        data = stream(110)
        out_a = a.feed(data)
        out_b = []
        pos = 0
        while pos < len(data):
            out_b.extend(b.feed(data[pos:pos + 1570]))  # 3 frames + 130 samples
            pos += 1570
        self.assertEqual(len(out_a), len(out_b))
        np.testing.assert_array_equal(out_a[0][0], out_b[0][0])
        self.assertEqual(out_a[0][1]["reason"], out_b[0][1]["reason"])


class TestRealVadIntegration(unittest.TestCase):
    def test_clean_speech_fixture_produces_chunks(self):
        import asr
        path = os.path.join(asr.HERE, "samples", "lang_zh.wav")
        if not os.path.isfile(path):
            self.skipTest("samples/lang_zh.wav not present")
        audio = asr.load_wav(path)
        c = chunker.Chunker()  # real webrtcvad, default AGC
        out = c.feed(audio)
        out.extend(c.flush())
        self.assertGreaterEqual(len(out), 1)
        for _, meta in out:
            self.assertIn(meta["reason"], ("pause", "softcap", "maxlen", "end"))
            self.assertGreaterEqual(meta["dur"], 2.0)
        print(f"\nlang_zh.wav: {len(audio)/16000:.1f}s -> {len(out)} chunks, stats={c.stats}")


if __name__ == "__main__":
    unittest.main()
