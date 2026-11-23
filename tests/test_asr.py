# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for the shared asr helpers (no GPU, no model downloads)."""
import os
import sys
import tempfile
import types
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asr  # noqa: E402


class FakeModel:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        seg = types.SimpleNamespace(text="hello", avg_logprob=-0.2,
                                    compression_ratio=1.5, no_speech_prob=0.1)
        info = types.SimpleNamespace(language="en", language_probability=0.99)
        return [seg], info


def seg(logprob=-0.2, comp=1.5, nospeech=0.1):
    return types.SimpleNamespace(text="x", avg_logprob=logprob,
                                 compression_ratio=comp, no_speech_prob=nospeech)


class TestProfiles(unittest.TestCase):
    def test_live_profile(self):
        p = asr.PROFILES["live"]
        self.assertEqual(p["beam_size"], 1)
        self.assertFalse(p["vad_filter"])
        self.assertFalse(p["condition_on_previous_text"])

    def test_file_profile(self):
        p = asr.PROFILES["file"]
        self.assertEqual(p["beam_size"], 5)
        self.assertTrue(p["vad_filter"])
        self.assertFalse(p["condition_on_previous_text"])

    def test_transcribe_kwargs_live_pinned(self):
        m = FakeModel()
        asr.transcribe(m, np.zeros(1600, np.float32), task="translate",
                       lang="zh", profile="live", no_repeat_ngram_size=3)
        k = m.kwargs
        self.assertEqual(k["beam_size"], 1)
        self.assertFalse(k["vad_filter"])
        self.assertEqual(k["language"], "zh")
        self.assertEqual(k["task"], "translate")
        self.assertEqual(k["no_repeat_ngram_size"], 3)
        self.assertFalse(k["condition_on_previous_text"])

    def test_transcribe_auto_lang_omits_language(self):
        m = FakeModel()
        asr.transcribe(m, np.zeros(1600, np.float32), profile="file")
        self.assertNotIn("language", m.kwargs)
        self.assertEqual(m.kwargs["beam_size"], 5)


class TestGates(unittest.TestCase):
    def test_segment_gates_thresholds(self):
        flags = asr.segment_gates([seg(), seg(logprob=-1.5), seg(comp=3.0),
                                   seg(nospeech=0.8)])
        self.assertEqual(flags["low_logprob"], 1)
        self.assertEqual(flags["high_compression"], 1)
        self.assertEqual(flags["no_speech"], 1)

    def test_segment_gates_clean(self):
        flags = asr.segment_gates([seg(), seg()])
        self.assertEqual(sum(flags.values()), 0)

    def test_repetition_4gram_three_times(self):
        text = "the company is a global leader the company is a global leader " \
               "the company is a global leader"
        f = asr.repetition_flags(text)
        self.assertTrue(f["rep4"])
        self.assertGreater(f["ratio"], 0.0)

    def test_repetition_clean(self):
        f = asr.repetition_flags("China-US relations mean stability for the world economy")
        self.assertFalse(f["rep4"])
        self.assertEqual(f["max_run"], 1)

    def test_repetition_token_run(self):
        f = asr.repetition_flags("and and and and then")
        self.assertTrue(f["run4"])
        self.assertEqual(f["max_run"], 4)

    def test_cjk_ratio(self):
        self.assertEqual(asr.cjk_ratio("你好世界"), 1.0)
        self.assertEqual(asr.cjk_ratio("hello world"), 0.0)
        self.assertAlmostEqual(asr.cjk_ratio("你好 hello"), 2 / 7)

    def test_dedup_overlap_trims_four_words(self):
        prev = "one two three four five"
        new = "two three four five six"
        self.assertEqual(asr.dedup_overlap(prev, new), "six")

    def test_dedup_overlap_below_minimum_untouched(self):
        prev = "one two three four five"
        new = "three four five six"
        self.assertEqual(asr.dedup_overlap(prev, new), new)

    def test_dedup_overlap_empty_prev(self):
        self.assertEqual(asr.dedup_overlap("", "a b c"), "a b c")


class TestDecode(unittest.TestCase):
    def test_decode_audio_roundtrip(self):
        t = np.arange(16000) / 16000
        wav = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "tone.wav")
            sf.write(p, wav, 16000)
            out = asr.decode_audio(p)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(len(out), 16000)
        self.assertTrue(np.allclose(out, wav, atol=2 / 32768))

    def test_load_wav_stereo_means(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "st.wav")
            sf.write(p, np.stack([np.ones(1600, np.float32),
                                  np.zeros(1600, np.float32)], axis=1), 16000)
            out = asr.load_wav(p)
        self.assertEqual(out.ndim, 1)
        self.assertTrue(np.allclose(out, 0.5, atol=2 / 32768))

    def test_load_wav_wrong_rate_asserts(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.wav")
            sf.write(p, np.zeros(8000, np.float32), 8000)
            with self.assertRaises(AssertionError):
                asr.load_wav(p)


if __name__ == "__main__":
    unittest.main()
