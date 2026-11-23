# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for the shared Translator core (fake model, no GPU)."""
import os
import sys
import time
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from translator import Translator  # noqa: E402


class FakeModel:
    def __init__(self, text="hello world", lang="fr", comp=1.5, nospeech=0.1):
        self.text, self.lang, self.comp, self.nospeech = text, lang, comp, nospeech
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        seg = types.SimpleNamespace(text=self.text, avg_logprob=-0.2,
                                    compression_ratio=self.comp, no_speech_prob=self.nospeech)
        info = types.SimpleNamespace(language=self.lang, language_probability=0.99)
        return [seg], info


def meta(dur=3.0, reason="pause"):
    return dict(dur=dur, reason=reason, start=time.time() - 1.0)


class TestTranslator(unittest.TestCase):
    def test_clean_chunk_event(self):
        t = Translator(FakeModel("a clean english sentence", lang="fr"))
        ev = t.process_chunk(None, meta())
        self.assertEqual(ev["type"], "chunk")
        self.assertEqual(ev["chunk"], 1)
        self.assertEqual(ev["language"], "fr")
        self.assertEqual(ev["text"], "a clean english sentence")
        self.assertFalse(ev["suppressed"])
        self.assertEqual(ev["reasons"], [])
        self.assertEqual(ev["reason"], "pause")
        self.assertGreaterEqual(ev["latency_s"], 0.0)

    def test_empty_text_returns_none(self):
        t = Translator(FakeModel("   "))
        self.assertIsNone(t.process_chunk(None, meta()))

    def test_repetition_suppressed(self):
        t = Translator(FakeModel("go go go go"))
        ev = t.process_chunk(None, meta())
        self.assertTrue(ev["suppressed"])
        self.assertIn("repetition", ev["reasons"])
        self.assertEqual(t.suppressed, 1)

    def test_cjk_echo_suppressed(self):
        t = Translator(FakeModel("你好世界测试中文"))
        ev = t.process_chunk(None, meta())
        self.assertTrue(ev["suppressed"])
        self.assertIn("cjk-echo", ev["reasons"])

    def test_compression_suppressed(self):
        t = Translator(FakeModel("some text", comp=3.0))
        ev = t.process_chunk(None, meta())
        self.assertTrue(ev["suppressed"])
        self.assertIn("compression", ev["reasons"])

    def test_no_speech_suppressed(self):
        t = Translator(FakeModel("thanks for watching", nospeech=0.9))
        ev = t.process_chunk(None, meta())
        self.assertTrue(ev["suppressed"])
        self.assertIn("no-speech", ev["reasons"])

    def test_gates_off_passes_repetition(self):
        t = Translator(FakeModel("go go go go"), gates=False)
        ev = t.process_chunk(None, meta())
        self.assertFalse(ev["suppressed"])

    def test_idx_increments(self):
        t = Translator(FakeModel("hello"))
        self.assertEqual(t.process_chunk(None, meta())["chunk"], 1)
        self.assertEqual(t.process_chunk(None, meta())["chunk"], 2)

    def test_carry_dedup(self):
        t = Translator(FakeModel("the quick brown fox jumps over"), carry=0.3)
        t.process_chunk(None, meta())  # sets prev_text
        t.model.text = "brown fox jumps over the lazy dog"
        ev = t.process_chunk(None, meta())
        self.assertEqual(ev["text"], "the lazy dog")  # 4-word overlap trimmed

    def test_sticky_language_mode(self):
        # zh dominates; a single noisy 'en' burst must not flip the report
        m = FakeModel("hello", lang="zh")
        t = Translator(m, lang_window=5)
        for _ in range(3):
            t.process_chunk(None, meta())
        m.lang = "en"
        ev = t.process_chunk(None, meta())
        self.assertEqual(ev["language"], "zh")      # sticky mode
        self.assertEqual(ev["language_raw"], "en")  # per-chunk still reported
        m.lang = "zh"
        t.process_chunk(None, meta())
        self.assertEqual(t.sticky_lang, "zh")

    def test_live_profile_and_ngram_override(self):
        m = FakeModel("hello")
        t = Translator(m)
        t.process_chunk(None, meta())
        # asr.transcribe consumes `profile`; the model sees the live kwargs
        self.assertEqual(m.kwargs["task"], "translate")
        self.assertEqual(m.kwargs["beam_size"], 1)
        self.assertFalse(m.kwargs["vad_filter"])
        self.assertEqual(m.kwargs["no_repeat_ngram_size"], 4)


if __name__ == "__main__":
    unittest.main()
