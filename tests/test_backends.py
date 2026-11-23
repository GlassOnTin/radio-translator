# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for translation backends (no GPU, no network)."""
import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backends  # noqa: E402


class FakeWhisper:
    def __init__(self, lang="es", text="hola mundo"):
        self.lang, self.text = lang, text
        self.transcribe_calls = []

    def detect_language(self, audio, **kw):
        return self.lang, 0.99, [(self.lang, 0.99)]

    def transcribe(self, audio, **kw):
        self.transcribe_calls.append(kw)
        seg = types.SimpleNamespace(text=self.text, avg_logprob=-0.2,
                                    compression_ratio=1.5, no_speech_prob=0.1)
        info = types.SimpleNamespace(language=self.lang, language_probability=0.99)
        return [seg], info


class FakeCanary:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append(kw.get("language"))
        seg = backends.make_segment("hello world")
        info = types.SimpleNamespace(language=kw.get("language"), language_probability=1.0)
        return [seg], info


class TestBackends(unittest.TestCase):
    def test_make_segment_metadata(self):
        s = backends.make_segment("hello world")
        self.assertEqual(s.text, "hello world")
        self.assertGreater(s.compression_ratio, 0)
        self.assertEqual(s.no_speech_prob, 0.1)

    def test_resample_ratio_length(self):
        x = np.zeros(16000, dtype=np.float32)
        y = backends.resample_ratio(x, 3, 2)  # 16k -> 24k
        self.assertEqual(len(y), 24000)

    def test_whisper_mt_uses_transcribe_then_mt(self):
        w = FakeWhisper(lang="es", text="hola mundo")
        mt = backends.WhisperMTModel(w, mt_model="x")
        mt.mt_ollama = lambda *a, **k: "hello world"  # patch network call
        # patch the module-level function the method calls
        orig = backends.mt_ollama
        backends.mt_ollama = lambda text, lang, model, host: "hello world"
        try:
            segs, info = mt.transcribe(np.zeros(16000, dtype=np.float32))
        finally:
            backends.mt_ollama = orig
        self.assertEqual(segs[0].text, "hello world")
        self.assertEqual(info.language, "es")  # source lang preserved
        self.assertEqual(w.transcribe_calls[0]["task"], "transcribe")

    def test_canary_router_european_to_canary(self):
        w = FakeWhisper(lang="es")
        c = FakeCanary()
        r = backends.CanaryRouter(w, c)
        segs, info = r.transcribe(np.zeros(16000, dtype=np.float32))
        self.assertEqual(c.calls, ["es"])
        self.assertEqual(segs[0].text, "hello world")
        self.assertEqual(w.transcribe_calls, [])  # whisper not used for European

    def test_canary_router_non_european_fallback(self):
        w = FakeWhisper(lang="zh", text="你好")
        c = FakeCanary()
        r = backends.CanaryRouter(w, c)
        segs, info = r.transcribe(np.zeros(16000, dtype=np.float32))
        self.assertEqual(c.calls, [])  # canary skipped
        self.assertEqual(w.transcribe_calls[0]["task"], "translate")
        self.assertEqual(info.language, "zh")


if __name__ == "__main__":
    unittest.main()
