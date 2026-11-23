# SPDX-License-Identifier: AGPL-3.0-or-later
"""Streaming VAD adapters for the live chunker.

SileroVad wraps the silero v6 ONNX model bundled with faster-whisper (no extra
dependency) in the webrtcvad-compatible is_speech(pcm_bytes, sr) interface, so
chunker.Chunker(vad=SileroVad()) is a drop-in swap. Unlike webrtcvad, silero
rejects music beds — the main source of fluent whisper hallucinations on SW
broadcasts (measured on captures/zh_radio_180s.wav and fr_radio_180s.wav).

Silero v6 consumes 512-sample windows (32 ms @ 16 kHz) with 64 samples of
context and persistent LSTM state (h, c). The chunker feeds 30 ms frames, so
the adapter buffers across calls and answers with the latest window probability.
"""
import numpy as np

WINDOW = 512
CONTEXT = 64


class SileroVad:
    def __init__(self, threshold=0.5, sr=16000):
        from faster_whisper.vad import get_vad_model
        assert sr == 16000, "silero v6 bundled model expects 16 kHz"
        self.model = get_vad_model()
        self.threshold = threshold
        self._buf = np.zeros(0, dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._last_prob = 0.0
        self.probs = []  # every window probability, for diagnostics/tests

    def _window(self, win):
        x = np.concatenate([self._context, win[None, :]], axis=1)
        out, self._h, self._c = self.model.session.run(
            None, {"input": x, "h": self._h, "c": self._c})
        self._context = win[-CONTEXT:][None, :]
        return float(out[0])

    def is_speech(self, pcm, sr):
        """pcm: s16le mono bytes (any length); returns bool for the latest
        completed 512-sample window (previous decision until one completes)."""
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, x])
        while len(self._buf) >= WINDOW:
            win, self._buf = self._buf[:WINDOW], self._buf[WINDOW:]
            self._last_prob = self._window(win)
            self.probs.append(self._last_prob)
        return self._last_prob >= self.threshold

    def reset(self):
        self._buf = np.zeros(0, dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._last_prob = 0.0
