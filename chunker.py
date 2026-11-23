# SPDX-License-Identifier: AGPL-3.0-or-later
"""VAD-gated speech chunker for the live translation path.

Cuts a 16 kHz mono float32 stream into chunks at pauses, with:
  - pre-VAD AGC (rolling-RMS boost toward agc_target; monitor taps arrive
    ~-40 dBFS and webrtcvad is level-sensitive),
  - pre-roll ring so chunk starts aren't clipped,
  - soft-cap: within soft_window_s of the cap, a shorter pause closes the
    chunk, avoiding mid-word cuts at the hard cap,
  - optional carry-over audio into the next chunk (dedup the text overlap
    with asr.dedup_overlap),
  - minimum chunk duration (short bursts are discarded).

Pure logic, no I/O: feed() takes arbitrary-length float32 arrays and returns
completed (audio, meta) pairs. The VAD is injectable for tests; production
uses webrtcvad.Vad(2) on 30 ms s16le frames.
"""
import math
import time
from collections import deque

import numpy as np
import webrtcvad

SR = 16000
FRAME_MS = 30


class Chunker:
    def __init__(self, vad=None, sr=SR, frame_ms=FRAME_MS, silence_ms=300,
                 cap_s=7.5, soft_pause_ms=150, soft_window_s=1.5,
                 preroll_ms=300, min_chunk_s=2.0, carry_s=0.0,
                 agc_target=0.1, agc_max_gain=40.0, agc_tau_s=0.3):
        self.vad = vad if vad is not None else webrtcvad.Vad(2)
        self.sr = sr
        self.frame_ms = frame_ms
        self.frame_n = sr * frame_ms // 1000
        self.silence_frames = math.ceil(silence_ms / frame_ms)
        self.cap_frames = round(cap_s * 1000 / frame_ms)
        self.soft_pause_frames = math.ceil(soft_pause_ms / frame_ms)
        self.soft_start_frames = self.cap_frames - round(soft_window_s * 1000 / frame_ms)
        self.min_frames = round(min_chunk_s * 1000 / frame_ms)
        self.carry_frames = round(carry_s * 1000 / frame_ms)
        self.agc_target = agc_target
        self.agc_max_gain = agc_max_gain
        self._agc_alpha = 1.0 - math.exp(-(frame_ms / 1000) / agc_tau_s)

        self._buf = np.zeros(0, dtype=np.float32)
        self._ring = deque(maxlen=round(preroll_ms / frame_ms))
        self._carry = []
        self._frames = []
        self._in_speech = False
        self._silence_run = 0
        self._last_speech_idx = -1
        self._start_wall = None
        self._power_ema = 0.0
        self._gain = 1.0
        self.stats = dict(emitted=0, discarded=0, maxlen=0, softcap=0)

    def _agc(self, x):
        """Boost quiet frames toward agc_target RMS; never attenuate."""
        power = float(np.mean(x * x))
        a = self._agc_alpha
        self._power_ema = (1 - a) * self._power_ema + a * power
        rms = math.sqrt(self._power_ema)
        desired = self.agc_target / rms if rms > 1e-9 else self.agc_max_gain
        desired = min(max(desired, 1.0), self.agc_max_gain)
        self._gain = (1 - a) * self._gain + a * desired
        return np.clip(x * self._gain, -1.0, 1.0)

    def _close(self, reason):
        frames, started = self._frames, self._start_wall
        last_speech = self._last_speech_idx
        self._frames, self._in_speech = [], False
        self._silence_run, self._start_wall = 0, None
        self._last_speech_idx = -1
        if len(frames) < self.min_frames:
            self.stats["discarded"] += 1
            return []
        audio = np.concatenate(frames)
        if self.carry_frames and last_speech >= 0:
            # overlap the next chunk with the tail of this chunk's SPEECH
            # (not its trailing silence); asr.dedup_overlap trims the text.
            self._carry = frames[max(0, last_speech + 1 - self.carry_frames):last_speech + 1]
        self.stats["emitted"] += 1
        if reason in ("maxlen", "softcap"):
            self.stats[reason] += 1
        return [(audio, dict(start=started, reason=reason, dur=len(audio) / self.sr))]

    def _frame(self, x):
        y = self._agc(x)
        pcm = (y * 32767).astype(np.int16).tobytes()
        speech = self.vad.is_speech(pcm, self.sr)
        out = []
        if not self._in_speech:
            if speech:
                self._frames = self._carry + list(self._ring) + [y]
                self._carry = []
                self._ring.clear()
                self._in_speech = True
                self._silence_run = 0
                self._last_speech_idx = len(self._frames) - 1
                self._start_wall = time.time()
            else:
                self._ring.append(y)
        else:
            self._frames.append(y)
            self._silence_run = 0 if speech else self._silence_run + 1
            if speech:
                self._last_speech_idx = len(self._frames) - 1
            age = len(self._frames)
            if self._silence_run >= self.silence_frames:
                out = self._close("pause")
            elif age >= self.soft_start_frames and self._silence_run >= self.soft_pause_frames:
                out = self._close("softcap")
            elif age >= self.cap_frames:
                out = self._close("maxlen")
        return out

    def feed(self, samples):
        """Feed float32 mono samples; return list of completed (audio, meta)."""
        self._buf = np.concatenate([self._buf, np.asarray(samples, dtype=np.float32)])
        out = []
        while len(self._buf) >= self.frame_n:
            frame, self._buf = self._buf[:self.frame_n], self._buf[self.frame_n:]
            out.extend(self._frame(frame))
        return out

    def flush(self):
        """Force-close any open chunk at end of stream."""
        if self._in_speech:
            return self._close("end")
        return []
