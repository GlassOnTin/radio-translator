# SPDX-License-Identifier: AGPL-3.0-or-later
"""Block-wise AM/SSB demodulator core, numpy-only (no scipy).

2 MS/s IQ -> 16 kHz mono float32 audio, in arbitrary block sizes:
  - single-stage polyphase direct-form decimation by fs/out_sr (125),
    4096-tap Hamming windowed-sinc low-pass at audio_bw (2048 taps would
    alias at the transition width we need),
  - AM envelope |iq|, or SSB via NCO shift with phase continuous across blocks,
  - carrier DC removed by an EMA (tau 0.3 s) at the output rate,
  - AGC: RMS EMA (attack 0.1 s / release 3 s), gain clamped [0.1, 20],
    tanh soft-clip knee at 0.95 with a clip counter.

DC/AGC run per output sample (16 kHz -> a few % CPU live) so the result is
independent of the input block size; the FIR keeps a ntaps-1 tail and the
decimation phase across calls. demod_stream.py wraps this for pipes; the
equivalence tests compare block-wise output against a whole-file call.
"""
import math

import numpy as np


def design_fir(fs, cutoff, ntaps=4096):
    """Unity-DC-gain Hamming windowed-sinc low-pass, float32, symmetric."""
    n = np.arange(ntaps) - (ntaps - 1) / 2
    h = np.sinc(2 * cutoff / fs * n) * np.hamming(ntaps)
    return (h / h.sum()).astype(np.float32)


class DemodCore:
    def __init__(self, fs=2_000_000.0, out_sr=16000, mode="am", bfo=0.0,
                 audio_bw=5000.0, ntaps=4096, dc_tau=0.3, agc_target=0.1,
                 agc_attack=0.1, agc_release=3.0, agc_min=0.1, agc_max=20.0,
                 soft_clip=0.95, agc=True):
        assert fs % out_sr == 0, "fs must be an integer multiple of out_sr"
        self.fs, self.out_sr, self.mode, self.bfo = fs, out_sr, mode, bfo
        self.decim = int(fs // out_sr)
        self.ntaps = ntaps
        self.fir = design_fir(fs, audio_bw, ntaps)
        self.tail = np.zeros(ntaps - 1, dtype=np.float32)
        self.pos = 0                 # global input sample count (decim phase)
        self.nco_phase = 0.0
        self.dc = 0.0
        self.rms = 0.0
        self.gain = 1.0
        self.agc = agc
        self.agc_target, self.agc_min, self.agc_max = agc_target, agc_min, agc_max
        self.soft_clip = soft_clip
        self._a_dc = math.exp(-1.0 / (out_sr * dc_tau))
        self._a_att = math.exp(-1.0 / (out_sr * agc_attack))
        self._a_rel = math.exp(-1.0 / (out_sr * agc_release))
        self.clip_count = 0
        self.gain_min_seen, self.gain_max_seen = 1.0, 1.0

    def _baseband(self, iq):
        if self.mode == "am":
            return np.abs(iq).astype(np.float32)
        n = len(iq)
        ph = self.nco_phase + 2 * np.pi * self.bfo * np.arange(n) / self.fs
        self.nco_phase = (self.nco_phase + 2 * np.pi * self.bfo * n / self.fs) % (2 * np.pi)
        shift = np.exp(-1j * ph)
        if self.mode == "lsb":
            return np.real(np.conj(iq) * shift).astype(np.float32)
        return np.real(iq * shift).astype(np.float32)

    def _decimate(self, sig):
        ext = np.concatenate([self.tail, sig])
        m0 = (-self.pos) % self.decim
        win = np.lib.stride_tricks.sliding_window_view(ext, self.ntaps)[m0::self.decim]
        if len(win) == 0:
            y = np.zeros(0, dtype=np.float32)
        else:
            y = np.concatenate([win[i:i + 1024] @ self.fir
                                for i in range(0, len(win), 1024)]).astype(np.float32)
        self.tail = ext[-(self.ntaps - 1):].copy()
        self.pos += len(sig)
        return y

    def _condition(self, y):
        """DC removal + AGC + soft clip, per output sample (block-invariant)."""
        out = np.empty(len(y), dtype=np.float32)
        a_dc, a_att, a_rel = self._a_dc, self._a_att, self._a_rel
        dc, rms, gain = self.dc, self.rms, self.gain
        tgt, gmin, gmax, knee = self.agc_target, self.agc_min, self.agc_max, self.soft_clip
        clips = 0
        for i in range(len(y)):
            v = float(y[i])
            dc = a_dc * dc + (1 - a_dc) * v
            x = v - dc
            ax = abs(x)
            a = a_att if ax > rms else a_rel
            rms = a * rms + (1 - a) * ax
            if self.agc:
                desired = tgt / rms if rms > 1e-9 else gmax
                gain = min(max(desired, gmin), gmax)
            z = x * gain
            az = abs(z)
            if az > knee:
                clips += 1
                z = math.copysign(knee + (1 - knee) * math.tanh((az - knee) / (1 - knee)), z)
            out[i] = z
        self.dc, self.rms, self.gain = dc, rms, gain
        self.clip_count += clips
        if len(y):
            self.gain_min_seen = min(self.gain_min_seen, gain)
            self.gain_max_seen = max(self.gain_max_seen, gain)
        return out

    def process(self, iq):
        """iq: complex samples (any block length) -> float32 audio @ out_sr."""
        iq = np.asarray(iq, dtype=np.complex64)
        if len(iq) == 0:
            return np.zeros(0, dtype=np.float32)
        return self._condition(self._decimate(self._baseband(iq)))
