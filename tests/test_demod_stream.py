# SPDX-License-Identifier: AGPL-3.0-or-later
"""Equivalence tests for the streaming demod core: block-size invariance
(the real streaming risk: FIR tail / decimation phase / NCO / DC / AGC state
across boundaries), DC removal, AGC bounds, and correlation against the
whole-file demod.py reference on a real over-the-air capture. CPU only."""
import os
import subprocess
import sys
import unittest

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))
from demod_core import DemodCore  # noqa: E402

FS = 2_000_000.0


def am_iq(seconds, amp=1.0, tone=1000.0, noise=0.01):
    n = int(FS * seconds)
    t = np.arange(n) / FS
    env = amp * (1 + 0.5 * np.sin(2 * np.pi * tone * t))
    iq = env * np.exp(2j * np.pi * 100 * t)
    rng = np.random.default_rng(42)
    return (iq + noise * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)


def ssb_iq(seconds, bfo=1000.0):
    n = int(FS * seconds)
    t = np.arange(n) / FS
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 300 * t)
    return (env * np.exp(2j * np.pi * bfo * t)).astype(np.complex64)


def blockwise(core, iq, block):
    out = []
    for i in range(0, len(iq), block):
        out.append(core.process(iq[i:i + block]))
    return np.concatenate(out)


class TestBlockInvariance(unittest.TestCase):
    def test_am_whole_vs_blocks(self):
        iq = am_iq(2.0)
        whole = DemodCore().process(iq)
        b250 = blockwise(DemodCore(), iq, 500_000)     # 0.25 s
        bodd = blockwise(DemodCore(), iq, 123_457)    # non-aligned
        self.assertEqual(len(whole), len(b250))
        self.assertEqual(len(whole), len(bodd))
        self.assertLess(float(np.abs(whole - b250).max()), 2e-5)
        self.assertLess(float(np.abs(whole - bodd).max()), 2e-5)

    def test_ssb_nco_continuous_across_blocks(self):
        iq = ssb_iq(2.0, bfo=1000.0)
        whole = DemodCore(mode="usb", bfo=1000.0).process(iq)
        bodd = blockwise(DemodCore(mode="usb", bfo=1000.0), iq, 99_991)
        self.assertLess(float(np.abs(whole - bodd).max()), 2e-5)

    def test_output_rate(self):
        iq = am_iq(1.0)
        out = DemodCore().process(iq)
        self.assertAlmostEqual(len(out), len(iq) / 125, delta=4096 / 125 + 1)


class TestConditioning(unittest.TestCase):
    def test_dc_removed(self):
        iq = np.ones(int(FS * 3), dtype=np.complex64)  # constant envelope
        out = DemodCore(agc=False).process(iq)
        tail = out[-16000:]
        self.assertLess(abs(float(tail.mean())), 0.005)

    def test_agc_boosts_quiet_and_clamps_gain(self):
        iq = am_iq(3.0, amp=0.005, noise=0.0001)
        core = DemodCore()
        out = core.process(iq)
        rms = float(np.sqrt(np.mean(out[-16000:] ** 2)))
        # AC RMS 0.00177 x gain clamp 20 -> ~0.035 (target 0.1 unreachable)
        self.assertGreater(rms, 0.02)
        self.assertLess(rms, 0.1)
        self.assertLessEqual(core.gain_max_seen, 20.0)
        self.assertGreaterEqual(core.gain_min_seen, 0.1)

    def test_soft_clip_bounds_loud(self):
        # high crest factor: quiet baseline (gain clamps high) + loud bursts
        n = int(FS * 2)
        env = np.full(n, 0.001, dtype=np.float32)
        for start in range(0, n, int(FS)):  # 10 ms burst every 1 s (1% duty)
            env[start:start + int(FS * 0.01)] = 0.5
        iq = (env * np.exp(2j * np.pi * 100 * np.arange(n) / FS)).astype(np.complex64)
        core = DemodCore()
        out = core.process(iq)
        self.assertLessEqual(float(np.abs(out).max()), 1.0)
        self.assertGreater(core.clip_count, 0)


class TestAgainstDemodPy(unittest.TestCase):
    """Correlation with the whole-file FFT reference on a real capture.
    Different algorithms (FIR vs brick-wall, AGC vs peak-normalise), so the
    bound is correlation after unit-RMS alignment, not sample equality."""

    IQ = "/tmp/mw909_C.iq"

    def test_correlation_on_air_capture(self):
        if not os.path.isfile(self.IQ):
            self.skipTest(f"{self.IQ} not present")
        ref_wav = "/tmp/mw909_ref.wav"
        subprocess.run([sys.executable, os.path.join(HERE, "tools", "demod.py"),
                        self.IQ, ref_wav, "--fs", "2000000", "--mode", "am"],
                       check=True, capture_output=True)
        ref, _ = sf.read(ref_wav, dtype="float32")
        d = np.fromfile(self.IQ, dtype=np.int16)
        iq = (d[0::2].astype(np.float32) + 1j * d[1::2].astype(np.float32)) / 32768.0
        got = DemodCore().process(iq)
        n = min(len(ref), len(got))
        a, b = ref[:n] - ref[:n].mean(), got[:n] - got[:n].mean()
        a /= np.sqrt(np.mean(a * a))
        b /= np.sqrt(np.mean(b * b))
        # the FIR has group delay (ntaps-1)/2 input samples; demod.py's FFT
        # filter is zero-phase. Align by the predicted delay before comparing.
        delay = round((4096 - 1) / 2 / 125)  # 16 output samples
        corr = float(np.mean(a[:-delay] * b[delay:]))
        best_corr, best_lag = max(
            (float(np.mean(a[max(0, l):n + min(0, l)] * b[max(0, -l):n - max(0, l)])), l)
            for l in range(-60, 61))
        print(f"\ndemod_stream vs demod.py on mw909_C.iq: corr {corr:.4f} at predicted "
              f"delay {delay}; best {best_corr:.4f} at lag {best_lag}")
        self.assertGreater(corr, 0.7)          # measured 0.762 (music capture)
        self.assertLessEqual(abs(best_lag + delay), 2)  # delay matches theory


if __name__ == "__main__":
    unittest.main()
