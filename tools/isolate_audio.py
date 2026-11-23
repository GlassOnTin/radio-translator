#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Voice-isolation enhancement stage (inworld-ai/voice-isolator, Apache-2.0
code and weights). Band-split dual-path network, 48 kHz internal; the library
resamples to/from the input rate. Runs under .venv-tts (torch), separate from
the lean .venv used by live.py.

Usage:
  .venv-tts/bin/python tools/isolate_audio.py IN.wav OUT.wav [--mode offline|streaming]
"""
import argparse
import sys
import time

import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("inp")
ap.add_argument("out")
ap.add_argument("--mode", default="offline", choices=["offline", "streaming"])
ap.add_argument("--device", default="cuda")
ap.add_argument("--lookahead", type=int, default=2, help="streaming lookahead frames (20 ms each)")
args = ap.parse_args()

from inworld import voice_isolation  # noqa: E402

audio, sr = sf.read(args.inp, dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)

t0 = time.time()
iso = voice_isolation.VoiceIsolator(
    "inworld/voice-isolator", device=args.device, mode=args.mode,
    lookahead_frames=args.lookahead if args.mode == "streaming" else 0)
t_load = time.time() - t0

t1 = time.time()
voice = iso.isolate(audio, sr, output_sample_rate=sr)
t_inf = time.time() - t1

if hasattr(voice, "cpu"):
    voice = voice.cpu().numpy()
voice = np.asarray(voice, dtype=np.float32)
if voice.ndim > 1:
    voice = voice.mean(axis=1)
sf.write(args.out, voice, sr)

dur = len(audio) / sr
rms_in = float(np.sqrt(np.mean(audio**2)))
rms_out = float(np.sqrt(np.mean(voice**2)))
print(f"{args.inp}: {dur:.1f}s | load {t_load:.1f}s | isolate {t_inf:.1f}s "
      f"RTF {t_inf/dur:.3f} | mode={args.mode} dev={args.device} | "
      f"RMS {20*np.log10(rms_in+1e-12):.1f} -> {20*np.log10(rms_out+1e-12):.1f} dBFS | "
      f"out {len(voice)/sr:.1f}s", file=sys.stderr)
