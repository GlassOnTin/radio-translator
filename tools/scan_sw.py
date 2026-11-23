#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scripted shortwave carrier scan: capture IQ on antenna C with sdrplay_cap,
demodulate AM with demod.py, then language-id every clip with one whisper load.
No GUI, no SDRconnect. The SDRplay API is exclusive: close SDRconnect first.

Usage:
  tools/scan_sw.py [--seconds 12] [--grdB 30] [--out-dir captures] FREQ_HZ...
  tools/scan_sw.py 909000 11790000 15325000   # etc.

Prints a table: freq | clip RMS dBFS | detected lang p | English text head.
Clips are kept as <out-dir>/scan_<freq>.wav for fixture selection.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import asr  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("freqs", type=int, nargs="+", help="carrier frequencies in Hz")
ap.add_argument("--seconds", type=float, default=12.0)
ap.add_argument("--grdB", type=int, default=30, help="SDRplay gain reduction dB (lower = more gain)")
ap.add_argument("--antenna", default="C")
ap.add_argument("--out-dir", default=os.path.join(ROOT, "captures"))
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
clips = []
for f in args.freqs:
    iq = os.path.join(args.out_dir, f"scan_{f}.iq")
    wav = os.path.join(args.out_dir, f"scan_{f}.wav")
    t0 = time.time()
    cap = subprocess.run(
        [os.path.join(HERE, "sdrplay_cap"), str(f), "2000000",
         str(args.seconds), iq, args.antenna, str(args.grdB)],
        capture_output=True, text=True)
    if cap.returncode != 0 or not os.path.isfile(iq):
        print(f"{f:>10} Hz  CAPTURE FAILED: {cap.stderr.strip().splitlines()[-1:]}", flush=True)
        continue
    dem = subprocess.run(
        [sys.executable, os.path.join(HERE, "demod.py"), iq, wav,
         "--fs", "2000000", "--mode", "am"],
        capture_output=True, text=True)
    os.remove(iq)  # 48 MB per clip; keep only the WAV
    if dem.returncode != 0:
        print(f"{f:>10} Hz  DEMOD FAILED: {dem.stderr.strip()[:120]}", flush=True)
        continue
    a, sr = sf.read(wav, dtype="float32")
    rms = float(np.sqrt(np.mean(a**2)))
    dbfs = 20 * np.log10(rms) if rms > 0 else -120.0
    clips.append((f, wav, dbfs))
    print(f"{f:>10} Hz  captured+demod {time.time()-t0:.0f}s  RMS {dbfs:.1f} dBFS", flush=True)

if not clips:
    print("no clips captured")
    sys.exit(1)

print(f"\nloading {os.environ.get('WHISPER_MODEL', 'large-v3')} for lang-id...", flush=True)
model = asr.load_whisper()
for f, wav, dbfs in clips:
    audio = asr.load_wav(wav)
    t0 = time.time()
    segs, info = asr.transcribe(model, audio, task="translate", profile="file")
    text = " ".join(s.text.strip() for s in segs).strip()
    print(f"\n### {f} Hz | RMS {dbfs:.1f} dBFS | lang={info.language} "
          f"p={info.language_probability:.2f} | infer {time.time()-t0:.1f}s")
    print(text[:400] if text else "(no speech detected)")
