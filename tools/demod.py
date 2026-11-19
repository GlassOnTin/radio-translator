#!/usr/bin/env python3
"""Demodulate captured SDR IQ (raw int16 I/Q) to a 16 kHz mono WAV for whisper.
numpy-only (no scipy). AM envelope detection; SSB via BFO frequency shift.

Usage:
  demod.py in.iq out.wav --fs 2000000 --mode am [--audio-bw 5000]
  demod.py in.iq out.wav --fs 2000000 --mode usb --bfo 0
  demod.py in.iq out.wav --fs 2000000 --mode lsb --bfo 0

For HF shortwave broadcast (AM) tune the receiver to the carrier; AM needs no
BFO. For SSB, tune to the suppressed carrier and set --bfo to the residual
offset (Hz); usb/lsb select the sideband. Output is 16 kHz mono float WAV,
ready for translate_file.py / live.py.
"""
import argparse
import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("iq")
ap.add_argument("wav")
ap.add_argument("--fs", type=float, default=2_000_000.0, help="IQ sample rate Hz")
ap.add_argument("--mode", choices=["am", "usb", "lsb"], default="am")
ap.add_argument("--bfo", type=float, default=0.0, help="SSB BFO offset Hz")
ap.add_argument("--audio-bw", type=float, default=5000.0, help="audio low-pass Hz")
ap.add_argument("--out-sr", type=int, default=16000)
args = ap.parse_args()

d = np.fromfile(args.iq, dtype=np.int16)
iq = d[0::2].astype(np.float64) + 1j * d[1::2].astype(np.float64)
fs = args.fs
n = len(iq)

if args.mode == "am":
    sig = np.abs(iq)                      # envelope
    sig = sig - sig.mean()                # remove DC (carrier)
else:
    t = np.arange(n) / fs
    shift = np.exp(-2j * np.pi * args.bfo * t)
    if args.mode == "lsb":
        sig = np.real(np.conj(iq) * shift)
    else:
        sig = np.real(iq * shift)

# FFT low-pass to audio bandwidth (anti-alias before decimation)
spec = np.fft.rfft(sig)
freqs = np.fft.rfftfreq(n, 1 / fs)
spec[freqs > args.audio_bw] = 0
sig = np.fft.irfft(spec, n)

# resample to output rate (linear interp; signal already band-limited)
n_out = int(n * args.out_sr / fs)
t_in = np.arange(n) / fs
t_out = np.arange(n_out) / args.out_sr
audio = np.interp(t_out, t_in, sig)

peak = np.abs(audio).max()
if peak > 0:
    audio = audio / peak * 0.9
sf.write(args.wav, audio.astype(np.float32), args.out_sr)
print(f"{args.mode} demod: {n} IQ samples @ {fs/1e6:.2f} MS/s -> "
      f"{n_out/args.out_sr:.1f}s @ {args.out_sr} Hz -> {args.wav}")