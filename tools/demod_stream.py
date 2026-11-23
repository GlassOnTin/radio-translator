#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Streaming AM/SSB demodulator: raw s16 interleaved IQ in, 16 kHz mono audio out.

Pipe mode (default): reads IQ from stdin in --block-s blocks, writes s16le
16 kHz mono to stdout, stats to stderr on exit. SIGINT/SIGPIPE stop cleanly
(upstream sdrplay_cap also tears down on SIGPIPE, freeing the RSP).

  tools/sdrplay_cap 7205000 2000000 0 - C | \\
      tools/demod_stream.py --fs 2000000 --mode am | ./run.sh live.py --stdin

File mode: demod_stream.py in.iq --wav out.wav  (whole-file, same core).
"""
import argparse
import os
import signal
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demod_core import DemodCore  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("iq", nargs="?", default="-", help="IQ file, or - for stdin")
ap.add_argument("--fs", type=float, default=2_000_000.0)
ap.add_argument("--mode", choices=["am", "usb", "lsb"], default="am")
ap.add_argument("--bfo", type=float, default=0.0, help="SSB BFO offset Hz")
ap.add_argument("--audio-bw", type=float, default=5000.0)
ap.add_argument("--block-s", type=float, default=0.25)
ap.add_argument("--wav", default=None, help="write a WAV instead of raw stdout")
ap.add_argument("--no-agc", action="store_true")
args = ap.parse_args()

core = DemodCore(fs=args.fs, mode=args.mode, bfo=args.bfo,
                 audio_bw=args.audio_bw, agc=not args.no_agc)

stop = False


def on_signal(sig, frame):
    global stop
    stop = True


signal.signal(signal.SIGINT, on_signal)
signal.signal(signal.SIGTERM, on_signal)

src = sys.stdin.buffer if args.iq == "-" else open(args.iq, "rb")
block_bytes = int(args.fs * args.block_s) * 4  # s16 I + s16 Q per sample
collected = []
n_in = n_out = 0
try:
    while not stop:
        buf = src.read(block_bytes)
        if not buf:
            break
        if len(buf) % 4:
            buf = buf[:len(buf) - len(buf) % 4]
        d = np.frombuffer(buf, dtype=np.int16)
        iq = (d[0::2].astype(np.float32) + 1j * d[1::2].astype(np.float32)) / 32768.0
        audio = core.process(iq)
        n_in += len(iq)
        n_out += len(audio)
        if args.wav:
            collected.append(audio)
        else:
            sys.stdout.buffer.write((audio * 32767).astype(np.int16).tobytes())
            sys.stdout.buffer.flush()
except BrokenPipeError:
    pass
finally:
    if src is not sys.stdin.buffer:
        src.close()

if args.wav and collected:
    audio = np.concatenate(collected)
    sf.write(args.wav, audio, core.out_sr)
    print(f"wrote {args.wav}: {len(audio)/core.out_sr:.1f}s", file=sys.stderr)
print(f"demod_stream: {n_in} IQ samples -> {n_out} audio samples "
      f"({n_out/core.out_sr:.1f}s), clips={core.clip_count}, "
      f"gain [{core.gain_min_seen:.2f}, {core.gain_max_seen:.2f}]", file=sys.stderr)
