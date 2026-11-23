#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Persistent streaming voice-isolation stage for the live path (inworld-ai/
voice-isolator, Apache-2.0). Reads raw s16le 16 kHz mono from stdin, isolates
voice with the causal streaming model (48 kHz internal, ~63 ms latency at
lookahead 2), writes s16le 16 kHz mono to stdout. The model loads once.

Runs under .venv-tts (torch + scipy). Resampling 16k<->48k uses overlap-save
polyphase (scipy.resample_poly) so block boundaries stay continuous.

  ffmpeg -i in.wav -f s16le -ar 16000 -ac 1 pipe:1 | \\
      .venv-tts/bin/python tools/isolate_stream.py --device cuda | \\
      ./run.sh live.py --stdin
"""
import argparse
import os
import signal
import sys

import numpy as np
from scipy.signal import resample_poly

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--device", default="cuda")
ap.add_argument("--lookahead", type=int, default=2, choices=[0, 2, 4])
ap.add_argument("--block-s", type=float, default=0.25)
ap.add_argument("--model", default="inworld/voice-isolator")
args = ap.parse_args()

SR_IN, SR_M = 16000, 48000


class Resampler:
    """Overlap-save polyphase resampler: keeps left context across blocks so
    the FIR transient at a block edge is recomputed with real history."""

    def __init__(self, up, down, pad=128):
        self.up, self.down, self.pad = up, down, pad
        self.hist = np.zeros(0, dtype=np.float32)

    def process(self, x):
        if len(x) == 0:
            return np.zeros(0, dtype=np.float32)
        buf = np.concatenate([self.hist, x])
        y = resample_poly(buf, self.up, self.down).astype(np.float32)
        off = int(round(len(self.hist) * self.up / self.down))
        self.hist = buf[-self.pad:].copy()
        return y[off:]


def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def main():
    from inworld import voice_isolation
    iso = voice_isolation.VoiceIsolator(args.model, device=args.device,
                                        mode="streaming", lookahead_frames=args.lookahead)
    stream = iso.stream()
    up, down = Resampler(3, 1), Resampler(1, 3)

    stop = False

    def on_sig(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    block_bytes = int(SR_IN * args.block_s) * 2
    n_in = n_out = 0
    out = sys.stdout.buffer
    try:
        while not stop:
            buf = sys.stdin.buffer.read(block_bytes)
            if not buf:
                break
            if len(buf) % 2:
                buf = buf[:len(buf) - 1]
            x = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
            n_in += len(x)
            y48 = to_np(stream.push(up.process(x)))
            if len(y48):
                y16 = down.process(y48)
                n_out += len(y16)
                out.write((np.clip(y16, -1, 1) * 32767).astype(np.int16).tobytes())
                out.flush()
        y48 = to_np(stream.flush(as_numpy=True))
        if len(y48):
            y16 = down.process(y48)
            n_out += len(y16)
            out.write((np.clip(y16, -1, 1) * 32767).astype(np.int16).tobytes())
            out.flush()
    except BrokenPipeError:
        pass
    print(f"isolate_stream: {n_in/ SR_IN:.1f}s in -> {n_out/SR_IN:.1f}s out, "
          f"latency {stream.latency_seconds*1000:.0f} ms, lookahead {args.lookahead}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
