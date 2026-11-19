# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline verification: foreign radio WAV -> English text (or source transcript).

Usage: python translate_file.py in.wav out.txt [translate|transcribe] [language]
Prints per-segment timestamps and a real-time-factor summary.
"""
import os
import sys
import time

import soundfile as sf
from faster_whisper import WhisperModel

# large-v3-turbo's translate task echoes the source language; use large-v3
# for translation. Override with WHISPER_MODEL.
MODEL = os.environ.get("WHISPER_MODEL", "large-v3")

wav, out_txt = sys.argv[1], sys.argv[2]
task = sys.argv[3] if len(sys.argv) > 3 else "translate"
language = sys.argv[4] if len(sys.argv) > 4 else None

# Load via soundfile and pass a float32 array: bypasses faster-whisper's
# av.open() path, which breaks on PyAV versions that dropped metadata_errors.
audio, sr = sf.read(wav, dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)
assert sr == 16000, f"expected 16 kHz, got {sr}"

t0 = time.time()
model = WhisperModel(MODEL, device="cuda", compute_type="float16")
t1 = time.time()
segments, info = model.transcribe(
    audio,
    task=task,
    language=language,
    beam_size=5,
    vad_filter=True,
    condition_on_previous_text=False,
)
lines = []
for s in segments:
    print(f"[{s.start:6.1f} -> {s.end:6.1f}] {s.text.strip()}", flush=True)
    lines.append(s.text.strip())
t2 = time.time()
print(
    f"\nlang={info.language} p={info.language_probability:.2f} "
    f"audio={info.duration:.1f}s load={t1 - t0:.1f}s infer={t2 - t1:.1f}s "
    f"RTF={(t2 - t1) / info.duration:.3f}"
)
with open(out_txt, "w") as f:
    f.write("\n".join(lines))
