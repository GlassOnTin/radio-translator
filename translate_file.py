# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline verification: foreign radio WAV -> English text (or source transcript).

Usage: python translate_file.py in.wav out.txt [translate|transcribe] [language]
Prints per-segment timestamps and a real-time-factor summary.
"""
import sys
import time

import asr

wav, out_txt = sys.argv[1], sys.argv[2]
task = sys.argv[3] if len(sys.argv) > 3 else "translate"
language = sys.argv[4] if len(sys.argv) > 4 else None

audio = asr.load_wav(wav)

t0 = time.time()
model = asr.load_whisper()
t1 = time.time()
segments, info = asr.transcribe(model, audio, task=task, lang=language,
                                profile="file")
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
