# SPDX-License-Identifier: AGPL-3.0-or-later
"""Batch verification: translate every samples/lang_*.wav to English with a
single model load. Prints detected language, English text, and RTF per file,
and writes <name>_en.txt next to each WAV.
"""
import glob
import os
import time

import soundfile as sf
from faster_whisper import WhisperModel

MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
files = sorted(glob.glob("/home/ian/Code/radio-translator/samples/lang_*.wav"))
print(f"loading {MODEL}; {len(files)} samples", flush=True)
model = WhisperModel(MODEL, device="cuda", compute_type="float16")

for f in files:
    audio, sr = sf.read(f, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        print(f"\n### {os.path.basename(f)}: EMPTY (capture failed)")
        continue
    t0 = time.time()
    segs, info = model.transcribe(
        audio, task="translate", beam_size=5,
        vad_filter=True, condition_on_previous_text=False,
    )
    text = " ".join(s.text.strip() for s in segs).strip()
    dt = time.time() - t0
    dur = len(audio) / sr
    print(
        f"\n### {os.path.basename(f)} | detected={info.language} "
        f"p={info.language_probability:.2f} | {dur:.0f}s | infer {dt:.1f}s RTF={dt / dur:.3f}"
    )
    print(text if text else "(no speech detected)")
    with open(f.replace(".wav", "_en.txt"), "w") as fh:
        fh.write(text)