# SPDX-License-Identifier: AGPL-3.0-or-later
"""English text -> English speech WAV via Kokoro-82M (ONNX, CPU).

Usage: python speak.py in.txt out.wav
Strips [t -> t] timestamp prefixes, splits long text on sentence boundaries
(Kokoro caps ~510 phonemes per call), concatenates the audio.
"""
import re
import sys
import time

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

MODELS = "/home/ian/Code/radio-translator/models"

txt, out_wav = sys.argv[1], sys.argv[2]
text = open(txt).read().strip()
text = re.sub(r"\[\s*\d+\.\d+ -> \d+\.\d+\]\s*", "", text)
text = re.sub(r"\s+", " ", text).strip()
if not text:
    raise SystemExit("no text to speak")

t0 = time.time()
kok = Kokoro(f"{MODELS}/kokoro-v1.0.onnx", f"{MODELS}/voices-v1.0.bin")

sentences = re.split(r"(?<=[.!?])\s+", text)
chunks, cur = [], ""
for s in sentences:
    if len(cur) + len(s) > 400 and cur:
        chunks.append(cur)
        cur = s
    else:
        cur = (cur + " " + s).strip()
if cur:
    chunks.append(cur)

audio = []
sr = None
for c in chunks:
    samples, sr = kok.create(c, voice="af_heart", speed=1.0, lang="en-us")
    audio.append(samples)
out = np.concatenate(audio)
sf.write(out_wav, out, sr)
dt = time.time() - t0
print(
    f"tts: {dt:.1f}s wall for {len(out) / sr:.1f}s audio "
    f"({len(out) / sr / dt:.1f}x realtime), {len(chunks)} chunks, sr={sr}"
)
