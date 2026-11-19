# SPDX-License-Identifier: AGPL-3.0-or-later
"""Voice-cloning proof: speak the English translation in the ORIGINAL speaker's
voice, using a few seconds of the source audio as a zero-shot reference (F5-TTS).

Two cases: Chinese (F5 supports ZH natively) and French (F5 is EN/ZH-optimised,
so this is a best-effort cross-lingual clone). Generates WAVs only -- no playback.
Run in the .venv-tts environment.
"""
import time

from f5_tts.api import F5TTS

S = "/home/ian/Code/radio-translator/samples"


def lines(path, n):
    with open(path) as f:
        return " ".join(x.strip() for x in f.read().splitlines()[:n] if x.strip())


print("loading F5-TTS (downloads base model on first run)...", flush=True)
t0 = time.time()
f5 = F5TTS()
print(f"loaded in {time.time() - t0:.1f}s", flush=True)

cases = [
    ("zh", f"{S}/ref_zh.wav", lines(f"{S}/zh_ref.txt", 4),
     lines(f"{S}/zh_en.txt", 3), f"{S}/cloned_zh_voice_en.wav"),
    ("fr", f"{S}/ref_fr.wav", lines(f"{S}/french.txt", 5),
     lines(f"{S}/english.txt", 5), f"{S}/cloned_fr_voice_en.wav"),
]
for name, ref, ref_text, gen, out in cases:
    t = time.time()
    wav, sr, spec = f5.infer(ref_file=ref, ref_text=ref_text, gen_text=gen, file_wave=out)
    dt = time.time() - t
    dur = len(wav) / sr
    print(f"[{name}] cloned {dur:.1f}s of English audio in {dt:.1f}s "
          f"(RTF {dt / dur:.2f}) sr={sr} -> {out}", flush=True)