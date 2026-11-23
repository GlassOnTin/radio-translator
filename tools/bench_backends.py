#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare translation backends on a fixture: whisper translate, Canary
(European, language-routed), whisper-mt (whisper transcribe + LLM MT). Reports
per-backend latency, sacreBLEU vs a reference English file, and the text.

Usage:
  tools/bench_backends.py samples/lang_es.wav --ref samples/lang_es_en.txt
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asr  # noqa: E402
import backends  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("fixture")
ap.add_argument("--ref", default=None, help="reference English text for BLEU")
ap.add_argument("--backends", default="whisper,canary,whisper-mt")
ap.add_argument("--mt-model", default="qwen3.6:27b")
args = ap.parse_args()

audio = asr.load_wav(args.fixture)
ref = open(args.ref).read().strip() if args.ref and os.path.isfile(args.ref) else None
whisper = asr.load_whisper()
src_lang, _p, _a = whisper.detect_language(audio)
print(f"fixture={args.fixture} detected source={src_lang}\n")

try:
    import sacrebleu
except ImportError:
    sacrebleu = None

canary = None
results = {}
for name in args.backends.split(","):
    name = name.strip()
    t0 = time.time()
    if name == "whisper":
        segs, info = asr.transcribe(whisper, audio, task="translate", profile="file")
        text = " ".join(s.text.strip() for s in segs).strip()
    elif name == "canary":
        if canary is None:
            canary = backends.SubprocessBackend("canary_backend.py", device="cuda")
        if src_lang in backends.CANARY_LANGS:
            segs, info = canary.transcribe(audio, language=src_lang)
            text = " ".join(s.text.strip() for s in segs).strip()
        else:
            segs, info = asr.transcribe(whisper, audio, task="translate", profile="file")
            text = " ".join(s.text.strip() for s in segs).strip() + "  (whisper fallback)"
    elif name == "whisper-mt":
        mt = backends.WhisperMTModel(whisper, mt_model=args.mt_model)
        segs, info = mt.transcribe(audio)
        text = " ".join(s.text.strip() for s in segs).strip()
    else:
        continue
    dt = time.time() - t0
    bleu = sacrebleu.corpus_bleu([text], [[ref]]).score if (ref and sacrebleu) else None
    results[name] = (dt, bleu, text)
    bstr = f"BLEU {bleu:.1f}" if bleu is not None else "BLEU n/a"
    print(f"=== {name} ({dt:.1f}s, {bstr}) ===\n{text}\n")

if canary is not None:
    canary.close()
print("summary:", {k: (round(v[0], 1), round(v[1], 1) if v[1] is not None else None)
                   for k, v in results.items()})
