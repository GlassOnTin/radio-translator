# SPDX-License-Identifier: AGPL-3.0-or-later
"""Batch regression: translate every samples/lang_*.wav to English with a
single model load. Expected language comes from the filename; prints a
PASS/FAIL table with detected language, probability and RTF per file.

Reference translations (lang_XX_en.txt) are NEVER written unless --write is
given, so regression runs cannot clobber the fixtures they check against.
"""
import argparse
import glob
import os
import time

import asr

ap = argparse.ArgumentParser(description="14-language detection/translation regression")
ap.add_argument("--write", action="store_true",
                help="regenerate lang_XX_en.txt references (default: leave them untouched)")
ap.add_argument("--out-dir", default=None, help="with --write: put references here")
ap.add_argument("--model", default=None, help="override model (default WHISPER_MODEL or large-v3)")
ap.add_argument("--device", default="cuda", help="cuda|cpu (compute tier)")
ap.add_argument("--compute-type", default="float16", help="float16|int8|int8_float16|...")
ap.add_argument("--cpu-threads", type=int, default=0)
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(asr.HERE, "samples", "lang_*.wav")))
model_name = args.model or os.environ.get("WHISPER_MODEL", "large-v3")
print(f"loading {model_name} device={args.device} compute={args.compute_type}; "
      f"{len(files)} samples", flush=True)
model = asr.load_whisper(args.model, device=args.device, compute_type=args.compute_type,
                         cpu_threads=args.cpu_threads)

npass = 0
for f in files:
    base = os.path.basename(f)
    expected = base[len("lang_"):-len(".wav")]
    audio = asr.load_wav(f)
    if len(audio) == 0:
        print(f"\n### {base}: EMPTY (capture failed)")
        continue
    t0 = time.time()
    segs, info = asr.transcribe(model, audio, profile="file")
    dt = time.time() - t0
    text = " ".join(s.text.strip() for s in segs).strip()
    ok = info.language == expected
    npass += ok
    dur = len(audio) / asr.SR
    print(
        f"\n### {base} | expected={expected} detected={info.language} "
        f"{'PASS' if ok else 'FAIL'} p={info.language_probability:.2f} | "
        f"{dur:.0f}s | infer {dt:.1f}s RTF={dt / dur:.3f}"
    )
    print(text if text else "(no speech detected)")
    if args.write:
        out = os.path.join(args.out_dir, base.replace(".wav", "_en.txt")) \
            if args.out_dir else f.replace(".wav", "_en.txt")
        with open(out, "w") as fh:
            fh.write(text)

print(f"\nlang-id: {npass}/{len(files)} PASS")
