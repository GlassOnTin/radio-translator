#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare raw vs enhanced audio through the live pipeline metrics: silero
speech fraction, chunker chunk count/reasons, and whisper translate text with
hallucination gates. Run under .venv (faster-whisper CUDA). Pairs of WAVs:

  tools/enhance_metrics.py RAW1.wav ISO1.wav RAW2.wav ISO2.wav ...
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import asr  # noqa: E402
import chunker  # noqa: E402
from vad import SileroVad  # noqa: E402


def speech_fraction(audio):
    v = SileroVad()
    pcm = (audio * 32767).astype(np.int16).tobytes()
    dec = [v.is_speech(pcm[i:i + 960], 16000) for i in range(0, len(pcm) - 960, 960)]
    return sum(dec) / max(1, len(dec))


def analyze(path, model):
    audio = asr.load_wav(path)
    frac = speech_fraction(audio)
    ck = chunker.Chunker(vad=SileroVad())
    out = ck.feed(audio)
    out.extend(ck.flush())
    rows = []
    for a, meta in out:
        segs, info = asr.transcribe(model, a, task="translate", profile="live",
                                    no_repeat_ngram_size=4)
        segs = list(segs)
        text = " ".join(s.text.strip() for s in segs).strip()
        rep = asr.repetition_flags(text)
        rows.append(dict(lang=info.language, dur=meta["dur"], reason=meta["reason"],
                         text=text, rep=rep["rep4"] or rep["run4"],
                         cjk=asr.cjk_ratio(text), gates=asr.segment_gates(segs)))
    return dict(dur=len(audio) / asr.SR, frac=frac, stats=ck.stats, rows=rows)


def main():
    args = sys.argv[1:]
    if len(args) % 2:
        print("give pairs: RAW ISO RAW ISO ...")
        return 1
    model = asr.load_whisper()
    for i in range(0, len(args), 2):
        raw, iso = args[i], args[i + 1]
        label = os.path.basename(raw)
        print(f"\n========== {label} ==========")
        for tag, path in (("RAW", raw), ("ISO", iso)):
            r = analyze(path, model)
            print(f"  {tag}: {r['dur']:.0f}s | silero speech frac {r['frac']:.2f} | "
                  f"chunks {r['stats']}")
            for row in r["rows"]:
                flag = []
                if row["rep"]:
                    flag.append("REP")
                if row["cjk"] > 0.3:
                    flag.append("CJK")
                if row["gates"]["high_compression"]:
                    flag.append("COMP")
                if row["gates"]["no_speech"]:
                    flag.append("NOSP")
                fs = (" [" + ",".join(flag) + "]") if flag else ""
                print(f"    [{row['lang']} | {row['dur']:.1f}s | {row['reason']}]{fs} {row['text'][:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
