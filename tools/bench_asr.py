#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Benchmark ASR candidates through the REAL live chunker (silero VAD), so
numbers reflect the production path, not whole-file decoding.

Candidates:
  w1  faster-whisper large-v3        (direct zh->en translate; incumbent)
  w2  faster-whisper large-v3-turbo  (rejected: translate echoes source)
  s1  SenseVoiceSmall int8 (sherpa)  (ASR zh/en/ja/ko/yue only; +MT for others)
  q1  Qwen3-ASR-0.6B int8 (sherpa)   (ASR; +MT)

Model downloads (untrusted: extract into their own dirs, pass paths here):
  s1: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/
      sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
  q1: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/
      sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2

Metrics per chunk: infer_s (median of --repeats after 1 warmup), rtf, detected
language, text, repetition ratio, segment gates, CJK ratio. --mt adds an ollama
translation stage for ASR-only arms (wall-time measured separately; end-to-end
RTF includes it). Results: captures/bench/results.jsonl + per-candidate texts.

Usage:
  tools/bench_asr.py --cands w1,s1,q1 [--repeats 3] [--mt] [--fixtures GLOB...]
"""
import argparse
import gc
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import asr  # noqa: E402
import chunker  # noqa: E402
from vad import SileroVad  # noqa: E402

SV_DIR = "/tmp/sensevoice-dl/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
Q3_DIR = "/tmp/qwen3asr-dl/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--cands", default="w1,s1,q1")
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--mt", action="store_true", help="add ollama MT stage for ASR-only arms")
ap.add_argument("--mt-model", default="qwen3.6:27b")
ap.add_argument("--s1-dir", default=SV_DIR)
ap.add_argument("--q1-dir", default=Q3_DIR)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--fixtures", nargs="*", default=None)
ap.add_argument("--out-dir", default=os.path.join(ROOT, "captures", "bench"))
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)


def fixtures():
    if args.fixtures:
        return sorted(f for pat in args.fixtures for f in glob.glob(pat))
    fs = sorted(glob.glob(os.path.join(ROOT, "samples", "lang_*.wav")))
    fs += [os.path.join(ROOT, "samples", "franceinfo_60s.wav")]
    fs += sorted(glob.glob(os.path.join(ROOT, "captures", "*radio*.wav")))
    return [f for f in fs if os.path.isfile(f)]


def expected_lang(path):
    b = os.path.basename(path)
    if b.startswith("lang_"):
        return b[len("lang_"):-len(".wav")]
    if b.startswith("franceinfo"):
        return "fr"
    return None


def chunks_of(audio):
    ck = chunker.Chunker(vad=SileroVad())
    out = ck.feed(audio)
    out.extend(ck.flush())
    return [a for a, _ in out], ck.stats


def load_cand(c):
    if c in ("w1", "w2"):
        name = "large-v3" if c == "w1" else "large-v3-turbo"
        return "whisper", asr.load_whisper(name)
    import sherpa_onnx
    if c == "s1":
        d = args.s1_dir
        return "sensevoice", sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=f"{d}/model.int8.onnx", tokens=f"{d}/tokens.txt",
            num_threads=args.threads, use_itn=True, language="")
    if c == "q1":
        d = args.q1_dir
        return "qwen3asr", sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=f"{d}/conv_frontend.onnx", encoder=f"{d}/encoder.int8.onnx",
            decoder=f"{d}/decoder.int8.onnx", tokenizer=f"{d}/tokenizer",
            num_threads=args.threads, max_new_tokens=256)
    raise ValueError(f"unknown candidate {c}")


def infer(kind, model, audio):
    if kind == "whisper":
        segs, info = asr.transcribe(model, audio, task="translate", profile="live",
                                    no_repeat_ngram_size=4)
        segs = list(segs)
        text = " ".join(s.text.strip() for s in segs).strip()
        return dict(text=text, lang=info.language,
                    lang_p=round(info.language_probability, 3),
                    gates=asr.segment_gates(segs), task="translate")
    s = model.create_stream()
    s.accept_waveform(asr.SR, audio)
    model.decode_stream(s)
    r = s.result
    lang = getattr(r, "lang", None) or None
    if lang:  # SenseVoice tags look like "<|zh|>"; normalise to "zh"
        lang = lang.strip("<|>") or None
    return dict(text=r.text.strip(), lang=lang,
                lang_p=None, gates=None, task="transcribe")


def mt_ollama(text):
    payload = {"model": args.mt_model, "stream": False, "think": False,
               "options": {"temperature": 0},
               "prompt": ("Translate this radio broadcast text to English. "
                          "Output only the English translation.\n\n" + text)}
    req = urllib.request.Request("http://localhost:11434/api/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=180))
    return resp["response"].strip(), time.time() - t0


def vram_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def main():
    fix = fixtures()
    print(f"fixtures: {len(fix)}; candidates: {args.cands}; repeats={args.repeats}; mt={args.mt}", flush=True)
    chunk_cache = {}
    for f in fix:
        audio = asr.load_wav(f)
        ch, stats = chunks_of(audio)
        chunk_cache[f] = (audio, ch, stats)
        print(f"  {os.path.basename(f)}: {len(audio)/asr.SR:.0f}s -> {len(ch)} chunks {stats}", flush=True)

    results = []
    for c in args.cands.split(","):
        c = c.strip()
        t_load = time.time()
        kind, model = load_cand(c)
        load_s = time.time() - t_load
        print(f"\n=== {c} ({kind}) loaded {load_s:.1f}s, VRAM0 {vram_mb()} MiB ===", flush=True)
        # warmup on the first non-empty chunk
        warm = next((ch[0] for f in fix if (ch := chunk_cache[f][1])), None)
        if warm is not None:
            infer(kind, model, warm)
        for f in fix:
            audio, ch, stats = chunk_cache[f]
            exp = expected_lang(f)
            per = []
            texts = []
            mt_total = 0.0
            for k, a in enumerate(ch):
                durs = []
                res = None
                for _ in range(args.repeats):
                    t0 = time.time()
                    res = infer(kind, model, a)
                    durs.append(time.time() - t0)
                infer_s = float(np.median(durs))
                dur = len(a) / asr.SR
                text = res["text"]
                mt_s = 0.0
                if args.mt and kind != "whisper" and text:
                    text, mt_s = mt_ollama(text)
                    mt_total += mt_s
                rep = asr.repetition_flags(text)
                cjk = asr.cjk_ratio(text)
                per.append(dict(chunk=k, dur=round(dur, 2), infer_s=round(infer_s, 3),
                                rtf=round(infer_s / dur, 4), lang=res["lang"],
                                lang_p=res["lang_p"], rep_ratio=round(rep["ratio"], 3),
                                cjk=round(cjk, 3), gates=res["gates"], mt_s=round(mt_s, 2)))
                texts.append(f"[{k} | {res['lang']} | {dur:.1f}s | infer {infer_s:.2f}s"
                             + (f" | mt {mt_s:.1f}s" if mt_s else "") + f"] {text}")
            n = len(ch)
            langs = [p["lang"] for p in per if p["lang"]]
            lang_match = (exp is None) or (exp in langs)
            rec = dict(cand=c, kind=kind, fixture=os.path.basename(f), expected=exp,
                       chunks=n, chunker=stats, load_s=round(load_s, 1),
                       lang_match=lang_match,
                       mean_rtf=round(float(np.mean([p["rtf"] for p in per])), 4) if per else None,
                       mean_infer_s=round(float(np.mean([p["infer_s"] for p in per])), 3) if per else None,
                       mt_total_s=round(mt_total, 1),
                       rep_chunks=sum(1 for p in per if p["rep_ratio"] > 0),
                       cjk_chunks=sum(1 for p in per if p["cjk"] > 0.3),
                       per=per)
            results.append(rec)
            with open(os.path.join(args.out_dir, "results.jsonl"), "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            with open(os.path.join(args.out_dir, f"{c}__{os.path.basename(f)}.txt"), "w") as fh:
                fh.write("\n".join(texts) + "\n")
            tag = "PASS" if lang_match else ("n/a" if exp is None else "FAIL")
            print(f"  {os.path.basename(f):28s} {c} | {n:2d} chunks | lang {tag} "
                  f"| rtf {rec['mean_rtf']} | infer {rec['mean_infer_s']}s | mt {mt_total:.1f}s "
                  f"| rep {rec['rep_chunks']} cjk {rec['cjk_chunks']}", flush=True)
        del model
        gc.collect()
        print(f"  {c} unloaded, VRAM0 {vram_mb()} MiB", flush=True)

    print("\n=== summary ===")
    for c in args.cands.split(","):
        rs = [r for r in results if r["cand"] == c.strip()]
        lid = [r for r in rs if r["expected"]]
        npass = sum(1 for r in lid if r["lang_match"])
        rtfs = [r["mean_rtf"] for r in rs if r["mean_rtf"] is not None]
        print(f"{c}: lang-id {npass}/{len(lid)} | mean RTF "
              f"{np.mean(rtfs):.4f} | rep-chunks {sum(r['rep_chunks'] for r in rs)} "
              f"| cjk-chunks {sum(r['cjk_chunks'] for r in rs)}")


if __name__ == "__main__":
    main()
