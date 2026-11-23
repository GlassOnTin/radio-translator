#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Canary-1b-v2 translation backend server (CC-BY-4.0). Runs in .venv-tts
(torch + transformers). Reads length-prefixed JSON requests {audio_b64, lang}
on stdin, writes length-prefixed JSON {text, language, language_probability} on
stdout. European-language X->English S2TT; the daemon routes by detected
language and falls back to whisper for non-European.

Launched by backends.SubprocessBackend("canary_backend.py"). Not run directly.
"""
import argparse
import base64
import json
import struct
import sys

import numpy as np
import torch


def read_req():
    hdr = sys.stdin.buffer.read(4)
    if len(hdr) < 4:
        return None
    n = struct.unpack("<I", hdr)[0]
    return json.loads(sys.stdin.buffer.read(n))


def write_resp(obj):
    data = json.dumps(obj).encode()
    sys.stdout.buffer.write(struct.pack("<I", len(data)) + data)
    sys.stdout.buffer.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="nvidia/canary-1b-v2")
    args = ap.parse_args()

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    dtype = torch.float16 if args.device != "cpu" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"[canary] loaded {args.model} on {args.device}", file=sys.stderr, flush=True)

    while True:
        req = read_req()
        if req is None:
            break
        try:
            pcm = base64.b64decode(req["audio_b64"])
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            src_lang = req.get("lang") or "en"
            inputs = processor.apply_transcription_request(
                audio=audio, source_language=src_lang, target_language="en")
            inputs = inputs.to(args.device)
            if "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(dtype)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=256)
            text = processor.decode(gen, skip_special_tokens=True)[0].strip()
            write_resp({"text": text, "language": src_lang, "language_probability": 1.0})
        except Exception as e:
            write_resp({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
