#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SeamlessM4T v2 speech-to-speech backend server. NON-COMMERCIAL (CC-BY-NC-4.0
weights) — opt-in experiment, never the default. Runs in .venv-tts (torch +
transformers). Reads length-prefixed JSON {audio_b64, lang, tgt_lang, s2st} on
stdin; writes {audio_b64 (f32le 16 kHz translated speech), text, language}.

The daemon resamples the 16 kHz output to 24 kHz for the plugin's type-1 TTS
player. Weights are fetched only via explicit `scripts/fetch_models.py
--seamless` (never auto-downloaded by a default install).

Launched by backends.SubprocessBackend("seamless_backend.py"). Not run directly.
"""
import argparse
import base64
import json
import struct
import sys

import numpy as np
import torch

# whisper 2-letter -> SeamlessM4T 3-letter codes (subset; extend as needed)
LANG_MAP = {
    "es": "spa", "fr": "fra", "de": "deu", "it": "ita", "pt": "por",
    "ru": "rus", "ar": "arb", "zh": "cmn", "ja": "jpn", "ko": "kor",
    "en": "eng", "tr": "tur", "pl": "pol", "uk": "ukr", "el": "ell",
    "fa": "pes", "hi": "hin", "nl": "nld", "ro": "ron", "sv": "swe",
}


def read_req():
    hdr = sys.stdin.buffer.read(4)
    if len(hdr) < 4:
        return None
    return json.loads(sys.stdin.buffer.read(struct.unpack("<I", hdr)[0]))


def write_resp(obj):
    data = json.dumps(obj).encode()
    sys.stdout.buffer.write(struct.pack("<I", len(data)) + data)
    sys.stdout.buffer.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="facebook/seamless-m4t-v2-large")
    args = ap.parse_args()

    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToSpeech
    dtype = torch.float16 if args.device != "cpu" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device).eval()
    print(f"[seamless] loaded {args.model} on {args.device} (CC-BY-NC, opt-in)",
          file=sys.stderr, flush=True)

    while True:
        req = read_req()
        if req is None:
            break
        try:
            pcm = base64.b64decode(req["audio_b64"])
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            src = LANG_MAP.get(req.get("lang") or "en", "eng")
            tgt = LANG_MAP.get(req.get("tgt_lang") or "en", "eng")
            inputs = processor(audio=audio, sampling_rate=16000, return_tensors="pt",
                               src_lang=src, tgt_lang=tgt)
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, tgt_lang=tgt)
            # ForSpeechToSpeech.generate returns translated audio waveform(s)
            wav = out.audio if hasattr(out, "audio") else out
            if isinstance(wav, (list, tuple)):
                wav = wav[0]
            wav = np.asarray(wav.detach().cpu().float()).reshape(-1)
            write_resp({"audio_b64": base64.b64encode(wav.tobytes()).decode(),
                        "language": req.get("lang") or "en", "sample_rate": 16000})
        except Exception as e:
            write_resp({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
