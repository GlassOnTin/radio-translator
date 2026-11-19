#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Haven-invocable local translation service (workstation CUDA GPU).

translate(audio file) -> English text (stdout) + optional English speech WAV.
Fully local, no internet, no playback. Designed to be driven remotely by Haven
(the pocket thin-client) over SSH exec (run_command) — the GPU work runs here
on the workstation, not in the phone's PRoot guest.

Usage:
  translate_service.py AUDIO [--speech OUT.wav] [--lang xx] [--task translate]
                       [--model large-v3] [--json]

  AUDIO    any audio file ffmpeg can read (decoded to 16 kHz mono internally)
  --speech write Kokoro English TTS to OUT.wav (CPU; never played)
  --lang   force source language (default: auto-detect)
  --task   translate (default) | transcribe
  --json   emit a JSON object on stdout instead of plain text

stdout: translation text (or JSON with --json). stderr: progress/summary.
Exit code 0 on success. No --play, no speaker output, no network.
"""
import os
import sys


def _bootstrap_cuda():
    """ctranslate2 needs libcublas.so.12 (CUDA 12); the system CUDA may be v13.
    The venv ships the v12 libs via nvidia-* wheels. Put them on LD_LIBRARY_PATH
    before ctranslate2 loads, re-execing once if needed (env must be set before
    the library is loaded, i.e. before import)."""
    here = os.path.dirname(os.path.abspath(__file__))
    nv = os.path.join(here, ".venv", "lib", "python3.12", "site-packages", "nvidia")
    if not os.path.isdir(nv):
        return
    libs = ":".join(os.path.join(nv, d, "lib") for d in sorted(os.listdir(nv))
                    if os.path.isdir(os.path.join(nv, d, "lib")))
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if libs and libs not in cur:
        os.environ["LD_LIBRARY_PATH"] = libs + (":" + cur if cur else "")
        os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:])


_bootstrap_cuda()

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

SR = 16000


def decode_audio(path):
    """Decode any ffmpeg-readable file to 16 kHz mono float32 via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
         "-f", "s16le", "-ac", "1", "-ar", str(SR), "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm


def main():
    ap = argparse.ArgumentParser(description="Local radio translation service (Haven-invocable)")
    ap.add_argument("audio", help="audio file to translate (any ffmpeg-readable format)")
    ap.add_argument("--speech", default=None, help="write English Kokoro TTS to this WAV (not played)")
    ap.add_argument("--lang", default=None, help="force source language (default auto)")
    ap.add_argument("--task", default="translate", choices=["translate", "transcribe"])
    ap.add_argument("--model", default=os.environ.get("WHISPER_MODEL", "large-v3"))
    ap.add_argument("--json", action="store_true", help="emit JSON on stdout")
    args = ap.parse_args()

    if not os.path.isfile(args.audio):
        print(f"error: no such file: {args.audio}", file=sys.stderr)
        return 2

    t0 = time.time()
    audio = decode_audio(args.audio)
    dur = len(audio) / SR
    print(f"[service] decoded {args.audio}: {dur:.1f}s @ {SR}Hz; loading {args.model} (cuda)...",
          file=sys.stderr, flush=True)
    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    t_load = time.time() - t0

    kwargs = dict(task=args.task, beam_size=5, vad_filter=True,
                  condition_on_previous_text=False)
    if args.lang:
        kwargs["language"] = args.lang
    t1 = time.time()
    segments, info = model.transcribe(audio, **kwargs)
    lines = [s.text.strip() for s in segments if s.text.strip()]
    text = "\n".join(lines)
    t_infer = time.time() - t1

    speech_file = None
    tts_s = 0.0
    if args.speech and text:
        from kokoro_onnx import Kokoro
        models = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        kok = Kokoro(f"{models}/kokoro-v1.0.onnx", f"{models}/voices-v1.0.bin")
        tt = time.time()
        samples, sr = kok.create(text, voice="af_heart", speed=1.0, lang="en-us")
        sf.write(args.speech, np.asarray(samples, dtype=np.float32), sr)
        tts_s = time.time() - tt
        speech_file = os.path.abspath(args.speech)
        print(f"[service] wrote speech {speech_file} ({len(samples)/sr:.1f}s, {tts_s:.1f}s CPU)",
              file=sys.stderr, flush=True)

    rtf = t_infer / dur if dur else 0.0
    print(f"[service] lang={info.language} p={info.language_probability:.2f} "
          f"audio={dur:.1f}s load={t_load:.1f}s infer={t_infer:.1f}s RTF={rtf:.3f}",
          file=sys.stderr, flush=True)

    if args.json:
        print(json.dumps({
            "text": text, "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration_s": round(dur, 2), "rtf": round(rtf, 4),
            "speech_file": speech_file, "task": args.task,
        }, ensure_ascii=False))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())