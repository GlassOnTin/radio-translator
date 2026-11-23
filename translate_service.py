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
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _bootstrap():
    """Make a direct call work regardless of cwd/interpreter (Haven run_command
    execs with cwd=$HOME and no login profile). Two fixes, applied by re-execing
    once before any heavy import:
      1. Run under the project venv python (.venv/bin/python) — the system
         python3 lacks soundfile/faster-whisper/kokoro.
      2. Put the venv's CUDA-12 nvidia libs (libcublas.so.12) on LD_LIBRARY_PATH
         before ctranslate2 loads (system CUDA may be v13).
    run.sh already does both, so when invoked via run.sh this is a no-op."""
    venv_py = os.path.join(HERE, ".venv", "bin", "python")
    nv = glob.glob(os.path.join(HERE, ".venv", "lib", "python3.*", "site-packages", "nvidia"))
    libs = ""
    if nv:
        libs = ":".join(os.path.join(nv[0], d, "lib") for d in sorted(os.listdir(nv[0]))
                        if os.path.isdir(os.path.join(nv[0], d, "lib")))
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    need_libs = bool(libs) and libs not in cur
    use_venv = os.path.isfile(venv_py) and \
        os.path.realpath(sys.executable) != os.path.realpath(venv_py)
    if use_venv or need_libs:
        if need_libs:
            os.environ["LD_LIBRARY_PATH"] = libs + (":" + cur if cur else "")
        exe = venv_py if use_venv else sys.executable
        os.execv(exe, [exe, os.path.abspath(__file__)] + sys.argv[1:])


_bootstrap()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import asr  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

SR = asr.SR


def main():
    ap = argparse.ArgumentParser(description="Local radio translation service (Haven-invocable)")
    ap.add_argument("audio", help="audio file to translate (any ffmpeg-readable format)")
    ap.add_argument("--speech", default=None, help="write English Kokoro TTS to this WAV (not played)")
    ap.add_argument("--lang", default=None, help="force source language (default auto)")
    ap.add_argument("--task", default="translate", choices=["translate", "transcribe"])
    ap.add_argument("--model", default=os.environ.get("WHISPER_MODEL", "large-v3"))
    ap.add_argument("--json", action="store_true", help="emit JSON on stdout")
    args = ap.parse_args()

    # Resolve relative input against the project dir so a direct call works
    # regardless of the exec cwd (Haven run_command cwd is $HOME, not here).
    audio_path = args.audio if os.path.isabs(args.audio) else os.path.join(HERE, args.audio)
    if not os.path.isfile(audio_path):
        print(f"error: no such file: {audio_path}", file=sys.stderr)
        return 2
    args.audio = audio_path
    if args.speech:
        args.speech = os.path.abspath(args.speech)

    t0 = time.time()
    audio = asr.decode_audio(args.audio)
    dur = len(audio) / SR
    print(f"[service] decoded {args.audio}: {dur:.1f}s @ {SR}Hz; loading {args.model} (cuda)...",
          file=sys.stderr, flush=True)
    model = asr.load_whisper(args.model)
    t_load = time.time() - t0

    t1 = time.time()
    segments, info = asr.transcribe(model, audio, task=args.task,
                                    lang=args.lang, profile="file")
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