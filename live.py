# SPDX-License-Identifier: AGPL-3.0-or-later
"""Live foreign radio -> English text + English voice.

ffmpeg pulls an input (web stream, local file, ALSA/PipeWire source, or raw
s16le on stdin) as 16 kHz mono PCM; chunker.py cuts speech chunks at pauses
(pre-roll, soft-cap, pre-VAD AGC for quiet taps); faster-whisper translates
each chunk on GPU with hallucination gates. Language is auto-detected per
chunk, so retuning across stations on SW works (pin one with --lang). Kokoro
speaks the English on CPU in a worker thread.

Text -> stdout. Combined English audio -> WAV. Optional --play feeds audio to
ffplay as it is generated; OFF by default, so nothing reaches the speakers
unless you ask for it.

Usage:
  python live.py STREAM_URL [--seconds 90] [--out live_english.wav]
                 [--chunk 7.5] [--silence 300] [--preroll 300] [--carry 0]
                 [--lang xx] [--play] [--voice af_heart]
                 [--pulse SOURCE | --alsa DEVICE | --stdin]
                 [--af "highpass=f=80,lowpass=f=4500,afftdn"] [--no-gates]
                 [--vad silero|webrtc] [--vad-threshold 0.5]

RF paths: SDRconnect audio tap --pulse <sink>.monitor (GUI holds the RSP),
or tools/sdrconnect_ctl.py audio | live.py --stdin (headless WS), or the
headless IQ chain tools/sdrplay_cap | tools/demod_stream.py | live.py --stdin.
Worst-case chunk latency ~= --chunk + inference (default 7.5 s + ~0.5 s).
"""
import argparse
import asyncio
import os
import queue
import subprocess
import threading
import time

import asr
import chunker
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from translator import Translator
from vad import SileroVad

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
SR = asr.SR
FRAME_BYTES = SR * 30 // 1000 * 2  # 30 ms frames, 480 samples, s16le

ap = argparse.ArgumentParser(description="Live foreign radio -> English text + voice")
ap.add_argument("input", nargs="?", help="stream URL or local file")
ap.add_argument("--seconds", type=float, default=90.0, help="how long to listen")
ap.add_argument("--out", default="live_english.wav", help="combined English WAV")
ap.add_argument("--chunk", type=float, default=7.5, help="max chunk seconds (lower = less latency)")
ap.add_argument("--silence", type=int, default=300, help="trailing silence (ms) that closes a chunk")
ap.add_argument("--preroll", type=int, default=300, help="audio (ms) kept before chunk start")
ap.add_argument("--carry", type=float, default=0.0, help="speech overlap (s) carried into next chunk")
ap.add_argument("--lang", default=None, help="force source language (default: auto per chunk)")
ap.add_argument("--play", action="store_true", help="play English audio via ffplay (default off)")
ap.add_argument("--voice", default="af_heart", help="Kokoro voice")
ap.add_argument("--pulse", default=None, help="PipeWire/Pulse source to capture (e.g. a virtual-sink monitor)")
ap.add_argument("--alsa", default=None, help="ALSA capture device (e.g. hw:2,0)")
ap.add_argument("--stdin", action="store_true", help="read raw s16le 16 kHz mono from stdin")
ap.add_argument("--af", default=None, help="ffmpeg audio filter chain applied to the input")
ap.add_argument("--no-agc", action="store_true", help="disable pre-VAD AGC (A/B testing)")
ap.add_argument("--no-gates", action="store_true", help="disable hallucination gates (A/B testing)")
ap.add_argument("--vad", choices=["silero", "webrtc"], default="silero",
                help="chunker VAD: silero rejects music beds (default); webrtc is level-sensitive")
ap.add_argument("--vad-threshold", type=float, default=0.5, help="silero speech probability threshold")
ap.add_argument("--stream-tts", action="store_true",
                help="stream Kokoro audio sentence-by-sentence (lower time-to-first-audio with --play)")
ap.add_argument("--enhance", choices=["none", "isolate"], default="none",
                help="isolate: voice-isolation stage before the chunker, recovers speech under music/noise")
ap.add_argument("--enhance-device", default="cuda", help="device for the isolation stage (torch venv)")
ap.add_argument("--enhance-lookahead", type=int, default=2, choices=[0, 2, 4],
                help="isolation streaming lookahead frames (10 ms each; 2 = ~63 ms latency)")
args = ap.parse_args()

if not args.input and not args.pulse and not args.alsa and not args.stdin:
    ap.error("give an input URL/file, or --pulse/--alsa/--stdin")

# large-v3 (not turbo): turbo's translate task echoes the source language.
MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
print(f"loading models ({MODEL})...", flush=True)
t_load = time.time()
model = asr.load_whisper()
kok = Kokoro(f"{MODELS}/kokoro-v1.0.onnx", f"{MODELS}/voices-v1.0.bin")
print(f"models loaded in {time.time() - t_load:.1f}s", flush=True)

if args.stdin:
    in_args, src = ["-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0"], "stdin"
elif args.pulse:
    in_args, src = ["-f", "pulse", "-i", args.pulse], f"pulse:{args.pulse}"
elif args.alsa:
    in_args, src = ["-f", "alsa", "-i", args.alsa], f"alsa:{args.alsa}"
else:
    in_args, src = ["-i", args.input], args.input

af_args = ["-af", args.af] if args.af else []
proc = subprocess.Popen(
    ["ffmpeg", "-hide_banner", "-loglevel", "error", *in_args, *af_args,
     "-f", "s16le", "-ac", "1", "-ar", str(SR), "pipe:1"],
    stdout=subprocess.PIPE,
)

# Optional voice-isolation stage (inworld-ai/voice-isolator, Apache-2.0) runs
# in the torch venv as a persistent streaming subprocess: ffmpeg 16k PCM ->
# isolate_stream.py -> chunker. It inherits CUDA_VISIBLE_DEVICES so it shares
# the GPU mapping whisper uses. Adds ~63 ms latency at lookahead 2.
iso_proc = None
active_stream = proc.stdout
if args.enhance == "isolate":
    tts_py = os.path.join(HERE, ".venv-tts", "bin", "python")
    iso_proc = subprocess.Popen(
        [tts_py, os.path.join(HERE, "tools", "isolate_stream.py"),
         "--device", args.enhance_device, "--lookahead", str(args.enhance_lookahead)],
        stdin=proc.stdout, stdout=subprocess.PIPE)
    active_stream = iso_proc.stdout

# Opt-in playback. ffplay consumes 24 kHz float32 PCM (Kokoro's output rate).
play_proc = None
if args.play:
    play_proc = subprocess.Popen(
        ["ffplay", "-hide_banner", "-loglevel", "error", "-nodisp", "-autoexit",
         "-f", "f32le", "-ar", "24000", "-ch_layout", "mono", "-i", "pipe:0"],
        stdin=subprocess.PIPE,
    )

tts_q = queue.Queue()
tts_done = threading.Event()
tts_audio = []


def _play(pcm):
    if play_proc is not None:
        try:
            play_proc.stdin.write(pcm.tobytes())
            play_proc.stdin.flush()
        except BrokenPipeError:
            pass


async def _stream_collect(text):
    """Kokoro create_stream: yield audio per sentence; play each as it arrives.
    Returns (full_samples, sr, time_to_first_audio)."""
    chunks, sr, ttfa = [], 24000, None
    t0 = time.time()
    async for audio, s in kok.create_stream(text, voice=args.voice, speed=1.0, lang="en-us"):
        if ttfa is None:
            ttfa = time.time() - t0
        sr = s
        pcm = np.asarray(audio, dtype=np.float32)
        chunks.append(pcm)
        _play(pcm)
    full = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return full, sr, ttfa if ttfa is not None else time.time() - t0


def tts_worker():
    while True:
        item = tts_q.get()
        if item is None:
            break
        idx, text = item
        t0 = time.time()
        if args.stream_tts:
            samples, sr, ttfa = asyncio.run(_stream_collect(text))
        else:
            samples, sr = kok.create(text, voice=args.voice, speed=1.0, lang="en-us")
            samples = np.asarray(samples, dtype=np.float32)
            ttfa = time.time() - t0
            _play(samples)
        tts_audio.append((idx, samples, sr))
        print(f"  [voice {idx}] {len(samples) / sr:.1f}s audio in {time.time() - t0:.1f}s "
              f"(TTFA {ttfa:.2f}s)", flush=True)
    tts_done.set()


threading.Thread(target=tts_worker, daemon=True).start()


def read_exact(n):
    buf = bytearray()
    while len(buf) < n:
        part = active_stream.read(n - len(buf))
        if not part:
            return None
        buf.extend(part)
    return bytes(buf)


ck = chunker.Chunker(vad=SileroVad(threshold=args.vad_threshold) if args.vad == "silero" else None,
                     silence_ms=args.silence, cap_s=args.chunk,
                     preroll_ms=args.preroll, carry_s=args.carry,
                     agc_max_gain=1.0 if args.no_agc else 40.0)
trans = Translator(model, lang=args.lang, gates=not args.no_gates, carry=args.carry)
t_start = time.time()


def process_chunk(audio, meta):
    ev = trans.process_chunk(audio, meta)
    if ev is None:
        return
    head = (f"[chunk {ev['chunk']} | {ev['language']} | {ev['dur']:.1f}s | {ev['reason']} "
            f"| infer {ev['infer_s']:.1f}s | latency {ev['latency_s']:.1f}s]")
    if ev["suppressed"]:
        print(f"{head} SUPPRESSED ({','.join(ev['reasons'])}) {ev['text'][:100]}", flush=True)
        return
    print(f"{head} {ev['text']}", flush=True)
    tts_q.put((ev["chunk"], ev["text"]))


print(f"streaming {src} for {args.seconds:.0f}s (chunk<={args.chunk:.1f}s, "
      f"silence={args.silence}ms, vad={args.vad}, enhance={args.enhance}, "
      f"gates={not args.no_gates}, play={args.play}) ...", flush=True)
while time.time() - t_start < args.seconds:
    data = read_exact(FRAME_BYTES)
    if data is None:
        print("stream ended", flush=True)
        break
    frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    for audio, meta in ck.feed(frame):
        process_chunk(audio, meta)

for audio, meta in ck.flush():
    process_chunk(audio, meta)
proc.kill()
if iso_proc is not None:
    iso_proc.kill()
    iso_proc.wait()
tts_q.put(None)
tts_done.wait(timeout=120)

print(f"chunker: {ck.stats} suppressed={trans.suppressed}", flush=True)

if play_proc is not None:
    try:
        play_proc.stdin.close()
    except BrokenPipeError:
        pass
    rc = play_proc.wait(timeout=120)
    print(f"playback (ffplay) exit code: {rc}", flush=True)

if tts_audio:
    tts_audio.sort(key=lambda x: x[0])
    sr = tts_audio[0][2]
    out = np.concatenate([a for _, a, _ in tts_audio])
    sf.write(args.out, out, sr)
    print(f"\nwrote {args.out}: {len(out) / sr:.1f}s of English speech", flush=True)
else:
    print("\nno speech translated", flush=True)
