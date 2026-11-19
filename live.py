# SPDX-License-Identifier: AGPL-3.0-or-later
"""Live foreign radio -> English text + English voice.

ffmpeg pulls an input (web stream, local file, ALSA or PipeWire source) as
16 kHz mono PCM; webrtcvad cuts speech chunks at pauses or a max length;
faster-whisper translates each chunk on GPU. Language is auto-detected per
chunk, so retuning across stations on SW works (pin one with --lang). Kokoro
speaks the English on CPU in a worker thread.

Text -> stdout. Combined English audio -> WAV. Optional --play feeds audio to
ffplay as it is generated; OFF by default, so nothing reaches the speakers
unless you ask for it.

Usage:
  python live.py STREAM_URL [--seconds 90] [--out live_english.wav]
                 [--chunk 12] [--silence 450] [--lang xx] [--play]
                 [--voice af_heart] [--pulse SOURCE | --alsa DEVICE]

Lower --chunk (e.g. 6) cuts text latency at some coherence cost. For RF later:
SDR++/SDRangel audio -> a PipeWire virtual sink, then --pulse <sink>.monitor;
or a receiver's line-out into the onboard capture, then --alsa hw:2,0.
"""
import argparse
import os
import queue
import subprocess
import threading
import time

import numpy as np
import soundfile as sf
import webrtcvad
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

MODELS = "/home/ian/Code/radio-translator/models"
SR = 16000
FRAME_MS = 30
FRAME_BYTES = SR * FRAME_MS // 1000 * 2  # 480 samples, s16le
MIN_CHUNK_S = 2.0

ap = argparse.ArgumentParser(description="Live foreign radio -> English text + voice")
ap.add_argument("input", nargs="?", help="stream URL or local file")
ap.add_argument("--seconds", type=float, default=90.0, help="how long to listen")
ap.add_argument("--out", default="live_english.wav", help="combined English WAV")
ap.add_argument("--chunk", type=float, default=12.0, help="max chunk seconds (lower = less latency)")
ap.add_argument("--silence", type=int, default=450, help="trailing silence (ms) that closes a chunk")
ap.add_argument("--lang", default=None, help="force source language (default: auto per chunk)")
ap.add_argument("--play", action="store_true", help="play English audio via ffplay (default off)")
ap.add_argument("--voice", default="af_heart", help="Kokoro voice")
ap.add_argument("--pulse", default=None, help="PipeWire/Pulse source to capture (e.g. a virtual-sink monitor)")
ap.add_argument("--alsa", default=None, help="ALSA capture device (e.g. hw:2,0)")
args = ap.parse_args()

if not args.input and not args.pulse and not args.alsa:
    ap.error("give an input URL/file, or --pulse/--alsa")

# large-v3 (not turbo): turbo's translate task echoes the source language.
MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
print(f"loading models ({MODEL})...", flush=True)
t_load = time.time()
model = WhisperModel(MODEL, device="cuda", compute_type="float16")
kok = Kokoro(f"{MODELS}/kokoro-v1.0.onnx", f"{MODELS}/voices-v1.0.bin")
vad = webrtcvad.Vad(2)
print(f"models loaded in {time.time() - t_load:.1f}s", flush=True)

if args.pulse:
    in_args, src = ["-f", "pulse", "-i", args.pulse], f"pulse:{args.pulse}"
elif args.alsa:
    in_args, src = ["-f", "alsa", "-i", args.alsa], f"alsa:{args.alsa}"
else:
    in_args, src = ["-i", args.input], args.input

proc = subprocess.Popen(
    ["ffmpeg", "-hide_banner", "-loglevel", "error", *in_args,
     "-f", "s16le", "-ac", "1", "-ar", str(SR), "pipe:1"],
    stdout=subprocess.PIPE,
)

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


def tts_worker():
    while True:
        item = tts_q.get()
        if item is None:
            break
        idx, text = item
        t0 = time.time()
        samples, sr = kok.create(text, voice=args.voice, speed=1.0, lang="en-us")
        tts_audio.append((idx, samples, sr))
        if play_proc is not None:
            try:
                play_proc.stdin.write(np.asarray(samples, dtype=np.float32).tobytes())
                play_proc.stdin.flush()
            except BrokenPipeError:
                pass
        print(f"  [voice {idx}] {len(samples) / sr:.1f}s audio in {time.time() - t0:.1f}s", flush=True)
    tts_done.set()


threading.Thread(target=tts_worker, daemon=True).start()


def read_exact(n):
    buf = bytearray()
    while len(buf) < n:
        part = proc.stdout.read(n - len(buf))
        if not part:
            return None
        buf.extend(part)
    return bytes(buf)


chunk_frames = []
chunk_start_wall = None
silence_run = 0
idx = 0
t_start = time.time()


def flush_chunk(reason):
    global chunk_frames, chunk_start_wall, silence_run, idx
    if not chunk_frames:
        return
    pcm = b"".join(chunk_frames)
    dur = len(pcm) / 2 / SR
    started = chunk_start_wall
    chunk_frames, chunk_start_wall, silence_run = [], None, 0
    if dur < MIN_CHUNK_S:
        return
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    t0 = time.time()
    kwargs = dict(task="translate", beam_size=1,
                  condition_on_previous_text=False, vad_filter=False)
    if args.lang:
        kwargs["language"] = args.lang
    segments, info = model.transcribe(audio, **kwargs)
    text = " ".join(s.text.strip() for s in segments).strip()
    infer = time.time() - t0
    if text:
        idx += 1
        latency = time.time() - started  # audio-start -> text emitted
        print(
            f"[chunk {idx} | {info.language} | {dur:.1f}s | {reason} | infer {infer:.1f}s "
            f"| latency {latency:.1f}s] {text}",
            flush=True,
        )
        tts_q.put((idx, text))


print(f"streaming {src} for {args.seconds:.0f}s (chunk<={args.chunk:.0f}s, play={args.play}) ...", flush=True)
while time.time() - t_start < args.seconds:
    data = read_exact(FRAME_BYTES)
    if data is None:
        print("stream ended", flush=True)
        break
    is_speech = vad.is_speech(data, SR)
    if chunk_start_wall is None:
        if is_speech:
            chunk_start_wall = time.time()
            chunk_frames = [data]
        continue
    chunk_frames.append(data)
    silence_run = 0 if is_speech else silence_run + FRAME_MS
    chunk_dur = len(chunk_frames) * FRAME_MS / 1000
    if silence_run >= args.silence:
        flush_chunk("pause")
    elif chunk_dur >= args.chunk:
        flush_chunk("maxlen")

flush_chunk("end")
proc.kill()
tts_q.put(None)
tts_done.wait(timeout=120)

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