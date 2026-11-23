#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""WebSocket translation daemon for SDR front-end plugins (e.g. the SDR++
translator module). Streams of raw s16le 16 kHz mono PCM in, JSON chunk events
out. Reuses the live pipeline: silero chunker -> faster-whisper translate ->
hallucination gates. One backend across compute tiers; the tier is auto-probed
(CUDA + free VRAM -> large-v3 fp16; otherwise CPU small int8) and advertised at
handshake, so a thin plugin on a laptop can use a local CPU tier or point at a
remote GPU daemon.

Protocol:
  client -> server: binary s16le 16 kHz mono PCM (any block size);
                    optional JSON text control {"lang": "zh"}.
  server -> client: {"type":"status", tier, model, device, compute_type} once;
                    {"type":"chunk", chunk, language, language_probability,
                     text, dur, reason, infer_s, latency_s, suppressed, reasons}.

Run: ./run.sh tools/translate_daemon.py --port 8770
     (CPU tier) ./run.sh tools/translate_daemon.py --device cpu --model small
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _bootstrap():
    """Re-exec under .venv with the CUDA-12 libs (same as translate_service).
    On CPU-only installs the nvidia glob is empty and this degrades to just the
    venv re-exec."""
    venv_py = os.path.join(ROOT, ".venv", "bin", "python")
    nv = glob.glob(os.path.join(ROOT, ".venv", "lib", "python3.*", "site-packages", "nvidia"))
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
import asyncio  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import websockets  # noqa: E402
from websockets.asyncio.server import serve  # noqa: E402

sys.path.insert(0, ROOT)
import asr  # noqa: E402
import chunker  # noqa: E402
from translator import Translator  # noqa: E402
from vad import SileroVad  # noqa: E402


def gpu_free_mb(physical_index):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
             "-i", str(physical_index)], capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def detect_tier(device_index=0):
    """Pick a compute tier. CUDA device_index maps through CUDA_VISIBLE_DEVICES
    to a physical GPU for the VRAM probe."""
    import ctranslate2
    try:
        n = ctranslate2.get_cuda_device_count()
    except Exception:
        n = 0
    if n > device_index:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        phys = int(vis.split(",")[device_index]) if vis else device_index
        free = gpu_free_mb(phys)
        if free is None or free >= 6000:
            return dict(tier="gpu", model="large-v3", device="cuda", compute_type="float16")
        if free >= 2000:
            return dict(tier="gpu-low", model="large-v3", device="cuda", compute_type="int8_float16")
    # medium int8 translates cleanly (small echoes CJK on zh, measured);
    # fall back to small on low-core machines where medium would lag.
    cores = os.cpu_count() or 4
    cpu_model = "medium" if cores >= 8 else "small"
    return dict(tier="cpu", model=cpu_model, device="cpu", compute_type="int8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--model", default=None, help="override tier model")
    ap.add_argument("--compute-type", default=None)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--cpu-threads", type=int, default=0)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--chunk", type=float, default=12.0)
    ap.add_argument("--silence", type=int, default=300)
    ap.add_argument("--no-gates", action="store_true")
    ap.add_argument("--lang-window", type=int, default=5,
                    help="rolling window for sticky (mode) language detection")
    ap.add_argument("--enhance", choices=["none", "isolate"], default="none")
    ap.add_argument("--enhance-device", default="cuda")
    ap.add_argument("--record", default=None, help="dump received PCM to this WAV (diagnostics)")
    ap.add_argument("--audio-out", action="store_true",
                    help="return processed audio to the client for playback (client can toggle)")
    ap.add_argument("--tts", action="store_true", help="synthesize English Kokoro TTS per chunk")
    ap.add_argument("--voice", default="af_heart", help="Kokoro voice for TTS")
    ap.add_argument("--speed", type=float, default=1.0, help="Kokoro TTS speech rate")
    ap.add_argument("--backend", choices=["whisper", "whisper-mt", "canary", "seamless"],
                    default="whisper", help="translation backend")
    ap.add_argument("--mt-model", default="qwen3.6:27b", help="ollama model for whisper-mt")
    args = ap.parse_args()

    tier = detect_tier(args.device_index)
    if args.device != "auto":
        tier["device"] = args.device
        tier["tier"] = args.device
        if args.device == "cpu":
            tier["model"] = args.model or "small"
            tier["compute_type"] = args.compute_type or "int8"
        else:
            tier["model"] = args.model or "large-v3"
            tier["compute_type"] = args.compute_type or "float16"
    if args.model:
        tier["model"] = args.model
    if args.compute_type:
        tier["compute_type"] = args.compute_type

    print(f"[daemon] tier={tier['tier']} model={tier['model']} device={tier['device']} "
          f"compute_type={tier['compute_type']}", flush=True)
    base_model = asr.load_whisper(tier["model"], device=tier["device"],
                                  compute_type=tier["compute_type"],
                                  device_index=args.device_index, cpu_threads=args.cpu_threads)
    model_lock = threading.Lock()
    # torch backends (canary/seamless) are cached subprocesses, spawned on demand
    _sub_cache = {}
    _sub_lock = threading.Lock()

    def get_sub(script):
        with _sub_lock:
            if script not in _sub_cache:
                from backends import SubprocessBackend
                _sub_cache[script] = SubprocessBackend(script, device=tier["device"])
            return _sub_cache[script]

    def make_backend(name):
        """Return (translator_model, s2st_subprocess) for a backend name."""
        from backends import WhisperMTModel, CanaryRouter
        if name == "whisper-mt":
            return WhisperMTModel(base_model, mt_model=args.mt_model), None
        if name == "canary":
            sb = get_sub("canary_backend.py")
            return CanaryRouter(base_model, sb), sb
        if name == "seamless":
            sb = get_sub("seamless_backend.py")
            return base_model, sb
        return base_model, None
    # Kokoro loads lazily on first TTS use so CPU/no-TTS installs start without
    # the (gitignored, downloaded) model files.
    kokoro = None
    kokoro_lock = threading.Lock()

    def get_kokoro():
        nonlocal kokoro
        if kokoro is None:
            with kokoro_lock:
                if kokoro is None:
                    from kokoro_onnx import Kokoro
                    kokoro = Kokoro(os.path.join(ROOT, "models", "kokoro-v1.0.onnx"),
                                    os.path.join(ROOT, "models", "voices-v1.0.bin"))
        return kokoro

    status = dict(type="status", protocol_version=1, tier=tier["tier"],
                  model=tier["model"], device=tier["device"],
                  compute_type=tier["compute_type"], backend=args.backend)
    tts_py = os.path.join(ROOT, ".venv-tts", "bin", "python")
    if not os.path.isfile(tts_py):
        tts_py = sys.executable  # e.g. Docker GPU image: torch in the main python

    async def handler(ws):
        print("[daemon] client connected", flush=True)
        backend_name = args.backend
        m0, sub_backend = make_backend(backend_name)
        trans = Translator(m0, lang=args.lang, gates=not args.no_gates,
                           lang_window=args.lang_window)

        def set_backend(name):
            nonlocal backend_name, sub_backend
            m, sb = make_backend(name)
            trans.model = m
            sub_backend = sb
            backend_name = name
            print(f"[daemon] backend -> {name}", flush=True)

        ck = chunker.Chunker(vad=SileroVad(), silence_ms=args.silence, cap_s=args.chunk)
        loop = asyncio.get_running_loop()
        q = asyncio.Queue()
        iso_q = asyncio.Queue()
        use_iso = args.enhance == "isolate"
        audio_out = args.audio_out
        tts_enabled = args.tts
        voice = args.voice
        tts_speed = args.speed
        iso_proc = None
        iso_reader = None
        if args.enhance == "isolate":
            iso_proc = subprocess.Popen(
                [tts_py, os.path.join(HERE, "isolate_stream.py"),
                 "--device", args.enhance_device],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE)

            def iso_read():
                while True:
                    data = iso_proc.stdout.read(960)
                    if not data:
                        break
                    loop.call_soon_threadsafe(iso_q.put_nowait, data)
                loop.call_soon_threadsafe(iso_q.put_nowait, None)
            iso_reader = threading.Thread(target=iso_read, daemon=True)
            iso_reader.start()

        await ws.send(json.dumps({**status, "backend": backend_name}))

        async def worker():
            while True:
                item = await q.get()
                if item is None:
                    return
                a, m = item

                def proc():
                    with model_lock:
                        return trans.process_chunk(a, m)
                ev = await asyncio.to_thread(proc)
                if ev is not None:
                    print(f"[chunk {ev['chunk']} | {ev['language']} | {ev['dur']:.1f}s | "
                          f"lat {ev['latency_s']:.1f}s | sup={ev['suppressed']}] {ev['text'][:90]}",
                          flush=True)
                    try:
                        await ws.send(json.dumps(ev, ensure_ascii=False))
                    except websockets.ConnectionClosed:
                        return
                    if (not ev["suppressed"] and ev["text"]
                            and backend_name == "seamless" and sub_backend is not None):
                        def s2st():
                            from backends import resample_ratio
                            wav, _ = sub_backend.translate_audio(a, src_lang=ev["language_raw"])
                            return resample_ratio(wav, 3, 2).tobytes()  # 16k -> 24k
                        try:
                            sb = await asyncio.to_thread(s2st)
                            for i in range(0, len(sb), 4000):
                                await ws.send(b"\x01" + sb[i:i + 4000])
                        except websockets.ConnectionClosed:
                            return
                        except Exception as e:
                            print(f"[daemon] s2st error: {e}", flush=True)
                    elif tts_enabled and not ev["suppressed"] and ev["text"]:
                        def synth():
                            samples, _ = get_kokoro().create(ev["text"], voice=voice,
                                                             speed=tts_speed, lang="en-us")
                            return np.asarray(samples, dtype=np.float32).tobytes()
                        try:
                            tts_bytes = await asyncio.to_thread(synth)
                            # small frames: libwebsockets fragments messages
                            # > its rx buffer, which would corrupt the type byte
                            for i in range(0, len(tts_bytes), 4000):
                                await ws.send(b"\x01" + tts_bytes[i:i + 4000])
                        except websockets.ConnectionClosed:
                            return
                        except Exception as e:
                            print(f"[daemon] tts error: {e}", flush=True)

        async def dispatcher():
            # isolated PCM path: feed the chunker and (optionally) return audio
            while True:
                data = await iso_q.get()
                if data is None:
                    return
                if use_iso:
                    x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    for a, m in ck.feed(x):
                        q.put_nowait((a, m))
                    if audio_out:
                        try:
                            await ws.send(b"\x00" + data)  # type 0 = filtered RF (s16le 16k mono)
                        except websockets.ConnectionClosed:
                            return

        wt = asyncio.create_task(worker())
        dt = asyncio.create_task(dispatcher()) if iso_proc is not None else None
        rx_bytes = 0
        rec_file = None
        if args.record:
            import soundfile as sf
            rec_file = sf.SoundFile(args.record, "w", samplerate=16000,
                                    channels=1, subtype="FLOAT")
        try:
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    rx_bytes += len(msg)
                    if rec_file is not None:
                        rec_file.write(np.frombuffer(msg, dtype=np.int16).astype(np.float32) / 32768.0)
                    if iso_proc is not None:
                        await asyncio.to_thread(iso_proc.stdin.write, bytes(msg))
                    if not use_iso:
                        x = np.frombuffer(msg, dtype=np.int16).astype(np.float32) / 32768.0
                        for a, m in ck.feed(x):
                            q.put_nowait((a, m))
                        if audio_out:
                            await ws.send(b"\x00" + bytes(msg))
                elif isinstance(msg, str):
                    try:
                        ctrl = json.loads(msg)
                    except json.JSONDecodeError:
                        continue
                    if "backend" in ctrl and ctrl["backend"] != backend_name:
                        await asyncio.to_thread(set_backend, ctrl["backend"])
                        await ws.send(json.dumps({**status, "backend": backend_name}))
                    if "lang" in ctrl:
                        trans.lang = ctrl["lang"] or None
                    if "chunk" in ctrl or "silence" in ctrl:
                        chunk_s = float(ctrl.get("chunk", args.chunk))
                        silence_ms = int(ctrl.get("silence", args.silence))
                        ck = chunker.Chunker(vad=SileroVad(), silence_ms=silence_ms,
                                             cap_s=chunk_s)
                        print(f"[daemon] rechunk cap={chunk_s:.1f}s silence={silence_ms}ms", flush=True)
                    if "enhance" in ctrl:
                        use_iso = (ctrl["enhance"] == "isolate") and iso_proc is not None
                    if "audio" in ctrl:
                        audio_out = bool(ctrl["audio"])
                    if "tts" in ctrl:
                        tts_enabled = bool(ctrl["tts"])
                    if "voice" in ctrl:
                        voice = ctrl["voice"] or args.voice
                    if "speed" in ctrl:
                        tts_speed = float(ctrl["speed"])
                    if "gates" in ctrl:
                        trans.gates = bool(ctrl["gates"])
                    if ctrl.get("flush"):
                        for a, m in ck.flush():
                            q.put_nowait((a, m))
        except websockets.ConnectionClosed:
            pass
        finally:
            print(f"[daemon] client disconnected, received {rx_bytes/2/16000:.1f}s audio, "
                  f"chunks={trans.idx} suppressed={trans.suppressed}", flush=True)
            if rec_file is not None:
                rec_file.close()
                print(f"[daemon] recorded {args.record}: {rx_bytes/2/16000:.1f}s (continuous)", flush=True)
            if iso_proc is not None:
                try:
                    iso_proc.stdin.close()
                except BrokenPipeError:
                    pass
                if iso_reader is not None:
                    iso_reader.join(timeout=10)
                iso_proc.kill()
                iso_proc.wait()
            for a, m in ck.flush():
                q.put_nowait((a, m))
            q.put_nowait(None)
            if dt is not None:
                try:
                    await asyncio.wait_for(dt, timeout=10)
                except asyncio.TimeoutError:
                    dt.cancel()
            try:
                await asyncio.wait_for(wt, timeout=30)
            except asyncio.TimeoutError:
                wt.cancel()

    async def run():
        async with serve(handler, args.host, args.port, max_size=2**22):
            print(f"[daemon] listening on ws://{args.host}:{args.port}", flush=True)
            await asyncio.Future()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[daemon] stopped", flush=True)


if __name__ == "__main__":
    main()
