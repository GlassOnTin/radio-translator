# SPDX-License-Identifier: AGPL-3.0-or-later
"""Translation backends for the daemon. A text backend is duck-typed to the
faster-whisper contract used by translator.Translator: .transcribe(audio,
**kwargs) -> (segments, info), where segments carry .text/.avg_logprob/
.compression_ratio/.no_speech_prob and info carries .language/
.language_probability. This lets alternative backends plug into the existing
Translator/gates/sticky-language machinery unchanged.

Backends:
  - whisper (default): faster-whisper translate, wrapped as-is.
  - whisper-mt: whisper *transcribe* (source text) -> LLM MT -> English, for
    better idiom/coherence than whisper's built-in translate.
  - canary / seamless: torch subprocess backends (see tools/*_backend.py).
"""
import base64
import json
import os
import struct
import subprocess
import sys
import threading
import types
import urllib.request
import zlib

import numpy as np

import asr

ROOT = os.path.dirname(os.path.abspath(__file__))


def resample_ratio(x, up, down):
    """Lightweight rational resample (linear interp) for S2ST 16k->24k output;
    avoids a scipy dependency in the lean .venv."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n_out = int(len(x) * up / down)
    return np.interp(np.arange(n_out) * down / up, np.arange(len(x)), x).astype(np.float32)


def make_segment(text):
    """Synthesize faster-whisper-style segment metadata for non-whisper text so
    the production gates (compression/no-speech/repetition/CJK) still apply."""
    raw = text.encode("utf-8")
    comp = len(raw) / max(1, len(zlib.compress(raw)))
    return types.SimpleNamespace(text=text, avg_logprob=-0.2,
                                 compression_ratio=comp, no_speech_prob=0.1)


def mt_ollama(text, src_lang, model="qwen3.6:27b", host="http://localhost:11434"):
    """Translate source text to English via a local ollama model (thinking off,
    temperature 0). Returns English text."""
    prompt = (f"Translate this {src_lang or 'foreign'} radio broadcast text into "
              f"natural, idiomatic English. Output only the English translation, "
              f"nothing else.\n\n{text}")
    payload = {"model": model, "prompt": prompt, "stream": False, "think": False,
               "options": {"temperature": 0}}
    req = urllib.request.Request(host + "/api/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=180))
    return resp.get("response", "").strip()


class WhisperMTModel:
    """whisper transcribe (source) -> LLM MT -> English. info.language stays the
    detected source language (for sticky language + routing); the returned text
    is English. The CJK-echo gate doubles as an MT-failure detector."""

    def __init__(self, whisper_model, mt_model="qwen3.6:27b", mt_host="http://localhost:11434"):
        self.whisper = whisper_model
        self.mt_model = mt_model
        self.mt_host = mt_host

    def transcribe(self, audio, **kwargs):
        lang = kwargs.get("language")
        segs, info = asr.transcribe(self.whisper, audio, task="transcribe",
                                    lang=lang, profile="live")
        src = " ".join(s.text.strip() for s in segs).strip()
        if not src:
            return [], info
        english = mt_ollama(src, info.language, self.mt_model, self.mt_host)
        if not english:
            return [], info
        return [make_segment(english)], info


# Canary-1b-v2 covers these 25 European languages; others fall back to whisper.
CANARY_LANGS = {"bg", "cs", "da", "de", "et", "el", "en", "es", "fi", "fr",
                "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro",
                "sk", "sl", "sv", "uk"}


class CanaryRouter:
    """Detect source language (whisper LID); route European languages to Canary
    (fast, strong X->EN BLEU) and everything else to whisper translate."""

    def __init__(self, whisper, canary):
        self.whisper = whisper
        self.canary = canary

    def transcribe(self, audio, **kwargs):
        lang = kwargs.get("language")
        if not lang:
            lang, _prob, _all = self.whisper.detect_language(audio)
        if lang in CANARY_LANGS:
            return self.canary.transcribe(audio, language=lang)
        return asr.transcribe(self.whisper, audio, task="translate", lang=lang,
                              profile="live", no_repeat_ngram_size=4)


class SubprocessBackend:
    """Client for a torch backend running in .venv-tts as a persistent
    subprocess (Canary, SeamlessM4T). Length-prefixed JSON request/response with
    base64 audio. Implements the duck-typed .transcribe contract for text
    backends; S2ST audio is returned in resp['audio_b64']."""

    def __init__(self, script, device="cuda", extra_args=None):
        py = os.path.join(ROOT, ".venv-tts", "bin", "python")
        env = os.environ.copy()
        if not os.path.isfile(py):
            py = sys.executable
        else:
            # torch cu132 needs its own nvidia libs (cudnn engines etc.) on the path
            import glob
            nv = glob.glob(os.path.join(ROOT, ".venv-tts", "lib", "python3.*",
                                        "site-packages", "nvidia", "*", "lib"))
            if nv:
                env["LD_LIBRARY_PATH"] = ":".join(nv) + (
                    ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        self.proc = subprocess.Popen(
            [py, os.path.join(ROOT, "tools", script), "--device", device] + (extra_args or []),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        self.lock = threading.Lock()

    def request(self, obj):
        data = json.dumps(obj).encode()
        with self.lock:
            self.proc.stdin.write(struct.pack("<I", len(data)) + data)
            self.proc.stdin.flush()
            hdr = self.proc.stdout.read(4)
            if len(hdr) < 4:
                raise RuntimeError("backend subprocess closed")
            n = struct.unpack("<I", hdr)[0]
            return json.loads(self.proc.stdout.read(n))

    def transcribe(self, audio, **kwargs):
        pcm = (np.asarray(audio, dtype=np.float32) * 32767).astype(np.int16).tobytes()
        resp = self.request({"audio_b64": base64.b64encode(pcm).decode(),
                             "lang": kwargs.get("language")})
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        text = resp.get("text", "")
        info = types.SimpleNamespace(language=resp.get("language", "en"),
                                     language_probability=resp.get("language_probability", 1.0))
        if not text:
            return [], info
        return [make_segment(text)], info

    def translate_audio(self, audio, src_lang=None, tgt_lang="en"):
        pcm = (np.asarray(audio, dtype=np.float32) * 32767).astype(np.int16).tobytes()
        resp = self.request({"audio_b64": base64.b64encode(pcm).decode(),
                             "lang": src_lang, "tgt_lang": tgt_lang, "s2st": True})
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        out = base64.b64decode(resp.get("audio_b64", ""))
        return np.frombuffer(out, dtype=np.float32), resp

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.kill()
        self.proc.wait()
