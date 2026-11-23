# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared ASR helpers for live.py, translate_file.py, translate_service.py,
verify_languages.py and tools/bench_asr.py: model load, audio decode,
transcribe profiles, and pure text/segment gates. No CLI, no side effects
on import."""
import os
import re
import subprocess
from collections import Counter

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))

# live: greedy, no internal VAD (the chunker already cut at pauses).
# file: beam 5 + internal VAD, as the offline scripts always used.
PROFILES = {
    "live": dict(beam_size=1, vad_filter=False, condition_on_previous_text=False),
    "file": dict(beam_size=5, vad_filter=True, condition_on_previous_text=False),
}

def _is_cjk(c):
    o = ord(c)
    return 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF


def load_whisper(model=None, device="cuda", compute_type="float16", device_index=0,
                 cpu_threads=0, num_workers=1):
    return WhisperModel(model or os.environ.get("WHISPER_MODEL", "large-v3"),
                        device=device, compute_type=compute_type,
                        device_index=device_index, cpu_threads=cpu_threads,
                        num_workers=num_workers)


def decode_audio(path):
    """Decode any ffmpeg-readable file to 16 kHz mono float32 via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
         "-f", "s16le", "-ac", "1", "-ar", str(SR), "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm


def load_wav(path):
    """Read a 16 kHz WAV via soundfile, bypassing faster-whisper's av.open()
    path (PyAV metadata_errors breakage on some broadcast-derived files)."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == SR, f"{path}: expected {SR} Hz, got {sr}"
    return audio


def transcribe(model, audio, task="translate", lang=None, profile="file", **overrides):
    kwargs = dict(PROFILES[profile])
    kwargs["task"] = task
    if lang:
        kwargs["language"] = lang
    kwargs.update(overrides)
    return model.transcribe(audio, **kwargs)


def segment_gates(segments):
    """Count segments failing faster-whisper's own quality thresholds
    (compression_ratio 2.4, avg_logprob -1.0, no_speech_prob 0.6)."""
    flags = dict(high_compression=0, low_logprob=0, no_speech=0)
    for s in segments:
        if s.compression_ratio > 2.4:
            flags["high_compression"] += 1
        if s.avg_logprob < -1.0:
            flags["low_logprob"] += 1
        if s.no_speech_prob > 0.6:
            flags["no_speech"] += 1
    return flags


def repetition_flags(text):
    """Hallucination heuristics: any 4-gram occurring >=3 times, or any word
    repeated >=4 times consecutively. ratio = share of 4-gram occurrences
    belonging to >=3-time 4-grams."""
    words = text.split()
    grams = list(zip(words, words[1:], words[2:], words[3:]))
    counts = Counter(grams)
    rep = sum(c for c in counts.values() if c >= 3)
    max_run = run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        max_run = max(max_run, run)
    return dict(rep4=rep > 0, run4=max_run >= 4,
                ratio=rep / max(1, len(grams)), max_run=max_run)


def cjk_ratio(text):
    """Share of CJK characters among non-space chars. Translate output should
    be English; a high ratio means the model echoed the source language."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _is_cjk(c)) / len(chars)


def dedup_overlap(prev, new, min_words=4):
    """Trim a word-level suffix/prefix overlap (>=min_words) from new when
    carry-over audio made the next chunk repeat the tail of the previous."""
    pw, nw = prev.split(), new.split()
    for k in range(min(len(pw), len(nw)), min_words - 1, -1):
        if pw[-k:] == nw[:k]:
            return " ".join(nw[k:])
    return new
