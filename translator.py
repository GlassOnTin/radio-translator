# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared translation core for live.py and the WebSocket daemon: transcribe a
speech chunk, apply hallucination gates, dedup carry-over, and return a chunk
event dict. Per-session state (chunk index, previous text, suppressed count)
lives here so each connection/stream is independent. No I/O — the caller
prints/queues (live.py) or sends JSON (daemon)."""
import time
from collections import Counter, deque

import asr


class Translator:
    def __init__(self, model, lang=None, gates=True, carry=0.0, lang_window=5):
        self.model = model
        self.lang = lang
        self.gates = gates
        self.carry = carry
        self.idx = 0
        self.prev_text = ""
        self.suppressed = 0
        # sticky language: rolling mode over recent chunks so noisy bursts
        # don't flip the reported language; per-chunk detection stays unpinned
        # so a genuine station language change still adapts within the window.
        self.lang_history = deque(maxlen=lang_window)
        self.sticky_lang = None

    def gate_reasons(self, segments, text):
        """Hallucination gates: repetition loops, CJK echo of the source,
        high compression, or every segment marked no-speech. Low avg_logprob
        alone is logged but not suppressed (noisy radio can still be real)."""
        why = []
        rep = asr.repetition_flags(text)
        if rep["rep4"] or rep["run4"]:
            why.append("repetition")
        if asr.cjk_ratio(text) > 0.3:
            why.append("cjk-echo")
        flags = asr.segment_gates(segments)
        if flags["high_compression"]:
            why.append("compression")
        if segments and flags["no_speech"] == len(segments):
            why.append("no-speech")
        return why, flags

    def process_chunk(self, audio, meta):
        """Return a chunk event dict, or None for empty/fully-deduped chunks.
        meta comes from chunker.Chunker: {dur, reason, start}."""
        self.idx += 1
        dur, reason, started = meta["dur"], meta["reason"], meta["start"]
        t0 = time.time()
        overrides = {"no_repeat_ngram_size": 4} if self.gates else {}
        segments, info = asr.transcribe(self.model, audio, task="translate",
                                        lang=self.lang, profile="live", **overrides)
        segments = list(segments)
        self.lang_history.append(info.language)
        self.sticky_lang = Counter(self.lang_history).most_common(1)[0][0]
        text = " ".join(s.text.strip() for s in segments).strip()
        infer = time.time() - t0
        if not text:
            return None
        if self.carry > 0:
            text = asr.dedup_overlap(self.prev_text, text)
            if not text:
                return None
        why, _ = self.gate_reasons(segments, text)
        latency = time.time() - started  # first-speech-frame -> text emitted
        suppressed = bool(why) and self.gates
        if suppressed:
            self.suppressed += 1
        else:
            self.prev_text = text
        return dict(type="chunk", chunk=self.idx, language=self.sticky_lang,
                    language_raw=info.language,
                    language_probability=round(info.language_probability, 3),
                    text=text, dur=round(dur, 2), reason=reason,
                    infer_s=round(infer, 3), latency_s=round(latency, 3),
                    suppressed=suppressed, reasons=why)
