#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scriptable control of SDRconnect via the official WebSocket API (1.0.3).

Works against SDRconnect_headless (WS enabled by default, port 5454) or the
GUI with Options > WebSocket Interface enabled. Replaces GUI clicking:
tune, read signal metrics, stream demodulated audio. The SDRplay API is
exclusive, so only one SDRconnect instance may hold the RSP at a time.

Binary framing: 2-byte little-endian uint16 payload type, then payload.
Type 1 = signed 16-bit PCM stereo 48 kHz LRLR (primary demodulated audio).

CLI (run via ./run.sh so the venv provides websockets/numpy/soundfile):
  tools/sdrconnect_ctl.py status
  tools/sdrconnect_ctl.py get PROPERTY
  tools/sdrconnect_ctl.py set PROPERTY VALUE
  tools/sdrconnect_ctl.py event EVENT_TYPE VALUE   # e.g. selected_device_serial 24030AB670
  tools/sdrconnect_ctl.py tune FREQ_HZ [--demod AM] [--bw 11000] [--antenna C]
  tools/sdrconnect_ctl.py record SECONDS OUT.wav [--freq FREQ_HZ] [...]
  tools/sdrconnect_ctl.py audio          # 16 kHz mono s16le to stdout, until SIGINT
"""
import argparse
import asyncio
import json
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import soundfile as sf
from websockets.asyncio.client import connect

URL = "ws://127.0.0.1:5454/"
AUDIO_TYPE = 1          # primary demodulated audio
AUDIO_SR = 48000        # Hz, stereo s16le LRLR
OUT_SR = 16000          # what the ASR pipeline wants

STATUS_PROPS = [
    "api_version", "valid_devices", "active_device", "started", "can_control",
    "valid_antennas", "active_antenna", "device_vfo_frequency",
    "device_center_frequency", "device_sample_rate", "demodulator",
    "filter_bandwidth", "lna_state", "agc_enable", "audio_volume_percent",
    "audio_mute", "signal_power", "signal_snr", "overload",
]


class SdrConnect:
    """One WebSocket connection; request/response plus audio streaming."""

    def __init__(self, url=URL):
        self.url = url
        self.ws = None

    async def __aenter__(self):
        self.ws = await connect(self.url, max_size=2**22, open_timeout=10)
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()

    async def send(self, event_type, value="", prop="", device=None):
        msg = {"event_type": event_type, "property": prop, "value": str(value)}
        if device:
            msg["device"] = device
        await self.ws.send(json.dumps(msg))

    async def get(self, prop, timeout=5.0):
        """Request a property; return its string value, or None on timeout."""
        await self.send("get_property", prop=prop)
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=left)
            except asyncio.TimeoutError:
                return None
            if isinstance(raw, bytes):
                continue  # binary stream data, not a response
            m = json.loads(raw)
            if m.get("event_type") == "get_property_response" and m.get("property") == prop:
                return m.get("value")

    async def set(self, prop, value):
        await self.send("set_property", value, prop=prop)

    async def audio_on(self):
        await self.send("audio_stream_enable", "true")

    async def audio_off(self):
        await self.send("audio_stream_enable", "false")

    async def tune(self, freq_hz, demod=None, bw=None, antenna=None, settle=1.0):
        if antenna:
            await self.set("active_antenna", antenna)
        if demod:
            await self.set("demodulator", demod)
        if bw:
            await self.set("filter_bandwidth", str(bw))
        await self.set("device_vfo_frequency", str(int(freq_hz)))
        await asyncio.sleep(settle)

    async def collect_audio(self, seconds):
        """Enable the audio stream, collect type-1 payloads for `seconds`
        wall-clock, disable it. Returns raw s16le stereo 48 kHz bytes."""
        await self.audio_on()
        buf = bytearray()
        deadline = time.monotonic() + seconds
        try:
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(self.ws.recv(), timeout=left)
                except asyncio.TimeoutError:
                    break
                if isinstance(raw, bytes) and len(raw) > 2:
                    ptype = struct.unpack("<H", raw[:2])[0]
                    if ptype == AUDIO_TYPE:
                        buf.extend(raw[2:])
        finally:
            await self.audio_off()
        return bytes(buf)


def raw_to_wav(raw, path):
    """s16le stereo 48 kHz bytes -> 16 kHz mono WAV via ffmpeg. Returns stats."""
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "s16le", "-ar", str(AUDIO_SR), "-ac", "2", "-i", "pipe:0",
         "-f", "wav", "-ar", str(OUT_SR), "-ac", "1", path],
        input=raw, check=True)
    audio, sr = sf.read(path, dtype="float32")
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    dbfs = 20 * np.log10(rms) if rms > 0 else -120.0
    return len(audio) / sr, dbfs, peak


async def cmd_status(args):
    async with SdrConnect(args.url) as s:
        for p in STATUS_PROPS:
            v = await s.get(p, timeout=3.0)
            print(f"{p:26s} = {v if v is not None else '(no response)'}")


async def cmd_get(args):
    async with SdrConnect(args.url) as s:
        v = await s.get(args.prop)
        print(v if v is not None else "(no response)")


async def cmd_set(args):
    async with SdrConnect(args.url) as s:
        await s.set(args.prop, args.value)
        await asyncio.sleep(0.3)
        print(f"{args.prop} = {await s.get(args.prop)}")


async def cmd_event(args):
    async with SdrConnect(args.url) as s:
        await s.send(args.event_type, args.value)
        await asyncio.sleep(1.0)
        for p in ("active_device", "started", "can_control", "active_antenna"):
            print(f"{p:26s} = {await s.get(p)}")


async def cmd_tune(args):
    async with SdrConnect(args.url) as s:
        await s.tune(args.freq, demod=args.demod, bw=args.bw, antenna=args.antenna)
        for p in ("device_vfo_frequency", "demodulator", "active_antenna",
                  "signal_power", "signal_snr"):
            print(f"{p:26s} = {await s.get(p)}")


async def cmd_record(args):
    async with SdrConnect(args.url) as s:
        if args.freq:
            await s.tune(args.freq, demod=args.demod, bw=args.bw,
                         antenna=args.antenna)
            print(f"tuned {args.freq} Hz", file=sys.stderr)
        raw = await s.collect_audio(args.seconds)
    if not raw:
        print("error: no audio frames received (device started? antenna?)",
              file=sys.stderr)
        return 1
    dur, dbfs, peak = raw_to_wav(raw, args.out)
    print(f"{args.out}: {dur:.1f}s, RMS {dbfs:.1f} dBFS, peak {peak:.3f}")
    return 0


async def cmd_audio(args):
    """Continuous 16 kHz mono s16le on stdout (for live.py --stdin)."""
    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "s16le", "-ar", str(AUDIO_SR), "-ac", "2", "-i", "pipe:0",
         "-f", "s16le", "-ar", str(OUT_SR), "-ac", "1", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    threading.Thread(target=shutil.copyfileobj,
                     args=(ff.stdout, sys.stdout.buffer), daemon=True).start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    async with SdrConnect(args.url) as s:
        await s.audio_on()
        try:
            while not stop.is_set():
                try:
                    raw = await asyncio.wait_for(s.ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if isinstance(raw, bytes) and len(raw) > 2:
                    if struct.unpack("<H", raw[:2])[0] == AUDIO_TYPE:
                        await loop.run_in_executor(None, ff.stdin.write, raw[2:])
        finally:
            await s.audio_off()
            ff.stdin.close()
            ff.wait()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=URL)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="dump key properties")
    g = sub.add_parser("get", help="read one property")
    g.add_argument("prop")
    st = sub.add_parser("set", help="write one property, read it back")
    st.add_argument("prop")
    st.add_argument("value")
    e = sub.add_parser("event", help="send a raw client event (e.g. selected_device 0)")
    e.add_argument("event_type")
    e.add_argument("value")
    t = sub.add_parser("tune", help="set VFO frequency (+ optional demod/bw/antenna)")
    t.add_argument("freq", type=int)
    t.add_argument("--demod")
    t.add_argument("--bw", type=int)
    t.add_argument("--antenna")
    r = sub.add_parser("record", help="record demodulated audio to a 16 kHz mono WAV")
    r.add_argument("seconds", type=float)
    r.add_argument("out")
    r.add_argument("--freq", type=int)
    r.add_argument("--demod")
    r.add_argument("--bw", type=int)
    r.add_argument("--antenna")
    sub.add_parser("audio", help="stream 16 kHz mono s16le to stdout until SIGINT")
    args = ap.parse_args()
    fn = {"status": cmd_status, "get": cmd_get, "set": cmd_set,
          "event": cmd_event, "tune": cmd_tune, "record": cmd_record,
          "audio": cmd_audio}[args.cmd]
    sys.exit(asyncio.run(fn(args)) or 0)


if __name__ == "__main__":
    main()
