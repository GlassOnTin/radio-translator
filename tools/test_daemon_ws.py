#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""WS integration check: stream a 16 kHz mono WAV to the translation daemon and
print the status + chunk events. Usage:
  tools/test_daemon_ws.py [WAV] [ws://HOST:PORT]
"""
import asyncio
import json
import sys

import numpy as np
import soundfile as sf
from websockets.asyncio.client import connect


async def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "samples/lang_zh.wav"
    url = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:8770"
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 16000, f"expected 16 kHz, got {sr}"
    pcm = (audio * 32767).astype(np.int16).tobytes()
    events = []
    returned = bytearray()
    async with connect(url, max_size=2**22) as ws:
        print("STATUS", json.loads(await ws.recv()), flush=True)
        await ws.send(json.dumps({"audio": True}))  # request filtered audio return
        block = 8000  # 0.25 s
        for i in range(0, len(pcm), block):
            await ws.send(pcm[i:i + block])
            await asyncio.sleep(0.005)
        await ws.send(json.dumps({"flush": True}))
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
            except (asyncio.TimeoutError, Exception):
                break
            if isinstance(msg, (bytes, bytearray)):
                if msg and msg[0] == 0:  # type 0 = filtered RF audio
                    returned.extend(msg[1:])
                continue
            ev = json.loads(msg)
            events.append(ev)
            if ev.get("type") == "chunk":
                print(f"CHUNK {ev['chunk']} | {ev['language']} | sup={ev['suppressed']} "
                      f"| lat={ev['latency_s']}s | {ev['text'][:90]}", flush=True)
    clean = [e for e in events if e.get("type") == "chunk" and not e["suppressed"]]
    print(f"\nevents={len(events)} clean_chunks={len(clean)} returned_audio={len(returned)/2/16000:.1f}s")
    if returned:
        a = np.frombuffer(bytes(returned), dtype=np.int16).astype(np.float32) / 32768.0
        sf.write("/tmp/returned_audio.wav", a, 16000)
        print(f"wrote /tmp/returned_audio.wav RMS {20*np.log10(np.sqrt(np.mean(a**2))+1e-12):.1f} dBFS")


if __name__ == "__main__":
    asyncio.run(main())
