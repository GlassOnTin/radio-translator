# Haven-invocable translation service

How Haven (the pocket thin-client) drives local radio translation on the
workstation GPU. Fully local, no internet, no playback, receive-only.

## Compute reality

faster-whisper large-v3 needs the **workstation CUDA GPU**; Kokoro TTS runs on
CPU. The phone's PRoot guest cannot run large-v3 CUDA. So the service runs on
the **workstation** and Haven invokes it **remotely** — it is not a phone-proot
guest service.

## Chosen form: Haven `run_command` (SSH exec)

The smallest form that fits Haven's contracts. `translate_service.py` is a
one-shot CLI: `translate(audio file) -> English text + optional speech WAV`.
Haven drives it with `run_command`, which is an SSH exec channel (no PTY)
returning `{exitCode, stdout, stderr}` — exactly the mechanism demonstrated
below.

Why not the alternatives:
- **MCP server** (loopback port, multiplexed over Haven's tunnel): cleaner tool
  surface, but needs a persistent server, an extra dependency, a port, and
  Haven MCP-tunnel wiring. Larger; deferred. `run_command` is the acceptable
  first rung and is sufficient end-to-end.
- **Foreground SSH worker**: a persistent supervised process adds lifecycle
  management for no benefit over one-shot exec for occasional translation.

No loopback port is used (SSH exec, not MCP).

## Service command (canonical: run.sh + ABSOLUTE paths)

Haven `run_command` execs with **cwd = $HOME (/home/ian)** and **no login
profile**, so use `run.sh` (venv + CUDA-12 libs) and **absolute** script and
audio paths. This is the verified form:

```
/home/ian/Code/radio-translator/run.sh \
  /home/ian/Code/radio-translator/translate_service.py \
  /home/ian/Code/radio-translator/samples/franceinfo_60s.wav \
  [--speech /tmp/out.wav] [--lang xx] [--task translate|transcribe] [--json]
```

- `run.sh` sets `LD_LIBRARY_PATH` to the venv's CUDA-12 nvidia libs and execs
  `.venv/bin/python`. **Do not** call `translate_service.py` via its shebang
  (`#!/usr/bin/env python3` → system python3, which lacks soundfile and the
  CUDA-12 libs) and **do not** use relative paths (they resolve against
  /home/ian, not the project dir).
- `AUDIO_FILE`: any ffmpeg-readable file on the workstation (decoded to 16 kHz
  mono internally). Use an absolute path.
- `--speech OUT.wav`: write English Kokoro TTS to `OUT.wav` (CPU; **never
  played**). Use an absolute path.
- `--json`: stdout is a JSON object `{text, language, language_probability,
  duration_s, rtf, speech_file, task}`; otherwise stdout is the plain text.
- Progress/summary go to **stderr**; stdout is clean text/JSON.
- No `--play`, no speaker output, no network.

**Robustness:** `translate_service.py` also self-activates — if run under a
different interpreter it re-execs under `.venv/bin/python` with the CUDA-12
`LD_LIBRARY_PATH`, and it resolves a relative audio path against the project
dir. So a direct absolute call works too regardless of cwd. `run.sh` +
absolute paths remains the canonical, hard-to-get-wrong entry.

## Haven invocation

With a Haven SSH profile pointing at the workstation (`ian@192.168.0.180:22`,
key auth; a `~/.ssh/id_ed25519_sk_haven` key exists for this):

```
run_command(
  profileId = "<workstation SSH profile, e.g. msi-z790>",
  command   = "/home/ian/Code/radio-translator/run.sh "
              "/home/ian/Code/radio-translator/translate_service.py "
              "/home/ian/Code/radio-translator/samples/franceinfo_60s.wav "
              "--speech /tmp/out.wav --json",
  timeoutMs = 120000
)
```

All paths absolute; `run.sh` is the entry (venv + CUDA-12). Returns
`exitCode=0`, `stdout` = the JSON (translation text + `speech_file` path),
`stderr` = progress. To translate audio captured on the phone: upload it to the
workstation first (Haven `upload_file` / SFTP), `run_command` on that absolute
path, then fetch the speech WAV via `serve_file` / SFTP if wanted.

## End-to-end demonstration (non-login exec == Haven run_command)

Re-verified through a **non-login** SSH exec (cwd=$HOME, no profile sourcing —
the same conditions as Haven `run_command`), workstation `msi-z790`
(192.168.0.180), RTX 5090:

```
ssh ian@192.168.0.180 \
  '/home/ian/Code/radio-translator/run.sh \
   /home/ian/Code/radio-translator/translate_service.py \
   /home/ian/Code/radio-translator/samples/franceinfo_60s.wav --json'
```

exitCode 0, fr→en, RTF 0.0156 (load 1.5 s, infer 0.9 s). A direct absolute call
(no run.sh) also passes via non-login exec (self-activation), de→en RTF 0.029.

**Delivery bugs fixed (found by the conductor via real Haven run_command):**
(1) calling `translate_service.py` via its shebang used system python3 (no
soundfile / CUDA-12 libs) — fixed by self-activating the venv + CUDA libs and
by documenting `run.sh` as the entry; (2) relative paths resolved against the
exec cwd (/home/ian) — fixed by documenting absolute paths and resolving any
relative input against the project dir.

Result (stdout JSON):
```json
{"text": "And one hour, the flash, Myriam Rock Massarin.\nSpaceX on the roof of the world, ... Elon Musk signs an exploit never seen before in the history of astronomy.\nThe ship of his Starship rocket has descended to Earth after having toured the planet 14 times. ... Benjamin Netanyahu agitates the Iranian threat ...",
 "language": "fr", "language_probability": 1.0, "duration_s": 60.0,
 "rtf": 0.0163, "speech_file": "/tmp/haven_fr.wav", "task": "translate"}
```
stderr: `decoded 60.0s; loading large-v3 (cuda); wrote speech /tmp/haven_fr.wav
(35.7s, 4.5s CPU); lang=fr p=1.00 infer=1.0s RTF=0.016`. Speech WAV written
(1.7 MB, 35.7 s English, 24 kHz), not played. A German sample (`lang_de.wav`)
likewise translated de→en (p=0.997) with an 18.6 s speech WAV.

## Not verified

- **Phone-side integration**: Haven's MCP endpoint was disconnected during this
  work, so the actual `run_command` call from Haven was not exercised; the demo
  used an equivalent SSH exec channel (the same mechanism). Wiring Haven's
  workstation SSH profile and calling `run_command` from the phone remains to
  be done.
- **Live HF over-the-air**: still hardware-gated (SDRplay needs an HF antenna;
  PZSDR/AD9361 cannot reach HF). The service translates files; the RF capture
  path (`tools/sdrplay_cap.c` -> `tools/demod.py` -> PipeWire -> `live.py
  --pulse`) is documented but waits on hardware.
- **Audio transfer phone<->workstation** (upload/serve_file) not exercised here.