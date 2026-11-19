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

## Service command

```
/home/ian/Code/radio-translator/.venv/bin/python \
  /home/ian/Code/radio-translator/translate_service.py \
  <AUDIO_FILE> [--speech <OUT.wav>] [--lang xx] [--task translate|transcribe] [--json]
```

- `AUDIO_FILE`: any ffmpeg-readable file on the workstation (decoded to 16 kHz
  mono internally).
- `--speech OUT.wav`: write English Kokoro TTS to `OUT.wav` (CPU; **never
  played**).
- `--json`: stdout is a JSON object `{text, language, language_probability,
  duration_s, rtf, speech_file, task}`; otherwise stdout is the plain text.
- Progress/summary go to **stderr**; stdout is clean text/JSON.
- The script self-bootstraps `LD_LIBRARY_PATH` for the CUDA-12 libs
  (libcublas.so.12) before importing ctranslate2, so no `run.sh` wrapper is
  needed for remote exec.
- No `--play`, no speaker output, no network.

## Haven invocation

With a Haven SSH profile pointing at the workstation (`ian@192.168.0.180:22`,
key auth; a `~/.ssh/id_ed25519_sk_haven` key exists for this):

```
run_command(
  profileId = "<workstation SSH profile>",
  command   = "/home/ian/Code/radio-translator/.venv/bin/python "
              "/home/ian/Code/radio-translator/translate_service.py "
              "/path/on/workstation/audio.wav --speech /tmp/out.wav --json",
  timeoutMs = 120000
)
```

Returns `exitCode=0`, `stdout` = the JSON (translation text + `speech_file`
path), `stderr` = progress. To translate audio captured on the phone: upload it
to the workstation first (Haven `upload_file` / SFTP), `run_command` on that
path, then fetch the speech WAV via `serve_file` / SFTP if wanted.

## End-to-end demonstration (Haven-style SSH exec)

Run from a remote SSH exec channel (the `run_command` mechanism), workstation
`msi-z790` (192.168.0.180), RTX 5090:

```
ssh ian@192.168.0.180 \
  '/home/ian/Code/radio-translator/.venv/bin/python \
   /home/ian/Code/radio-translator/translate_service.py \
   /home/ian/Code/radio-translator/samples/franceinfo_60s.wav \
   --speech /tmp/haven_fr.wav --json'
```

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