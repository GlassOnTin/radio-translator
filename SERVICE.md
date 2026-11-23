# Haven-invocable translation service

How Haven (the pocket thin-client) drives local radio translation on a
workstation GPU. Fully local, no internet, no playback, receive-only.

Paths below use `/opt/radio-translator` as the example install directory;
replace it with the absolute path where you cloned this repo.

## Compute reality

faster-whisper large-v3 needs a CUDA GPU (the workstation). Kokoro TTS runs on
CPU. The phone's PRoot guest cannot run large-v3 CUDA, so the service runs on
the workstation and Haven invokes it remotely. It is not a phone-proot guest
service.

## Chosen form: Haven run_command (SSH exec)

The smallest form that fits Haven's contracts. `translate_service.py` is a
one-shot CLI: `translate(audio file) -> English text + optional speech WAV`.
Haven drives it with `run_command`, an SSH exec channel (no PTY) returning
`{exitCode, stdout, stderr}`.

Why not the alternatives:
- MCP server (loopback port over Haven's tunnel): cleaner tool surface, but a
  persistent server, an extra dependency, a port, and tunnel wiring. Larger;
  deferred. run_command is sufficient end-to-end.
- Foreground SSH worker: lifecycle overhead for no gain on occasional
  translation.

No loopback port is used (SSH exec, not MCP).

## Service command (canonical: run.sh + ABSOLUTE paths)

Haven `run_command` execs with cwd = the user's home dir and no login profile.
Use `run.sh` (sets up the venv + CUDA-12 libs) and absolute script and audio
paths:

```
/opt/radio-translator/run.sh \
  /opt/radio-translator/translate_service.py \
  /opt/radio-translator/samples/franceinfo_60s.wav \
  [--speech /tmp/out.wav] [--lang xx] [--task translate|transcribe] [--json]
```

- `run.sh` puts the venv's CUDA-12 nvidia libs (libcublas.so.12) on
  LD_LIBRARY_PATH and execs `.venv/bin/python`. Do not call
  `translate_service.py` through its shebang (system python3 lacks soundfile
  and the CUDA-12 libs), and do not use relative paths (they resolve against
  the home dir, not the project).
- audio: any ffmpeg-readable file on the workstation, decoded to 16 kHz mono.
  Absolute path.
- `--speech OUT.wav`: write English Kokoro TTS to OUT.wav (CPU, never played).
  Absolute path.
- `--json`: stdout is `{text, language, language_probability, duration_s, rtf,
  speech_file, task}`; otherwise plain text. Progress goes to stderr.
- No --play, no speaker output, no network.

Robustness: `translate_service.py` also self-activates. Run under another
interpreter, it re-execs under `.venv/bin/python` with the CUDA-12
LD_LIBRARY_PATH; a relative audio path is resolved against the project dir. A
direct absolute call therefore works regardless of cwd. run.sh + absolute
paths is still the canonical entry.

## Haven invocation

With a Haven SSH profile for the workstation (key auth):

```
run_command(
  profileId = "<workstation SSH profile>",
  command   = "/opt/radio-translator/run.sh "
              "/opt/radio-translator/translate_service.py "
              "/opt/radio-translator/samples/franceinfo_60s.wav "
              "--speech /tmp/out.wav --json",
  timeoutMs = 120000
)
```

Returns exitCode 0, stdout = the JSON (text + speech_file path), stderr =
progress. To translate audio captured on the phone: upload it to the
workstation (Haven upload_file / SFTP), run_command on that absolute path, then
fetch the speech WAV via serve_file / SFTP if wanted.

## End-to-end demonstration (non-login exec == run_command)

Verified through a non-login SSH exec (cwd = home dir, no profile sourcing, the
same conditions as run_command) on a workstation with an RTX 5090:

```
ssh <user>@<workstation-host> \
  '/opt/radio-translator/run.sh \
   /opt/radio-translator/translate_service.py \
   /opt/radio-translator/samples/franceinfo_60s.wav --json'
```

exitCode 0, fr->en, RTF 0.0156 (load 1.5 s, infer 0.9 s). A direct absolute call
(no run.sh) also passes via non-login exec (self-activation), de->en RTF 0.029.
The original failure cases (shebang call, relative path from the home dir) now
pass too.

Delivery bugs fixed after conductor verification via real Haven run_command:
(1) the shebang used system python3 (no soundfile / CUDA-12 libs) — fixed by
self-activating the venv + CUDA libs and documenting run.sh as the entry;
(2) relative paths resolved against the exec cwd — fixed by documenting
absolute paths and resolving relative input against the project dir.

## Not verified

- Phone-side integration: the actual run_command call from Haven (the demo used
  an equivalent SSH exec channel). Wiring Haven's workstation profile and
  calling from the phone remains.
- Audio transfer phone<->workstation (upload/serve_file) not exercised.

## Verified since the first demo

- Live HF over-the-air now works on an SDRplay RSPdx-R2 (antenna port C on the
  development machine). Mode A taps the SDRconnect GUI's audio output; Mode B
  is fully headless: `sdrplay_cap ... - C | demod_stream.py | live.py --stdin`.
  A 60 s Mode B run on 9.630 MHz produced 5 pause-closed chunks at 2.3-4.1 s
  latency with coherent English and no hallucinations. See README "RF path".
- The 14-language regression passes 14/14; the Haven one-shot still returns
  exit 0, fr->en, RTF 0.016 through a non-login SSH exec.
- The PZSDR/Pluto (AD9361) still cannot reach HF; the RSPdx is the HF receiver.