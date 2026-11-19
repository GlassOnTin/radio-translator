# radio-translator

Translates international shortwave radio to English text and speech, locally.
An SDR receives the broadcast, the audio is demodulated, faster-whisper
(large-v3) translates it on a CUDA GPU, and Kokoro speaks the English on CPU.
Everything runs on your machine. No internet, no web streams.

License: AGPL-3.0-or-later (see LICENSE).

## How it works

ffmpeg reads the audio (a file, ALSA, or a PipeWire source) and resamples to
16 kHz mono. webrtcvad cuts it into speech chunks at pauses. faster-whisper
large-v3 translates each chunk to English, detecting the language per chunk.
Kokoro turns the English into speech and writes a WAV. Text goes to stdout.

Playback is off by default. Nothing reaches the speakers unless you pass
`--play`.

## Requirements

- A CUDA GPU for faster-whisper large-v3 (tested on an RTX 5090). Kokoro runs
  on CPU.
- A Python 3.12 venv with faster-whisper, kokoro-onnx, webrtcvad, soundfile,
  numpy.
- ffmpeg on PATH.
- Models: faster-whisper large-v3 (HuggingFace cache) and Kokoro ONNX plus
  voices in `models/`. Download once, then run offline.
- CUDA 12 libraries. ctranslate2 needs `libcublas.so.12`. If your system CUDA
  is newer, `run.sh` points `LD_LIBRARY_PATH` at the venv's nvidia CUDA-12
  wheels.

## Run

Offline, on a 16 kHz mono WAV:

```
./run.sh translate_file.py input.wav output.txt translate
```

Live, on a file or audio source (text plus an English WAV):

```
./run.sh live.py input.wav --out english.wav --chunk 12
./run.sh live.py --pulse <source> --out english.wav
```

`live.py` flags: `--seconds`, `--chunk`, `--silence`, `--lang`, `--voice`,
`--pulse`, `--alsa`, `--out`. Leave `--play` off to keep audio out of the
speakers.

## Haven service

`translate_service.py` exposes `translate(audio file) -> English text +
optional speech WAV` as a one-shot command for Haven (the pocket thin-client)
to run remotely over SSH (`run_command`). The GPU work runs on the workstation,
not the phone. Use `run.sh` and absolute paths (run_command execs from the home
directory with no login profile):

```
/opt/radio-translator/run.sh \
  /opt/radio-translator/translate_service.py \
  /opt/radio-translator/samples/franceinfo_60s.wav --json
```

Replace `/opt/radio-translator` with your install directory. See SERVICE.md for
the Haven invocation and the end-to-end demo.

## RF path (needs HF hardware)

For live over-the-air shortwave:

```
HF antenna -> SDRplay RSPdx -> tools/sdrplay_cap (IQ)
  -> tools/demod.py (AM/SSB to 16 kHz) -> PipeWire virtual sink
  -> live.py --pulse <sink>.monitor
```

Create the sink:

```
pactl load-module module-null-sink sink_name=radio_translator
```

Feed demodulated audio to the sink and capture its monitor with `live.py
--pulse radio_translator.monitor`. The null sink has no physical output, so
nothing plays. `tools/demod.py` is validated on a synthetic AM tone.

## Hardware notes and limitations

- Live HF needs an HF antenna on the SDRplay (port A) or another HF-capable
  receiver. Without it, the offline file path still works.
- The PZSDR/Pluto (AD9361) covers 70 MHz to 6 GHz and cannot receive HF
  shortwave.
- The phone-side Haven integration and phone-to-workstation audio transfer are
  not yet exercised. The service is verified over an SSH exec channel.
- Sample broadcast WAVs and their translations are kept local (gitignored) and
  not redistributed. Provide your own 16 kHz mono WAVs to reproduce the
  translations.

## Layout

`live.py`, `translate_file.py`, `translate_service.py`: live, offline, and
Haven-invocable translation. `speak.py`, `clone_test.py`, `verify_languages.py`:
TTS helpers. `tools/`: SDRplay capture and AM/SSB demod. `run.sh`: venv and
CUDA-12 wrapper. `models/`: Kokoro (not committed). `samples/`, `captures/`:
local fixtures (not committed).

## License

AGPL-3.0-or-later. Source files carry
`SPDX-License-Identifier: AGPL-3.0-or-later`. You may redistribute and modify
this program under the GNU Affero General Public License, version 3 or (at your
option) any later version.