# radio-translator

Translates international shortwave radio to English text and speech, locally.
An SDR receives the broadcast, the audio is demodulated, faster-whisper
(large-v3) translates it on a CUDA GPU, and Kokoro speaks the English on CPU.
Everything runs on your machine. No internet, no web streams.

License: AGPL-3.0-or-later (see LICENSE).

## How it works

ffmpeg reads the audio (a file, ALSA, a PipeWire source, or raw PCM on stdin)
and resamples to 16 kHz mono. A silero-VAD chunker cuts it into speech chunks
at pauses, with pre-roll so starts aren't clipped and a soft cap to avoid
mid-word cuts. Silero rejects music beds, which otherwise produce fluent
whisper hallucinations; webrtcvad is available with `--vad webrtc` but fires
on music and noise. faster-whisper large-v3 translates each chunk to English,
detecting the language per chunk. Repetition, CJK-echo and segment-quality
gates suppress hallucinated chunks. Kokoro turns the English into speech and
writes a WAV. Text goes to stdout.

Playback is off by default. Nothing reaches the speakers unless you pass
`--play`.

## Requirements

- A CUDA GPU for faster-whisper large-v3 (tested on an RTX 5090). Kokoro runs
  on CPU.
- A Python 3.12 venv with the pinned deps in `requirements.txt`
  (faster-whisper, kokoro-onnx, webrtcvad, soundfile, numpy, websockets).
  `requirements-bench.txt` adds sherpa-onnx for the ASR benchmark only.
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
./run.sh live.py input.wav --out english.wav
./run.sh live.py --pulse <source> --out english.wav
./run.sh live.py --stdin --out english.wav   # raw s16le 16 kHz mono on stdin
```

`live.py` flags: `--seconds`, `--chunk` (default 7.5 s), `--silence` (300 ms),
`--preroll` (300 ms), `--carry`, `--lang`, `--voice`, `--pulse`, `--alsa`,
`--stdin`, `--out`, `--af` (ffmpeg filter chain, e.g.
`"highpass=f=80,lowpass=f=4500,afftdn"`), `--vad silero|webrtc`,
`--vad-threshold`, `--no-gates`, `--no-agc`, `--stream-tts`. Leave `--play` off
to keep audio out of the speakers. Worst-case chunk latency is about `--chunk`
plus inference; on air we measured 2-4 s with the default settings.
`--stream-tts` synthesizes Kokoro audio sentence-by-sentence; it cut
time-to-first-audio by roughly 0.1-0.45 s on our chunks (Kokoro on CPU is
already fast, so the gain is modest). `--out` is still written at the end.

### Voice isolation (optional)

`--enhance isolate` puts a voice-isolation stage
([inworld-ai/voice-isolator](https://github.com/inworld-ai/voice-isolator),
Apache-2.0 code and weights) before the chunker. It extracts speech from
music and noise, so talk buried under a music bed becomes translatable
instead of being rejected by the VAD. It runs as a persistent streaming
subprocess in the torch venv (`.venv-tts`, see `requirements-enhance.txt`),
adds about 63 ms of latency (`--enhance-lookahead 2`), and processes audio at
RTF ~0.22 on a GPU. It shares the GPU mapping whisper uses, so set
`CUDA_VISIBLE_DEVICES` to a free card.

Measured on our fixtures: a French broadcast with speech under music went
from 3 chunks (silero speech fraction 0.07) to 15-16 chunks (fraction
0.31-0.37), recovering coherent speech; clean speech was unchanged (zh
0.76->0.77, fr 0.54->0.58); pure instrumental music stayed rejected (0
chunks). It does not turn music into speech — it recovers speech that is
already there but masked. `tools/isolate_audio.py` does the same offline for
files; `tools/enhance_metrics.py` compares raw vs enhanced audio.

## SDR++ plugin + translation daemon

The live pipeline can run as an [SDR++](https://www.sdrpp.org/) plugin. A thin
C++ module (`sdrpp_translator/`) taps the radio's demod audio, resamples it to
16 kHz mono and streams it over WebSocket to a local daemon, which returns
translated text shown in a sidebar panel. The module runs no models, so it
works on any machine; the compute lives in the daemon, local or remote.

### Install the daemon

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg. CPU works without a GPU.

```
./setup.sh                 # .venv + deps + Kokoro weights (CPU-ready)
./setup.sh --enhance       # also .venv-tts (torch) for voice isolation
./setup.sh --whisper medium  # optionally pre-seed the whisper HF cache
./run.sh tools/translate_daemon.py --port 8770
```

Models download on first run (`scripts/fetch_models.py`, pinned URLs +
checksums; `--offline` verifies a pre-seeded cache). Whisper weights come from
HuggingFace; voice-isolator weights via the inworld package. The daemon
auto-selects a compute tier and advertises it at handshake (protocol v1): a CUDA
GPU with >=6 GB free runs large-v3 float16; otherwise CPU int8 (medium on >=8
cores, small below). Override with `--device cpu --model medium --cpu-threads N`,
or point the plugin at a remote GPU daemon by URL. `--enhance isolate --audio-out
--tts` enables voice isolation, filtered-audio return and English TTS.

Run as a user service: `cp deploy/radio-translator-daemon.service
~/.config/systemd/user/ && systemctl --user enable --now radio-translator-daemon`.

Docker (CPU image published; GPU needs the NVIDIA Container Toolkit):

```
docker compose -f docker/docker-compose.yml up daemon-cpu
docker compose -f docker/docker-compose.yml --profile gpu up daemon-gpu
```

### Translation backends

The daemon takes `--backend`:

- `whisper` (default): faster-whisper large-v3 translate — broadest language
  coverage, CPU-int8 fallback.
- `whisper-mt`: whisper *transcribe* (source text) → a local ollama LLM
  (`--mt-model`, default qwen3.6:27b) → English. More idiomatic/coherent than
  whisper's built-in translate; ~2 s warm latency. Needs ollama running.
- `canary`: Canary-1b-v2 (CC-BY-4.0) for European languages, auto-routed by
  detected language with whisper fallback for the rest. Fast (~0.2 s warm).
  Weights: `scripts/fetch_models.py --canary`.
- `seamless`: **SeamlessM4T v2 direct speech-to-speech (CC-BY-NC-4.0,
  NON-COMMERCIAL, opt-in experiment)**. Outputs translated English speech
  directly (no text→TTS cascade); the whisper text transcript still feeds the
  sidebar. Weights are never auto-downloaded — fetch explicitly with
  `scripts/fetch_models.py --seamless`. Not for commercial use.

Torch backends (canary, seamless) run in `.venv-tts` (setup.sh --enhance, plus
sentencepiece/tiktoken). Compare backends with `tools/bench_backends.py FIXTURE
--ref REF_en.txt`. Note: BLEU against the bundled whisper-generated references
is circular (whisper scores 100 vs its own output); judge coherence by human
read — on spontaneous Spanish, whisper-mt reads most fluently, Canary is
literal but fast, whisper is garbled.

### Build the module

SDR++ modules are **ABI-coupled to the exact core** and the loader does no
version check — a mismatched `.so` can crash SDR++. Use official SDR++ nightlies,
**not distro packages** (upstream warns they are incompatible). The build script
clones SDR++ at a pinned commit and builds the module in-tree:

```
scripts/build_module.sh --install-dir ~/.config/sdrpp/plugins
# needs libwebsockets-dev + SDR++ core deps (libvolk-dev libfftw3-dev
# libglfw3-dev libglew-dev libzstd-dev); see the script header
```

Copy `translator.so` into your SDR++ plugins dir and create a **Translator**
instance in the Module Manager. In the panel set the daemon URL (default
`ws://127.0.0.1:8770`) and audio stream (default `Radio`). For SDRplay
RSPdx/RSPdx-R2, select the antenna port your HF feed is wired to and set gain
manually (AGC off, LNA 0, IF gain reduction ~30 dB worked here); the source's
default LNA state is minimum gain, too weak for shortwave.

### Compatibility

| Module | Protocol | SDR++ commit (pinned) | Built for |
|---|---|---|---|
| 0.1.0 | v1 | `bad387c0` (nightly, 2026-11) | ubuntu jammy/noble, debian bookworm/trixie (amd64) |

Prebuilt binaries (CI, on tags) are labelled
`translator_<distro>_<arch>_sdrpp-<shortsha>.so`. They work with nightlies near
that commit; for other versions rebuild with `scripts/build_module.sh --commit
<sha>`. The WebSocket protocol (v1) is the stable boundary between module and
daemon — they version independently, so a laptop plugin can use a remote GPU
daemon. Licensing: see `THIRD_PARTY_NOTICES.md` (AGPL module + GPL-3.0-only SDR++
core combine under both §13s; the daemon is a separate work).

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

## RF path (SDRplay RSPdx/RSPdx-R2)

The SDRplay API gives exclusive device access: only one client can hold the
RSP at a time. Build the capture tool with `make -C tools` (needs the SDRplay
API v3 headers/libs in `/usr/local`). Pass the antenna port your HF feed is
wired to as the 5th argument (`A`, `B` or `C`); the default is `A`.

Mode B, fully headless (verified over the air):

```
tools/sdrplay_cap 9630000 2000000 0 - C | \
  ./run.sh tools/demod_stream.py --fs 2000000 --mode am | \
  ./run.sh live.py --stdin --out english.wav
```

`sdrplay_cap` writes interleaved s16 IQ to stdout (`-`); duration `0` runs
until SIGINT/SIGTERM/SIGPIPE, then uninits and releases the device cleanly so
it is immediately reusable. `demod_stream.py` decimates 2 MS/s to 16 kHz with
a polyphase FIR, removes the carrier DC and AGCs the audio; its block-wise
output matches a whole-file call, and correlates with the `demod.py` reference
on a real capture at the predicted filter delay. `tools/scan_sw.py FREQ...`
captures, demodulates and language-ids a list of carriers to find a station.

Mode A, SDRconnect GUI (verified over the air): tune the GUI, then tap its
audio output with `live.py --pulse <sink>.monitor`. SDRconnect also has a
headless WebSocket server (`SDRconnect_headless`, port 5454); see
`tools/sdrconnect_ctl.py` for scripted tune / signal reads / audio streaming.
Note: SDRconnect 1.0.7 does not implement antenna selection over the WebSocket
API, so on a site wired to port B or C use Mode B (which sets the antenna) or
select the antenna once in the GUI.

`tools/demod.py` remains the whole-file AM/SSB reference (FFT low-pass, peak
normalised); `demod_stream.py` is the streaming path.

## Hardware notes and limitations

- Live HF needs an HF antenna on the SDRplay or another HF-capable receiver.
  The antenna port is site-specific (port C on the development machine, not
  the default A); pass it to `sdrplay_cap`. Without HF hardware the offline
  file path still works.
- The PZSDR/Pluto (AD9361) covers 70 MHz to 6 GHz and cannot receive HF
  shortwave.
- The phone-side Haven integration and phone-to-workstation audio transfer are
  not yet exercised. The service is verified over an SSH exec channel.
- Sample broadcast WAVs and their translations are kept local (gitignored) and
  not redistributed. Provide your own 16 kHz mono WAVs to reproduce the
  translations.

## Layout

`live.py`, `translate_file.py`, `translate_service.py`: live, offline, and
Haven-invocable translation. `asr.py`: shared model/decode/gate helpers.
`chunker.py`, `vad.py`: silero/webrtc speech chunking. `speak.py`,
`clone_test.py`, `verify_languages.py`: TTS helpers and the 14-language
regression. `tools/`: SDRplay capture (`sdrplay_cap.c`, `Makefile`), AM/SSB
demod (`demod.py`, `demod_core.py`, `demod_stream.py`), SDRconnect WebSocket
control (`sdrconnect_ctl.py`), carrier scan (`scan_sw.py`), ASR benchmark
(`bench_asr.py`), voice isolation (`isolate_audio.py`, `isolate_stream.py`,
`enhance_metrics.py`). `tests/`: unit tests (`./run.sh -m unittest discover
-s tests`). `run.sh`: venv and CUDA-12 wrapper. `models/`: Kokoro (not committed).
`samples/`, `captures/`: local fixtures (not committed).

## License

AGPL-3.0-or-later. Source files carry
`SPDX-License-Identifier: AGPL-3.0-or-later`. You may redistribute and modify
this program under the GNU Affero General Public License, version 3 or (at your
option) any later version.