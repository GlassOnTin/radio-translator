# radio-translator v0.1.0

First release of the SDR++ translator plugin and its translation daemon.

## What's included

- **SDR++ module** (`sdrpp_translator/`): taps the radio's demod audio, streams
  to the daemon over WebSocket, and shows a building paragraph transcript with
  voice-isolation, filtered-audio and English-TTS toggles, independent volumes,
  source-language pin, voice/speed pickers, chunk/silence/transcript-length
  controls and copy-to-clipboard.
- **Translation daemon** (`tools/translate_daemon.py`): silero-VAD chunker +
  hallucination gates + faster-whisper translate, optional voice isolation and
  Kokoro TTS, compute tiering (GPU large-v3 → CPU int8 medium/small), versioned
  WebSocket protocol (v1).
- Headless RF capture/demod tools, voice-isolation stage, ASR benchmarks, and
  55 unit tests.

## Install

```
./setup.sh [--enhance] [--whisper medium]      # daemon venv + models
./run.sh tools/translate_daemon.py --port 8770
scripts/build_module.sh --install-dir <sdrpp-plugins-dir>   # module
```

Docker: `docker compose -f docker/docker-compose.yml up daemon-cpu` (GPU profile
needs the NVIDIA Container Toolkit). systemd user unit in `deploy/`.

## Compatibility

- Built against SDR++ nightly commit `bad387c0` (2026-11). WebSocket protocol v1.
- CI module binaries are labelled `translator_<distro>_amd64_sdrpp-bad387c0.so`
  (ubuntu jammy/noble, debian bookworm/trixie).
- Use official SDR++ nightlies, not distro packages. The module loader does no
  ABI check; for a different nightly, rebuild with
  `scripts/build_module.sh --commit <sha>`.

## Known limits

- Mismatched module/core builds can crash SDR++ (no ABI guard) — rebuild per
  nightly commit.
- First run downloads models: whisper ~0.5–3 GB by tier, Kokoro ~350 MB,
  voice-isolator ~250 MB. `scripts/fetch_models.py --offline` verifies a
  pre-seeded cache.
- CPU tier has higher latency than GPU; small int8 echoes the source language on
  some inputs (the gates suppress those chunks).
- GPU Docker is driver-sensitive; the CPU image is the reliable published artifact.

## Licensing

AGPL-3.0-or-later (`LICENSE`). The module combines with the GPL-3.0-only SDR++
core under GPLv3/AGPLv3 §13; the daemon is a separate work. Third-party licenses
and model-weight notices: `THIRD_PARTY_NOTICES.md` and `licenses/`.
