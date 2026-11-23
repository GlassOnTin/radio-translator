# Third-party notices

radio-translator is **AGPL-3.0-or-later** (see `LICENSE`). It incorporates or
links the following third-party components. License texts are in `licenses/`.

## SDR++ (plugin host) — GPL-3.0-only

The `sdrpp_translator` C++ module links the SDR++ core
(<https://github.com/AlexandreRouma/SDRPlusPlus>), licensed **GPL-3.0-only**
(no "or later"). GPLv3 §13 and AGPLv3 §13 explicitly permit combining GPLv3 and
AGPLv3 works: the module (AGPL) and core (GPL) form a combined work that carries
the AGPL §13 network-source clause; the core remains GPLv3. When conveying a
prebuilt `translator.so`, provide the corresponding source for both parts —
this repo's module source plus the **exact SDR++ commit** it was built against
(pinned in `scripts/build_module.sh` and the release notes). The Python daemon
is a separate process communicating over WebSocket and is not part of that
combined work.

## Python daemon dependencies (all permissive, GPL/AGPL-compatible)

| Component | License | Notes |
|---|---|---|
| faster-whisper | MIT | <https://github.com/SYSTRAN/faster-whisper> |
| CTranslate2 | MIT | <https://github.com/OpenNMT/CTranslate2> |
| whisper (CT2 conversions, Systran) | MIT | model weights permissive |
| Silero VAD | MIT | ships inside faster-whisper |
| kokoro-onnx (code) | MIT | `licenses/kokoro-onnx-LICENSE.txt` |
| Kokoro-82M (weights) | Apache-2.0 | `licenses/Apache-2.0.txt`; <https://huggingface.co/hexgrad/Kokoro-82M> |
| numpy, soundfile, av, websockets | BSD / MIT / LGPL | permissive; av (FFmpeg) LGPL |
| sherpa-onnx (optional bench) | Apache-2.0 | benchmark only |

## Voice isolation (optional) — Apache-2.0

inworld voice-isolator, **code and weights Apache-2.0**
(<https://github.com/inworld-ai/voice-isolator>). The weights' NOTICE
(`licenses/voice-isolator-NOTICE.txt`) must be preserved when redistributing the
weights; it credits training data including VocalSet (CC BY 4.0) and
IndicVoices-R (CC BY 4.0). See `licenses/voice-isolator-LICENSE.txt`.

## Native module dependency

| Component | License | Notes |
|---|---|---|
| libwebsockets | MIT (subcomponents BSD/ZLIB/Apache/CC0/OFL) | keep upstream notices when redistributing binaries |

## Receiver SDK (separate tool, not the plugin)

`tools/sdrplay_cap.c` links the **proprietary SDRplay API** (libsdrplay_api).
This is a standalone receive-only capture tool, separate from the GPL/AGPL
plugin and daemon; the SDR++ module does not link it (it taps demodulated audio).
The SDRplay API must be installed separately under SDRplay's own terms.

## Model weights (downloaded, not bundled)

Models are fetched on first run (`scripts/fetch_models.py`), not redistributed
in this repo or its releases: Kokoro (Apache-2.0), whisper (MIT), voice-isolator
(Apache-2.0). Preserve the associated LICENSE/NOTICE files alongside cached
weights (Apache-2.0 §4).
