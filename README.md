# radio-translator

Local, over-the-air international radio → English text + English speech.
No internet, no web streams: an SDR receives HF shortwave, demodulates to
audio, and faster-whisper (large-v3, CUDA) translates locally while Kokoro
(CPU) speaks the English to a WAV. AGPL-3.0.

**Quiet-hours rule:** never pass `--play`; nothing reaches the speakers.
Text goes to stdout, English speech goes to WAV files only.

## Pipeline

```
HF antenna -> SDR (RSPdx) -> IQ capture -> AM/SSB demod -> 16 kHz audio
   -> PipeWire virtual sink -> <sink>.monitor -> live.py --pulse
   -> webrtcvad chunks -> faster-whisper large-v3 (CUDA, translate)
   -> English text (stdout) + Kokoro TTS (CPU) -> English WAV
```

Offline path (no RF): `translate_file.py in.wav out.txt` or
`live.py in.wav --out english.wav` (file input instead of `--pulse`).

## Status

- **Offline translation: VERIFIED** on the sample WAVs (see below).
- **PipeWire sink → live.py --pulse path: VERIFIED** with a null sink.
- **Live HF over-the-air: BLOCKED on hardware** (see Blockers). The software
  is ready; it needs an HF antenna / HF-capable receiver.

## Offline verification (2026-11-19, RTX 5090, no internet)

`translate_file.py` (faster-whisper large-v3, float16, CUDA) on `samples/lang_*.wav`,
auto language detection, text only:

| sample | detected lang | p | audio | RTF |
|---|---|---|---|---|
| lang_zh | zh | 0.99 | 30 s | 0.033 |
| lang_ar | ar | 1.00 | 60 s | 0.024 |
| lang_ru | ru | 1.00 | 30 s | 0.029 |
| lang_de | de | 1.00 | 30 s | 0.030 |
| lang_ja | ja | 0.86 | 30 s | 0.011 |
| lang_es | es | 1.00 | 30 s | 0.024 |
| franceinfo_60s | fr | 1.00 | 60 s | 0.017 |

English text in `captures/<lang>_en.txt`.

`live.py` end-to-end (VAD chunk → whisper translate → Kokoro TTS → WAV),
**no `--play`**:
- file input `franceinfo_60s.wav` → 4 chunks fr→en → `captures/live_fr_test.wav`
  (50.6 s English speech, 24 kHz).
- PipeWire path: German sample fed to a null sink `radio_translator`, captured
  via `--pulse radio_translator.monitor` → 4 chunks de→en →
  `captures/live_pulse_test.wav` (13.8 s). The null sink discards audio (no
  speaker output); only its monitor is captured.

## Running

CUDA 12 libs: the system CUDA is v13 but ctranslate2 needs `libcublas.so.12`,
which the venv ships via the `nvidia-*` wheels. Use the wrapper, which sets
`LD_LIBRARY_PATH` to the venv nvidia lib dirs:

```sh
./run.sh translate_file.py samples/lang_de.wav out.txt translate
./run.sh live.py samples/franceinfo_60s.wav --out english.wav --chunk 12
./run.sh live.py --pulse radio_translator.monitor --out english.wav   # RF path
```

`live.py` flags: `--seconds`, `--chunk`, `--silence`, `--lang`, `--voice`,
`--pulse SOURCE`, `--alsa DEVICE`, `--out WAV`. `--play` is OFF by default —
leave it off during quiet hours.

## RF-to-whisper path (for when HF hardware arrives)

1. **Create a PipeWire virtual sink** (no physical output):
   ```sh
   pactl load-module module-null-sink sink_name=radio_translator \
       sink_properties=device.description=RadioTranslator
   # monitor source: radio_translator.monitor
   ```
2. **Capture IQ** from the SDRplay RSPdx (receive-only):
   ```sh
   tools/sdrplay_cap <freq_hz> 2000000 <seconds> cap.iq A 30
   # e.g. tools/sdrplay_cap 9570000 2000000 30 cap.iq A 30
   ```
3. **Demod AM/SSB to 16 kHz WAV** (numpy-only):
   ```sh
   .venv/bin/python tools/demod.py cap.iq audio.wav --fs 2000000 --mode am
   # SSB: --mode usb|lsb --bfo <offset_hz>
   ```
   AM validated on a synthetic 1 kHz tone (demod recovers 1000 Hz).
4. **Feed the sink and translate** (no speaker output — null sink):
   ```sh
   paplay --device=radio_translator audio.wav &
   ./run.sh live.py --pulse radio_translator.monitor --out english.wav
   ```
   For continuous live RF, stream the demodulated audio into the sink
   (e.g. `ffmpeg -f pulse -i ... ` or a demod loop writing to
   `parec --device=radio_translator`'s sink) instead of a one-shot `paplay`.
5. Tear down: `pactl unload-module <module-index>`.

The verified null-sink test above used `paplay` in place of the SDR demod
output; swapping in `tools/demod.py`'s WAV (or a live demod stream) is the
only change needed once an HF antenna/receiver is present.

## Tools

- `tools/sdrplay_cap.c` — receive-only RSPdx IQ capture via SDRplay API v3
  (build: `gcc -O2 -I/usr/local/include sdrplay_cap.c -o sdrplay_cap
  -L/usr/local/lib -lsdrplay_api -Wl,-rpath,/usr/local/lib`). Writes raw
  interleaved int16 I/Q. Args: freq_hz fs_hz duration_s out.iq [A|B|C] [grdB].
- `tools/demod.py` — AM/SSB IQ → 16 kHz mono WAV (numpy-only).
- `run.sh` — venv python + CUDA 12 `LD_LIBRARY_PATH`.

## Blockers (physical — do not pursue in software)

- **No HF antenna.** The SDRplay RSPdx (USB `1df7:3060`) is present, the
  SDRplay API v3.15 service is running, and `tools/sdrplay_cap` captures IQ
  successfully — but there is no HF antenna connected, so live over-the-air
  international shortwave cannot be received. A test capture at 9570 kHz
  yielded only noise/no usable broadcast.
- **PZSDR/Pluto (AD9361) cannot reach HF** (70 MHz–6 GHz); it is not an HF
  receiver. ADS-B at 1090 MHz was also confirmed too weak on its window
  wideband antenna (see the pzsdr-p201pro dataset).
- **Needed:** an HF antenna for the RSPdx (port A, 1 kHz–2 GHz), or an
  HF-capable receiver. SoapySDR has no SDRplay module installed; the capture
  tool uses the SDRplay API directly instead (no Soapy dependency).

Until then, the offline pipeline is verified and ready; advance software only.

## Layout

- `live.py`, `translate_file.py` — live / offline translation.
- `speak.py`, `clone_test.py`, `verify_languages.py` — TTS helpers.
- `samples/` — test WAVs (16 kHz mono, multiple languages; not committed).
- `captures/` — verification text + English WAV outputs.
- `models/` — Kokoro ONNX + voices (not committed; whisper large-v3 is in the
  HuggingFace cache).