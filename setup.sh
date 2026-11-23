#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Set up the translation daemon: create .venv (CPU works without a GPU; CUDA-12
# wheels enable GPU), optionally .venv-tts for voice isolation (--enhance), and
# fetch model weights. Requires uv (https://docs.astral.sh/uv/) and ffmpeg.
#
# Usage: ./setup.sh [--enhance] [--whisper small|medium|large-v3] [--offline]
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
cd "$DIR"

ENHANCE=0
FETCH_ARGS=""
for a in "$@"; do
  case "$a" in
    --enhance) ENHANCE=1 ;;
    --whisper|--offline) FETCH_ARGS="$FETCH_ARGS $a" ;;
    small|medium|large-v3) FETCH_ARGS="$FETCH_ARGS $a" ;;
  esac
done

command -v uv >/dev/null 2>&1 || { echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || echo "warning: ffmpeg not found (needed for audio decode)"

echo "==> main venv (.venv)"
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
# GPU libraries for ctranslate2 (CUDA 12 + cuDNN 9). Harmless to skip on CPU-only.
uv pip install --python .venv/bin/python nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" \
  || echo "==> GPU wheels skipped (CPU-only is fine)"

if [ "$ENHANCE" = 1 ]; then
  echo "==> enhancement venv (.venv-tts, torch)"
  uv venv .venv-tts --python 3.12
  uv pip install --python .venv-tts/bin/python torch==2.15.0 \
    --index-url https://download.pytorch.org/whl/cu132
  uv pip install --python .venv-tts/bin/python -r requirements-enhance.txt
fi

echo "==> model weights"
# shellcheck disable=SC2086
./run.sh scripts/fetch_models.py $FETCH_ARGS

echo ""
echo "Setup complete. Start the daemon:"
echo "  ./run.sh tools/translate_daemon.py --port 8770"
echo "With voice isolation + TTS (needs --enhance and a GPU for isolation):"
echo "  ./run.sh tools/translate_daemon.py --port 8770 --enhance isolate --audio-out --tts"
