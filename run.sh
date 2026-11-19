#!/bin/sh
# Wrapper: run the project venv python with the CUDA 12 libraries that
# ctranslate2/faster-whisper need (libcublas.so.12, cudnn) on LD_LIBRARY_PATH.
# The system CUDA is v13; the venv ships the v12 libs via the nvidia-* wheels.
# Usage: ./run.sh translate_file.py ...   |   ./run.sh live.py ...
DIR=$(cd "$(dirname "$0")" && pwd)
NV=$(echo "$DIR"/.venv/lib/python3.12/site-packages/nvidia/*/lib | tr ' ' ':')
export LD_LIBRARY_PATH="$NV${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$DIR/.venv/bin/python" "$@"