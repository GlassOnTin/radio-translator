#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Wrapper: run the project venv python with the CUDA 12 libraries that
# ctranslate2/faster-whisper need (libcublas.so.12, cudnn) on LD_LIBRARY_PATH.
# The system CUDA is v13; the venv ships the v12 libs via the nvidia-* wheels.
# Usage: ./run.sh translate_file.py ...   |   ./run.sh live.py ...
DIR=$(cd "$(dirname "$0")" && pwd)
NV=""
for d in "$DIR"/.venv/lib/python3.*/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && NV="$NV${NV:+:}$d"
done
export LD_LIBRARY_PATH="$NV${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$DIR/.venv/bin/python" "$@"