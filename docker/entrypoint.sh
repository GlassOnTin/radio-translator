#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Put the pip nvidia CUDA-12 libs on LD_LIBRARY_PATH (ctranslate2 needs them),
# then start the daemon. Mirrors run.sh for the container's system python.
NV=""
for d in /usr/local/lib/python3.*/dist-packages/nvidia/*/lib /usr/lib/python3.*/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && NV="$NV${NV:+:}$d"
done
export LD_LIBRARY_PATH="$NV${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec python3 tools/translate_daemon.py --host 0.0.0.0 --port 8770 "$@"
