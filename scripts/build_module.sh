#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Build the SDR++ translator module against a PINNED SDR++ commit. SDR++ modules
# are ABI-coupled to the exact core (the loader does no version check), so we
# pin the commit and label the output with it. Build deps: cmake, g++, git,
# libwebsockets-dev, libvolk-dev, libfftw3-dev, libglfw3-dev, libglew-dev,
# libzstd-dev (SDR++ core deps). Use official SDR++ nightlies, NOT distro packages.
#
# Usage: scripts/build_module.sh [--sdrpp-dir DIR] [--install-dir DIR] [--commit SHA]
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SDRPP_COMMIT="bad387c032d7299b6927a0c462cf98eaedaba835"
SDRPP_DIR="$HOME/Code/SDRPlusPlus"
INSTALL_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --sdrpp-dir) SDRPP_DIR="$2"; shift 2 ;;
    --install-dir) INSTALL_DIR="$2"; shift 2 ;;
    --commit) SDRPP_COMMIT="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

if [ ! -d "$SDRPP_DIR/.git" ]; then
  echo "==> cloning SDR++ into $SDRPP_DIR"
  git clone https://github.com/AlexandreRouma/SDRPlusPlus.git "$SDRPP_DIR"
fi
cd "$SDRPP_DIR"
git fetch origin --quiet
git checkout --quiet "$SDRPP_COMMIT"
echo "==> SDR++ at $(git rev-parse --short HEAD)"

echo "==> linking translator module"
ln -sfn "$ROOT/sdrpp_translator" decoder_modules/translator

echo "==> enabling OPT_BUILD_TRANSLATOR in SDR++ CMakeLists"
python3 - "$SDRPP_DIR/CMakeLists.txt" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
if "OPT_BUILD_TRANSLATOR" not in s:
    s = s.replace("option(OPT_BUILD_RADIO",
                  'option(OPT_BUILD_TRANSLATOR "Live audio translation (radio-translator)" ON)\noption(OPT_BUILD_RADIO', 1)
    anchor = 'if (OPT_BUILD_RADIO)\nadd_subdirectory("decoder_modules/radio")\nendif (OPT_BUILD_RADIO)'
    s = s.replace(anchor, anchor +
                  '\n\nif (OPT_BUILD_TRANSLATOR)\nadd_subdirectory("decoder_modules/translator")\nendif (OPT_BUILD_TRANSLATOR)', 1)
    open(p, "w").write(s)
PY

echo "==> configuring + building translator (and core)"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DOPT_BUILD_TRANSLATOR=ON \
  -DOPT_BUILD_SDRPLAY_SOURCE=OFF -DOPT_BUILD_RTL_SDR_SOURCE=OFF \
  -DOPT_BUILD_SOAPY_SOURCE=OFF -DOPT_BUILD_PLUTOSDR_SOURCE=OFF \
  -DOPT_BUILD_HACKRF_SOURCE=OFF -DOPT_BUILD_BLADERF_SOURCE=OFF \
  -DOPT_BUILD_LIMESDR_SOURCE=OFF -DOPT_BUILD_AIRSPY_SOURCE=OFF \
  -DOPT_BUILD_AIRSPYHF_SOURCE=OFF -DOPT_BUILD_DISCORD_PRESENCE=OFF >/dev/null
cmake --build build --target translator -j"$(nproc)"

SO="$SDRPP_DIR/build/decoder_modules/translator/translator.so"
SHORTSHA=$(git rev-parse --short HEAD)
if [ -n "$INSTALL_DIR" ]; then
  mkdir -p "$INSTALL_DIR"
  cp "$SO" "$INSTALL_DIR/translator.so"
  echo "==> installed $INSTALL_DIR/translator.so"
fi
echo "==> built $SO"
echo "==> compatibility: SDR++ commit $SDRPP_COMMIT ($SHORTSHA); module protocol v1"
echo "    Copy translator.so into your SDR++ plugins dir and create a 'Translator' instance."
