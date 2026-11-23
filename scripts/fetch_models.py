#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Download model weights (download-on-first-run; too large to bundle in a
GitHub release). Kokoro TTS weights are pinned by URL + sha256; whisper weights
come from HuggingFace (optionally pre-seeded here); voice-isolator weights are
fetched by the inworld-voice-isolation package on first use.

Usage:
  scripts/fetch_models.py                 # Kokoro into models/
  scripts/fetch_models.py --whisper medium  # also pre-seed the whisper HF cache
  scripts/fetch_models.py --offline       # verify existing files, no download
"""
import argparse
import hashlib
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO = {
    "kokoro-v1.0.onnx": (f"{KOKORO_BASE}/kokoro-v1.0.onnx",
                         "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"),
    "voices-v1.0.bin": (f"{KOKORO_BASE}/voices-v1.0.bin",
                        "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"),
}
WHISPER_REPO = {
    "large-v3": "Systran/faster-whisper-large-v3",
    "medium": "Systran/faster-whisper-medium",
    "small": "Systran/faster-whisper-small",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name, url, digest, dest, offline):
    path = os.path.join(dest, name)
    if os.path.isfile(path):
        if sha256(path) == digest:
            print(f"  {name}: present, checksum OK")
            return
        if offline:
            sys.exit(f"  {name}: checksum MISMATCH (offline mode)")
        print(f"  {name}: checksum mismatch, re-downloading")
    if offline:
        sys.exit(f"  {name}: missing (offline mode)")
    print(f"  {name}: downloading {url}")
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    got = sha256(tmp)
    if got != digest:
        os.remove(tmp)
        sys.exit(f"  {name}: download checksum MISMATCH {got}")
    os.replace(tmp, path)
    print(f"  {name}: OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-dir", default=os.path.join(ROOT, "models"))
    ap.add_argument("--whisper", choices=list(WHISPER_REPO), default=None,
                    help="also pre-seed the HuggingFace cache for this whisper size")
    ap.add_argument("--canary", action="store_true",
                    help="pre-seed Canary-1b-v2 (CC-BY-4.0, European S2TT)")
    ap.add_argument("--seamless", action="store_true",
                    help="OPT-IN: pre-seed SeamlessM4T v2 (CC-BY-NC-4.0, NON-COMMERCIAL S2ST)")
    ap.add_argument("--offline", action="store_true", help="verify only, no download")
    args = ap.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    print("Kokoro TTS weights (Apache-2.0):")
    for name, (url, digest) in KOKORO.items():
        fetch(name, url, digest, args.models_dir, args.offline)

    if args.whisper:
        repo = WHISPER_REPO[args.whisper]
        print(f"Whisper {args.whisper} (MIT): pre-seeding HF cache {repo}")
        if args.offline:
            print("  (offline: skip)")
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo)
            print("  OK")

    if args.canary:
        print("Canary-1b-v2 (CC-BY-4.0): pre-seeding HF cache nvidia/canary-1b-v2")
        if not args.offline:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="nvidia/canary-1b-v2")
            print("  OK")
    if args.seamless:
        print("SeamlessM4T v2 (CC-BY-NC-4.0, NON-COMMERCIAL opt-in): pre-seeding "
              "HF cache facebook/seamless-m4t-v2-large")
        if not args.offline:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="facebook/seamless-m4t-v2-large")
            print("  OK")

    print("Voice-isolator weights (Apache-2.0): fetched by the inworld package "
          "on first use (tools/isolate_stream.py).")
    print("Done.")


if __name__ == "__main__":
    main()
