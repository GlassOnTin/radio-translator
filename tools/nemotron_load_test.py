#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded load gate for Nemotron-3.5-Transcribe (N1) under transformers.
The official runtime is vLLM 0.32 + NeMo Speech; this checks whether the
transformers path in .venv-tts can load the NVFP4 checkpoint at all on a
32 GB RTX 5090. Run with .venv-tts/bin/python; pass the model dir.
"""
import sys
import time

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nemotron-nvfp4"


def main():
    import torch
    from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoProcessor
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(MODEL)
    print(f"config: {type(cfg).__name__} model_type={cfg.model_type}", flush=True)
    try:
        proc = AutoProcessor.from_pretrained(MODEL)
        print(f"processor: {type(proc).__name__} ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"processor FAILED: {type(e).__name__}: {e}", flush=True)
    print("loading model (device_map=cuda:0, bf16)...", flush=True)
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
        print(f"model LOADED: {type(model).__name__} ({time.time()-t0:.1f}s)", flush=True)
        print(f"VRAM0 {torch.cuda.memory_allocated(0)/2**30:.1f} GiB allocated, "
              f"{torch.cuda.max_memory_allocated(0)/2**30:.1f} GiB peak", flush=True)
        return 0
    except Exception as e:
        import traceback
        print(f"model LOAD FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
