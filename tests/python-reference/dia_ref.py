#!/usr/bin/env python3
"""Authoritative nari-labs Dia-1.6B reference for the fixed TTS->STT test sentence.

Generates the same sentence our C# HartsyInference DiaPipeline is meant to produce, saves a
WAV, and prints generation time + audio duration + RMS. Run this on the GPU box (needs the
`dia` package: `pip install nari-tts` / `pip install git+https://github.com/nari-labs/dia`)
to get the GOLD reference to STT-compare against the C# engine output.

Params match DiaConfig.cs (cfg_scale 3.0 / temperature 1.2 / top_p 0.95 / cfg_filter_top_k 45,
seed 42) so the comparison is apples-to-apples. Upstream _sample_next_token runs the ENTIRE
sampler (EOS-only-when-argmax rule, top_k, top_p, softmax, multinomial) on the CFG-COMBINED
logits (cond + cfg*(cond-uncond)). The C# engine instead substitutes the CONDITIONAL logits as
the sampling distribution and only uses the combined logits to pick the top_k candidate window
-- so CFG guidance is discarded from the actual draw. That is the suspected root-cause bug this
reference is meant to expose (the EOS-argmax rule itself is faithful and should NOT be removed).

Assumptions (no config.json ships in the local cache dir, so we rely on from_pretrained to pull
the repo config+weights from HF):
  * Dia is trained on dialogue; the sentence is prefixed with the [S1] speaker tag.
  * Repo id "nari-labs/Dia-1.6B-0626" (matches dia_ref_nari.py); fall back to "nari-labs/Dia-1.6B".
  * compute_dtype float32 for a faithful (non-fp16) reference.
DO NOT run this on the analysis box (loads a 1.6B model -> OOM). GPU only.
"""
import warnings; warnings.filterwarnings("ignore")
import os, time
import numpy as np
import torch
import scipy.io.wavfile as wav
from dia.model import Dia

TEXT = "[S1] The speech synthesizer is now working correctly."
OUT_DIR = "/tmp/hartsyinference_tts_to_stt"
OUT = os.path.join(OUT_DIR, "dia_REF_python.wav")
SR = 44100
SEED = 42
os.makedirs(OUT_DIR, exist_ok=True)

def load():
    for repo in ("nari-labs/Dia-1.6B-0626", "nari-labs/Dia-1.6B"):
        try:
            print(f"loading {repo} (float32) ...", flush=True)
            return Dia.from_pretrained(repo, compute_dtype="float32")
        except Exception as e:
            print(f"  {repo} failed: {e}", flush=True)
    raise SystemExit("could not load any Dia checkpoint")

def main():
    model = load()
    print(f"generating: {TEXT!r}", flush=True)
    print("params: cfg_scale=3.0 temperature=1.2 top_p=0.95 cfg_filter_top_k=45 "
          f"max_tokens=1720 seed={SEED}", flush=True)
    torch.manual_seed(SEED)
    t0 = time.time()
    audio = model.generate(
        TEXT,
        max_tokens=1720,
        cfg_scale=3.0,
        temperature=1.2,
        top_p=0.95,
        cfg_filter_top_k=45,
        use_torch_compile=False,
        verbose=True,
    )
    gen_s = time.time() - t0

    a = np.asarray(audio, dtype=np.float32).flatten()
    dur_s = len(a) / SR
    rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) if a.size else 0.0
    peak = float(np.abs(a).max()) + 1e-8

    # Save normalized 16-bit PCM (same convention as the other dia_ref_*.py scripts).
    wav.write(OUT, SR, (a / peak * 0.95 * 32767).astype(np.int16))

    print("---- Dia reference result ----", flush=True)
    print(f"WROTE      {OUT}", flush=True)
    print(f"gen_time   {gen_s:.2f} s", flush=True)
    print(f"duration   {dur_s:.2f} s  ({len(a)} samples @ {SR} Hz)", flush=True)
    print(f"RMS        {rms:.5f}   peak {peak:.4f}", flush=True)
    if dur_s < 0.4 or rms < 1e-4:
        print("WARNING: suspiciously short/quiet -- generation likely failed.", flush=True)

if __name__ == "__main__":
    main()
