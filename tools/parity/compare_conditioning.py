"""Tensor-by-tensor comparison of a LongCat avatar generation's step-0 conditioning (the engine's dump) against the reference's files.

Both sides are float32 .npy under the reference's names (text_embeddings_step00, text_mask_step00, audio_embedding_step00, audio_tokens_step00,
input_latent_step00, speech_16k_mono_step00). Each reference tensor is reshaped to the engine's layout by element count (C order), after the
audio_proj mapping below for audio_tokens_step00. Prints relative RMS (sqrt(mean((a-b)^2)) / sqrt(mean(b^2)), b = the reference) and max abs.

audio_tokens_step00 mapping (LongCat-Video @ 58eb336, modules/avatar/longcat_video_dit_avatar.py):
  audio_proj(first, latter) gives [B, T, 32, 768] (L520). With num_ref_latents > 0 the frame-0 copy is prepended along T (L522-523), then the
  last N_t entries are kept (L525). L534 rearranges "b t n c -> (b t) n c". For one clip T = N_t, so the prepend-and-slice is the identity.
  The engine's audio_tokens_step00 is [1, N_t, 32, 768], so the reference's file reshapes to it with no transpose.

Usage:
  python compare_conditioning.py <engine_dir> <reference_dir> [--num-ref-latents 0] [--bound 1e-3]
"""
import argparse
import os
import sys

import numpy as np

NAMES = [
    "text_embeddings_step00",
    "text_mask_step00",
    "audio_embedding_step00",
    "audio_tokens_step00",
    "input_latent_step00",
    "speech_16k_mono_step00",
]


def rel_rms(a, b):
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    denom = np.sqrt(np.mean(b * b))
    return float(np.sqrt(np.mean((a - b) ** 2)) / denom) if denom > 0 else float("nan")


def map_audio_tokens(ref, frames, num_ref_latents):
    """The reference's audio_proj output [B, T, 32, 768] to the engine's [1, frames, 32, 768] (L520-534)."""
    if ref.ndim == 3:
        ref = ref[None]
    if num_ref_latents > 0:
        ref = np.concatenate([ref[:, :1], ref], axis=1)
    ref = ref[:, -frames:]
    if ref.shape[1] != frames:
        raise SystemExit(f"audio_tokens: reference has {ref.shape[1]} time steps after the mapping, the engine has {frames}.")
    return ref


def load(path):
    return np.load(path).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine_dir")
    ap.add_argument("reference_dir")
    ap.add_argument("--num-ref-latents", type=int, default=0)
    ap.add_argument("--bound", type=float, default=1e-3)
    args = ap.parse_args()

    rows = []
    failed = []
    for name in NAMES:
        e_path = os.path.join(args.engine_dir, name + ".npy")
        r_path = os.path.join(args.reference_dir, name + ".npy")
        if not os.path.exists(e_path) or not os.path.exists(r_path):
            rows.append((name, "missing", "", "", "", ""))
            continue
        e = load(e_path)
        r = load(r_path)
        if name == "audio_tokens_step00":
            frames = e.shape[1] if e.ndim == 4 else e.shape[0]
            r = map_audio_tokens(r, frames, args.num_ref_latents)
        if r.size != e.size:
            rows.append((name, "COUNT MISMATCH", str(e.shape), str(r.shape), "", ""))
            failed.append(name)
            continue
        rr = rel_rms(e, r)
        mx = float(np.max(np.abs(e.reshape(-1) - r.reshape(-1))))
        verdict = "ok" if rr <= args.bound else "over bound"
        if rr > args.bound:
            failed.append(name)
        rows.append((name, verdict, str(e.shape), str(r.shape), f"{rr:.3e}", f"{mx:.3e}"))

    header = f"{'tensor':<26} {'verdict':<15} {'engine shape':<26} {'reference shape':<26} {'relRMS':>10} {'max abs':>10}"
    print(header)
    for row in rows:
        print(f"{row[0]:<26} {row[1]:<15} {row[2]:<26} {row[3]:<26} {row[4]:>10} {row[5]:>10}")
    if failed:
        print(f"RESULT: {len(failed)} tensor(s) over bound {args.bound:g} or unmatched: {', '.join(failed)}")
        return 1
    print(f"RESULT: every compared tensor within bound {args.bound:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
