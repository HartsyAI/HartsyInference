# LTX Video Family — End-to-End Parity Report

> Status of the LTX-Video family (base 0.9 → 13B) and LTX-2 (dual-stream A/V) under HartsyInference,
> GPU-only on an RTX 4090 (CUDA device 0). "Verified e2e" = ran on real downloaded weights with the
> output confirmed correct. Generated 2026-07-01. Companion memory notes: `ltx-video-e2e-bugs`,
> `ltx2-19b-vs-23-divergence`.

## Summary

| Model | Weights | Runs e2e | Coherent output | Audio | Numeric parity |
|---|---|---|---|---|---|
| LTX-Video 0.9 (2B) | `ltx-video-2b-v0.9.safetensors` | ✅ | ✅ sunlit garden | n/a | visual (layer-diff pending) |
| LTX-Video 0.9.5 (2B) | `ltx-video-2b-v0.9.5.safetensors` | ✅ | ✅ cat in garden | n/a | visual (layer-diff pending) |
| LTX-Video 13B (0.9.7 dev) | `ltxv-13b-0.9.7-dev-fp8.safetensors` | ✅ | ✅ photorealistic (30 steps) | n/a | visual (layer-diff pending) |
| LTX-2.3 22B (video+audio) | `ltx-2.3-22b-dev-fp8.safetensors` + Gemma-3-12B fp8 | ✅ | ✅ cats in sunlit garden | ✅ decodes to waveform | ✅ DiT per-block relL2 ~1e-7 vs diffusers |

## Bugs found & fixed (with evidence)

1. **VAE latent denormalization missing** (LTX-Video 0.9 AND LTX-2 22B) — the diffusion model works in
   normalized latent space; the decoder must apply `raw = latent·std + mean` from the checkpoint's
   `per_channel_statistics` (`std-of-means` ranges ~0.11–1.41). Skipping it → lattice/blotches. Fixed in
   both VAE decoders + wired the stats through the generation entry.
2. **Caption projection rank-3 `l = Shape[0] = 1`** — `CfgHelper.SliceBatchElement` returns `[1,L,4096]`;
   the projection read the batch dim as the token count, collapsing the caption to one token + a GPU OOB
   write. Fixed: derive `l` from `ElementCount / lastDim`.
3. **Unmasked T5 padding** — cross-attention attended ~120 PAD tokens, diluting the caption. Fixed: truncate
   to real tokens (`SliceBatchElementPrefix`).
4. **Timestep-conditioned VAE (0.9.5/13B)** — built the 0.9.5 converter rename table (`VAE_095_RENAME_DICT`
   + `IsTimestepVae`) and generalized `LtxVideoVaeDecoder` to the residual channel-changing pixel-shuffle
   upsamplers; validated e2e (0.9.5 coherent; 13B reuses it).
5. **LTX-2 grouped `ConvTranspose1d` (audio vocoder)** — the BigVGAN depthwise anti-aliased upsampling uses
   `groups=768`, which the CUDA backend rejected. Fixed: added `groups` to the `conv_transpose1d_f32` PTX
   kernel (mirrors the grouped `conv1d_f32`; `groups=1` bit-identical), threaded through the launcher +
   backend. Audio now decodes to a 48 kHz waveform (verified: full pipeline runs audio decode with fail-loud
   restored — no exception).
6. **LTX-2 optional 2.3-only features** — the code targets 2.3; made prompt-modulation + gated-attention
   optional so both 2.3 and earlier LTX-2 variants load.

## LTX-2 video grid artifact — diagnosed and FIXED

Symptom: prompt-responsive (correct garden colors) + temporally varying, but a persistent 32-px lattice,
**identical at 8 and 30 steps** (ruled out under-denoising).

Root-cause hunt (each step decisive):
1. **VAE ruled out:** a fixed random latent decoded through `LtxVideo2VaeDecoder` on **CUDA vs CPU** is
   byte-identical (mean 87.1 / std 27.9) and yields a **smooth, textured, no-grid** image — so the VAE is
   correct and it's NOT a CUDA host-glue/cache bug. Grid ⇒ the DiT.
2. **DiT rope_type = split (the bug):** the 22B checkpoint config declares `rope_type = "split"`, but
   `LtxVideo2Rope` only implemented the **interleaved** apply. Interleaved rotates adjacent lanes `(2j,2j+1)`
   over the full dim with pair-duplicated `dim`-wide cos/sin; **split** rotates the two halves *within each
   head* — `(h·headDim+i, h·headDim+i+headDim/2)` — with compact `dim/2`-wide, front-padded, non-duplicated
   freqs (`apply_split_rotary_emb` in diffusers `transformer_ltx2.py`, dispatched by `attn.rope_type`).
   Applying interleaved where the weights expect split scrambles every token's positional phase → the 32-px
   lattice (colors survive because the text cross-attn carries no RoPE).

**Fix:** implemented the split rope in `LtxVideo2Rope` (added `RopeType`, per-head split apply + `dim/2`
front-padded freqs), set `LtxVideo2Config.V23.RopeType = Split`, threaded the correct per-rope head layout
(video self 32×128; audio/cross 32×64), fed `prompt_adaln` the unscaled sigma, and set `QkNormEps = 1e-6`.

### LTX-2.3 DiT per-block parity (tiny matched config, C# vs vendored diffusers `LTX2VideoTransformer3DModel`)

| Stage | relL2 |
|---|---|
| proj_in | 7.8e-8 |
| block 0 | 1.6e-7 |
| block 1 | 1.7e-7 |
| output velocity | 7.6e-7 |

All at f32 numerical-noise level ⇒ the dual-stream transformer forward is byte-faithful to the reference.
(Harness: `tests/HartsyInference.Video.Tests/Parity/ltx2_transformer_parity_dump.py` [ref, run with the
ComfyUI venv's diffusers 0.38] + `LtxVideo2ParityTests` [ours, CpuBackend]. Isolated-rope dump shows ~4e-4,
a diagnostic dump-layout artifact — the per-block outputs that *use* the rope match at 1e-7.)

**Result:** with split rope, the 32-px grid is gone — the full pipeline renders a coherent, prompt-faithful
scene (cats in a sunlit garden), temporally varying, and audio decodes to a waveform.

## Infrastructure built this session

- **nvrtc CUDA→PTX compiler** (`native/cuda/nvrtc_compile.c`) — nvcc is not installed; used to rebuild
  `dit_f32.ptx`-class kernels and the grouped `conv1d_f32.ptx`.
- **Block-swap for LTX-2** — `IStreamingBlock` + `BlockStreamingController` (Flux pattern) stream the ~19 GB
  fp8 22B DiT in a ~1.2 GB resident window, so it runs on a 24 GB card (no OOM).
- **Generalized timestep-conditioned VAE decoder** (0.9.5/13B, shared).
- LTX-2 generation harness + `TestPaths.LtxVideo2` + Gemma fp8/tokenizer wiring + VRAM ordering (free Gemma
  before the DiT).

## Environment / repro

- GPU: RTX 4090 (device 0 under CUDA ordering), `LD_LIBRARY_PATH=~/.local/lib/cuda13`.
- Tests: `tests/HartsyInference.Video.Tests` — `LtxVideo*GenerationTests` (env-gated on weights + VRAM).
