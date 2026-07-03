# HunyuanVideo 13B T2V — end-to-end optimization benchmark (2026-07-02)

Model-level wall-clock record for the HunyuanVideo optimization arc (not a Phase-B kernel harness run —
these are `HunyuanVideo_Gpu_T2V_ShortClip` / `HunyuanVideo_Gpu_VaeDecode_FullRes` timings on real weights).

**Hardware:** RTX 4090 24 GB (CUDA 13.1, `LD_LIBRARY_PATH=~/.local/lib/cuda13`, `CUDA_VISIBLE_DEVICES=0`).
**Workload:** 25 frames, 512×320, 20 denoise steps, embedded-guidance 6.0, seed 42, prompt "A cat walks on
the grass, realistic style." Conditioning: LLaVA-Llama-3-8B fp8 (layer −3, template+crop-95) + CLIP-L pooled.
**Checkpoints:** DiT `hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors` (Kijai, 13.2 GB identity-scale
fp8; morning baseline used the 24 GB bf16), VAE `hunyuan_video_vae_bf16.safetensors`.

## Denoise step time

| Config | s/step | Change that got there |
|---|---|---|
| bf16 block-swapped, host-glue blocks (session start) | ~75 | — |
| + GPU-resident Qwen-recipe blocks | ~19 | `HunyuanImageBlock`/`SingleBlock` → backend ops |
| + fp8-resident DiT (no block streaming) | ~16.5 | Kijai fp8 ckpt + `NormalizeTencentRaw` + size-based residency |
| + GPU RoPE (`HunyuanImageRope.ApplyGpu` → `WanRopeInterleaved`, pre-permute) | ~7.0 | removed per-block Q/K D2H + host trig |
| + `FreeActivations(trimPool:false)` per step | ~6.25 | pool reservation reused instead of per-step release/re-map |
| + `HARTSY_EPILOGUE_FUSION=1` (opt-in) | ~6.08 | bias fused into cuBLASLt epilogue (fallback path) |
| + `HARTSY_FP8_NATIVE=1` (opt-in) | **~2.15** | dynamic e4m3 activation quant → fp8 tensor cores, zero weight casts |

## VAE decode (full-res tiled, 7×40×64 latent → 25f 512×320)

| Config | Wall |
|---|---|
| Per-frame host loop (replicate padding disqualified the batched path) | ~9 min |
| Batched replicate-pad fast path + GPU `Vae3dLayout` + tile trims removed + row-sequential blend | **~9.6 s** |

## Full e2e (load + text-encode + 20 steps + decode + frame write)

| Config | Wall |
|---|---|
| Session start (bf16, host glue) | ~20 min |
| fp8-resident + GPU RoPE + VAE fast path | 3m09s |
| + pool-trim optimization | 2m49s |
| + `HARTSY_FP8_NATIVE=1` | **1m26s** |

## Quality gates (all at seed 42)

- DiT parity vs diffusers held through every migration: per-stage relL2 ~1e-6 (`HunyuanVideoDitParityTests`).
- VAE batched-vs-per-frame: corr=1.000000, maxAbs 7e-5 (`HunyuanVideo_Gpu_VaeDecode_BatchedMatchesPerFrame`).
- GPU RoPE / trim changes: frames byte-comparable to prior run (≤2/255 accumulation-order noise).
- `HARTSY_FP8_NATIVE`: frames visually identical, mean |Δ| ≈ 1%/255 vs F16-fallback baseline (pure e4m3
  activation-quant noise; no DC bias/darkening — embedded guidance, no CFG amplification).
- fp8 GEMM micro-parity: `Fp8NativeGemmTests` — pure-fp8 rel_err 7e-5 vs F16 fallback; F32-activation
  dynamic-quant rel_err ~2.5e-2 (= quantizer noise floor); quantizer exact scale, ≤half-ulp per element.

## VRAM

- DiT resident: 13,262 MB preloaded, no block streaming; 21.7 GB free at the pre-decode stage transition.
- Full-res tiled VAE decode peak unchanged (~5–7 GB working set).

Open perf levers: unfused SDPA (~1 s/step, 2 GB score buffers per block), fp8 bias-add not yet fused into
the cublasLt epilogue, `trimPool:false` port to the other ~25 per-step FreeActivations call sites.
