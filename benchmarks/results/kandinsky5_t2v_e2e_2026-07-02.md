# Kandinsky-5.0 T2V Lite (2B) — first e2e + same-day optimization benchmark (2026-07-02)

Model-level wall-clock record for the Kandinsky-5 T2V bring-up (`Kandinsky5_Gpu_T2V_ShortClip` on real
weights; companion to `hunyuanvideo_e2e_2026-07-02.md`).

**Hardware:** RTX 4090 24 GB (CUDA 13.1). **Workload:** 25 frames, 512×512, 30 steps, CFG 5.0 (2 forwards/step),
seed 42, pre-computed Qwen2.5-VL/CLIP-L embeddings (snow-leopard prompt).
**Checkpoints:** `Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers` transformer (BF16→F16, ~3.7 GB) + the shared
HunyuanVideo 3D VAE.

## Denoise step time

| Config | s/step | Change |
|---|---|---|
| First e2e (blocks already GPU-resident; host RoPE) | ~15.6 | — |
| + GPU RoPE (`Kandinsky5Rope.ApplyGpu` → `WanRopeInterleaved`, pre-permute, memoized tables) | ~9.8 | removed per-attention Q/K D2H + host trig |
| + `backend.Add` for `temb + pooled` (was host `AddInPlace`) | **~2.9** | the host read evicted `temb`; every block's Silu re-uploaded it via sync pageable H2D = full stream drain per block |

## Full generation (30 steps + tiled VAE decode)

| Config | Wall |
|---|---|
| First e2e | 486.7 s (test total 8m16s) |
| Both fixes | **102.0 s (test total 1m51s)** |

## Quality gates (seed 42)

- Frames identical to the first-run golden across all changes: mean |Δ| = 0.038/255, max 3 (float
  accumulation order only); prompt-faithful temporally-stable snow leopard.
- CPU structural pipeline tests (6) green throughout.

## Profiling notes (the diagnosis, reusable)

- Blocking-mode profile was a red herring: it smeared the stall into `Permute0213` (5.6 ms avg) — a
  microbenchmark with a GPU-cached input proved the permute kernel runs at ~900 GB/s at every shape tested.
- The decisive signal was the **non-blocking** `HARTSY_PROFILE=1` table: `Silu` at 16.5 ms avg host time =
  the backpressure point where the CPU waited on the stream (sync H2D of the evicted 2 KB `temb`).
- New per-op labels added to CudaBackend: Permute0213, Silu, Gelu, GatedResidual, AffineBroadcast,
  SliceLastDim, LayerNormNoAffine, RopeInterleaved.

Open perf levers: unfused SDPA (~1.2 s/step at 7168 tokens; shared item with HunyuanVideo), VAE decode
~28 s per clip at 512×512×25f, CFG batching (uncond+cond in one batched forward).
