# Video models — HartsyInference vs ComfyUI scoreboard

Canonical, single-source-of-truth scoreboard for video (T2V) diffusion models. Consolidates the
`video_comfy-vs-hartsy_*` campaign write-ups and the per-model bring-up benchmarks that formerly lived
as separate dated files in [`benchmarks/results/`](../results/) (now retired — this table is the
successor) into one table. Where multiple source runs covered the same model, the **freshest scoreboard
run wins** (07-11 over 07-08 over 07-03), unless a later per-model or per-feature result gave a more
precise number for that specific model — see Notes below for the one case where that applies (Wan2.2
TI2V-5B step-cache).

**Hardware:** RTX 4090 24 GB only — no video benchmarks have been run on the RTX 3060.
**Methodology:** end-to-end wall-clock through the **SwarmUI API** — the identical generation request
routed to the ComfyUI backend, then to the HartsyInference backend, on the same GPU, same request, warm
(model resident). This is the user-perceived latency gap, not an isolated kernel/pipeline timing. See
[`README.md`](README.md) for the engine's default performance profile and
how to reproduce these numbers. Standard workload (unless noted): 25 frames, 512×320, h264-mp4,
`videoresolution=Image`, seed randomized per gen to defeat SwarmUI's identical-params result cache.

## Results — warm generation (model resident)

| Model | GPU | HartsyInference | ComfyUI | Ratio | Date | Source |
|---|---|---:|---:|---:|---|---|
| Wan 2.1 T2V 14B (fp8, 15 steps) | RTX 4090 | 30.58 s | 30.62 s | 1.00× — tied (parity) | 2026-07-11 | video_comfy-vs-hartsy_2026-07-11.md |
| Wan 2.1 T2V 1.3B (fp16, 20 steps) | RTX 4090 | 11.22 s | **6.28 s** | 1.79× slower | 2026-07-11 | video_comfy-vs-hartsy_2026-07-11.md |
| LTX-0.9 2B (fp16, 20 steps) | RTX 4090 | 4.59 s | **2.84 s** | 1.62× slower | 2026-07-11 | video_comfy-vs-hartsy_2026-07-11.md |
| Wan 2.2 TI2V-5B (fp16, 20 steps) | RTX 4090 | 15.5 s | **4.52 s** | 3.4× slower | 2026-07-11 | video_comfy-vs-hartsy_2026-07-11.md |
| LTX-2.3 22B (video+audio, 20 steps) | RTX 4090 | 42.3 s | n/a — no comparable Comfy workflow | n/a | 2026-07-11 | video_comfy-vs-hartsy_2026-07-11.md |
| LTX-2.5 22B dev (video+audio, int8-convrot, 30 steps)† | RTX 4090 | **47.40 s** | **42.48 s** | **1.12× slower** | 2026-08-14 | bench_ltx25.py |
| LTX-2.5 22B dev, recommended profile (1280×736×145f, conv decoder, 20 steps, cfg 4.0, 24fps) | RTX 4090 | 153.18 s | **142.86 s** | **1.07× slower** | 2026-08-15 | ltx25_ab.sh |
| HunyuanVideo 13B T2V (fp8, 20 steps) | RTX 4090 | 1m26s e2e (~2.15 s/step) | n/a — no Comfy Hunyuan T2V workflow benched yet | n/a | 2026-07-02 | hunyuanvideo_e2e_2026-07-02.md |
| Kandinsky-5.0 T2V Lite (2B, 30 steps) | RTX 4090 | 102.0 s e2e (~2.9 s/step) | n/a — not yet wired through SwarmUI (in-engine text encoders pending) | n/a | 2026-07-02 | kandinsky5_t2v_e2e_2026-07-02.md |

Row count: 9. Bold marks the faster (lower-wall-clock) side of each head-to-head comparison; rows with no
ComfyUI baseline are left unbolded.

The 47.40s LTX-2.5 row predates the quality/perf fixes below; the recommended-profile row is the current
best-known configuration (conv decoder — the diffusion decoder is built and correct but not the shipped
default, see Findings below). Re-run `bench_ltx25.py` / `ltx25_ab.sh` (after checking the deployed extension
DLL against engine HEAD) to refresh either row.

**Before adding or trusting any row or delta here, read "The harness's noise floor" in Findings below.**

⚠️ **The deployed SwarmUI extension carries the sigma terminal stretch default-ON as of `715ee947`** — any
LTX-2.5 **quality** measurement taken before that deploy sampled a different schedule and is not comparable.
It does **not** void the perf rows: the stretch changes the sigma values, not the step count or the per-step
work, so wall-clock and ms/step figures taken before it stand.

## Findings that must not be re-discovered

Distilled from the LTX-2.5 perf/quality investigation (2026-08-12 through 2026-08-15). The full session-by-session
narrative — every hypothesis tried, every commit — is git history (`git log --oneline -- benchmarks/scoreboards/VIDEO.md
src/HartsyInference.Video src/HartsyInference.Cuda/Kernels/ltx25*`); this section keeps only what a future
optimization pass needs to not repeat dead work.

**Do not re-attempt (measured, refuted):**
- N-tiling the int32 GEMM accumulator into L2 — monotonically worse (no tiling 1457.2 ms/step, N=2048 1474.0,
  N=1024 1596.3).
- F16 RoPE cos/sin tables for the fused QK kernel — inside run noise (1461.0 → 1456.0 ms), not worth the precision cost.
- Batching the CFG pair into one batch-2 forward — attention projections 10% *slower*, not faster.
- Per-element rewrite of `ltx2_qk_norm_rope_headmajor` — 7% faster on the kernel, 0 ms of step time; the limiter
  is the scattered write, not bandwidth.
- L2-sizing the int32 accumulator row chunk — monotonically slower as the chunk shrinks.
- Forcing SageAttention INT8 at skv 4992 — slower (1777 vs 1749 ms/step); the default 12288 gate is correct.
- CUDA graphs / launch-overhead work on this path — the step is GPU-bound (SM 99–100%), not host-launch-bound.
- cuBLASLt algo selection caching/autotuning for the int8 GEMM — within run noise, and autotuning is *worse*
  (locks in a bad algo).
- Host-side driver-call caching (`cuMemGetInfo`, cuBLASLt descriptors) — a dead end; do not re-chase host-side
  launch/allocation cost on this path while SM occupancy is saturated.
- A second epilogue slab swizzle attempt — mechanically perfect, zero speedup. A bank-conflict counter is not a stall.
- Widening the fused-mma gate to admit `ffn_up` — gain is ~8× below the noise floor, not worth measuring.
- Pipeline-depth / ldmatrix / occupancy hypotheses for the fused mma kernel — all three refuted, including a
  measurement trap where `CU_FUNC_ATTRIBUTE_NUM_REGS` reads 4, not 0, for a resident kernel.
- Stream-K for the int8 GEMM — comfy-kitchen's shipped binary has zero `StreamK` symbols; drop it from consideration.
- Acting on `ncu`'s occupancy advice for the fused mma kernel — raising occupancy would raise L2 traffic, which is
  the actual constraint.
- Trunk NA-decoder kernel: a smaller 4×8 tile is slower, 128 threads/block is slower, and the "DRAM-bandwidth
  bound" diagnosis for it was wrong.
- Re-chasing attention itself for the Comfy gap — it's at 95–98% of the fp16 wall and Comfy pays the same cost;
  the real lever (INT8/FP8 attention) is blocked on VRAM headroom at this geometry, not throughput.
- A 23-run LTX-2.5 quality ablation sweep at the pre-fix sigma schedule was run and never written to disk beyond
  this line — re-measure if needed, do not re-cite a number that doesn't exist.

**Hard ceilings (measured, not inferred):**
- Bare cuBLASLt INT8 GEMM at LTX-2.5 DiT shapes: 568.7/575.3/668.0 TOPS cold on the 4090 — three separate fused-kernel
  attempts (custom mma, N-tiling, F16 tables) all failed to beat this.
- Max viable GEMM tile size for this kernel: 128×256 at 256 threads — 256×256 needs 256 accumulator
  registers/thread against a 128-register cap at 512 threads.
- Honest ceiling on the whole mma-GEMM optimization line: ~130 ms/step (~3.9 s over 30 steps) — even at that
  ceiling the total lands at ~43.5 s, still behind ComfyUI. This class of work narrows the gap, it does not close it.
- The int32 dequant round-trip cannot be fused by cuBLASLt — the ~203 ms/step it costs is real but unreachable
  without a CUTLASS-class fused int8 GEMM.
- **Harness noise floor (governs every future row here):** same build/seed run back-to-back spreads ~25 ms/step
  (sd across 4 reps: 17–20 ms), and the variance is per-process (not thermal — steps *within* one run vary
  ±5 ms). **A single run per arm cannot resolve anything under ~50 ms** — measure with `ltx25_ab.sh`
  (alternating arms, N reps, report mean/median/range), budget ~20 min for a credible 4-rep campaign, and don't
  spend it on a change predicted to be worth less than the spread. The SwarmUI-level (not CLI) harness is worse
  still — three N=3 warm campaigns spanned 8.4 s total, so a 2–5 s effect is invisible there. `Int8ResidentRowChunk`
  also polls free VRAM per call and must be pinned via `HARTSY_INT8_ROW_BUDGET_MB` for any A/B at large-token
  geometries, or nothing under ~150 ms/step is resolvable.
- **Never difference a cold measurement against a warm one** (inflated an early ComfyUI decode-cost estimate by
  a full model load) and **always record DiT residency with every SwarmUI number** — a run without
  `resident prefix 48, streamed 0` in its log is void; streaming mode starves the decode's VRAM-derived chunk
  budget and cripples the denoise, with no harness flag to catch it.
- **Cross-model warning:** the Sage F16 `sq`-floor fix (added for LTX-2.5's 151-query/17480-key attention, 2.53 ms
  vs cuDNN's 0.51 ms) changes output bytes on **any** model with short-query/long-key head-major attention that
  Sage used to serve. MiniMax-H3 is the obvious candidate and is unverified — nothing reverts the floor except
  editing the code; re-baseline before trusting a hash-gated H3 comparison.

**Recommended LTX-2.5 profile (2026-08-15, still the current answer):** 1280×736, 20 steps, cfg 4.0, conv
decoder, 24 fps — 159.4 s warm for 145 frames. 30/40 steps only add micro-detail; cfg 3.0/2.5 have oversharpening
halos; the diffusion decoder is 9.7× slower than conv for skin-smoothness only visible at ~3× zoom. Quality
parity with ComfyUI is reached at matched settings (both engines render clean, undistorted faces) but speed is
not — Hartsy is 7.2% slower warm at this geometry (see the Results row above), consistent with the 1.12×
conv-vs-conv gap at the smaller geometry.

**Known landmines relocated to `docs/Checklists/TROUBLESHOOTING.md`** (not scoreboard content, but real traps
found during this work): the LTX-2.5 diffusion-VAE symlink footgun (`HARTSY_LTX2_DIFFUSION_VAE=1` alone silently
falls through to the conv decoder unless the symlink is swapped, not added-alongside), the deployed SwarmUI
extension's stale refusal of the diffusion VAE (source not reconstructable from the current checkout — do not
patch casually), and the `((IBackend)this).X()` fallback idiom's infinite-recursion bug class (a second live
instance was found and fixed in `VulkanBackend.AffineBroadcastLastDim`).

## SeedVR2-3B restoration — bring-up baseline vs Python reference (2026-08-01)

Not a T2V row: restoration (`hartsy restore`), measured at the E2E-parity operating point — 9-frame
Big Buck Bunny 360p clip, 640×360-area output, 4090, N=5, 95% CI (Student-t df=4). Correctness is
settled separately (C# output ≡ Python at SSIM 0.99950 with injected reference noises — see
`PARITY_VERIFICATION.md`); this row is the SPEED baseline for the future perf pass.

| Impl | Shape | Wall (9 frames) | s/frame | Peak VRAM |
|---|---|---|---|---|
| Python reference | **warm in-process**, bf16, causal slicing, dit-offload | 1.45 s ± 0.09 | 0.161 | 17.6 GiB |
| HartsyInference (bring-up) | **cold CLI e2e** (process + 13.6 GB fp32 mmap load + ffmpeg decode/mux), fp32, host-math DiT | 44.00 s ± 0.27 | 4.89 | ~16 GiB |

**Read the caveats before quoting a ratio:** the runs differ in warmth (in-process warm vs full CLI
cold start), dtype (bf16 vs fp32), and DiT execution (torch device kernels vs the deliberate host-math
bring-up shape — window gather/scatter, RoPE, qk-norm, AdaSingle all CPU-side). From the E2E gate run,
pipeline-only C# time at this shape is ~52.7 s *including first CUDA touch*; the perf-pass levers
(device window pack/unpack, GPU RoPE à la `HunyuanImageRope.ApplyGpu`, F16 activations, graph capture)
are enumerated in `MODEL_STATUS_VIDEO.md` §SeedVR2 follow-ups. Matrix-scale numbers (25f, 960×540-area):
~14 s/frame, 17.1 GB peak, 7/7 clips green.

## MiniMax-H3 fl2va — DiT quantization builds (2026-08-12)

Not a T2V row: in-engine **step time**, not SwarmUI e2e. Same weights published at three precisions by
Comfy-Org, so this is a build-vs-build comparison, not an engine-vs-engine one. Workload is
`benchmarks/minimax_h3/h3_bench.sh`'s gold baseline — 141 frames @ 512×288, seed 1, 4090 (nvidia-smi
index 1), SwarmUI stopped, mean of steps 4..N.

| DiT build | File | Step time | Residency | Date |
|---|---|---:|---|---|
| `pruned_int8_convrot` | 20.97 GB | **5.807 s** (n=27, 5.781–5.868) | fully resident, 20.96 GB weights + 1.78 GB reserve vs 24.22 GB free | 2026-08-12 |
| `pruned_fp8_scaled` | 20.96 GB | 8.6 s | fully resident, 22.5 GB | 2026-08-05 |
| `fl2va_bf16` | 66.28 GB | ~90 s | streams per call | 2026-08-05 |

**Caveat on the fp8 row:** it is carried forward from its own bring-up session, not re-measured beside the
int8 run. Both used the same script and gold workload, but not the same day or the same driver/VRAM state,
so read the int8-vs-fp8 gap as indicative until the two are run back to back.

The int8 build is INT8 tensor-core (IMMA) work — activation ConvRot, per-row dynamic int8 quant, cuBLASLt
IMMA, dequant epilogue — against a weight that never leaves int8. Correctness is settled separately: the
GEMM chain matches comfy-kitchen's eager reference at relL2 5.1e-8–2.7e-7 with F32 activations
(`Int8ConvRotCudaParityTests`), and the generation's frames were inspected. Note that step time is
**insensitive to the row-chunk size** the path picks from free VRAM: int32 accumulation is exact and
order-independent, so chunking changes neither the result nor the arithmetic, only the transient footprint.

## Notes

- **LTX-2.5 22B dev is the first LTX-2.5 Comfy-vs-Hartsy row** — 1.34× slower, Hartsy 56.62 s vs Comfy
  42.25 s, quality-matched (both prompt-faithful, frames inspected on both sides). The first measurement of
  this row read 117.13 s and 2.77×; that was a stale deployed build, and the perf pass that followed it is
  distilled in Findings above. Benchmarking it required updating the live ComfyUI backend (was v0.28.0, no LTX-2.5
  support at all) to v0.32.0, and a temporary SwarmUI `SDModelFolder` root addition + two service
  restarts to route the split DiT/TE/VAE repack through Comfy's separate-VAE loader path instead of its
  bundled-checkpoint assumption — reverted after the benchmark, no lasting server config change.
- **Wan 2.1 T2V 14B is the only video model at parity with ComfyUI** (30.58 s vs 30.62 s) — first video
  model to catch Comfy. Per the campaign write-up it has reached its
  fp8 compute floor (CUDA-graph and batched-CFG closed out as dead ends with evidence), so parity is
  where it is expected to stay absent a fundamentally faster fp8 GEMM.
- **ComfyUI column is carried forward from the 2026-07-03 head-to-head** for every model that has one —
  ComfyUI's own performance did not change across engine versions, only the Hartsy side did (per the
  07-11 file), so reusing the 07-03 Comfy numbers against the 07-11 Hartsy numbers is valid.
- **Wan2.2 TI2V-5B step-cache is opt-in and NOT the shipped-default number in the table above.**
  `2026-07-22_accel_stepcache_wan_4090.md` measured
  1.18–1.55× speedups (44.1–57.7 s vs a 68 s warm baseline) via `HARTSY_STEP_CACHE`, but on a *different*
  workload (832×480, 33 frames, 50 steps — not the standard 512×320/25f/20-step scoreboard workload, so
  the 68 s baseline there isn't directly comparable to the 15.5 s row above). More importantly, the
  benchmark's own verdict is negative for the pinned gate: no threshold holds SSIM ≥ 0.95 (best case 0.88
  at 1.18×), because Wan's 50-step UniPC trajectory is chaotically sensitive to any reuse — outputs stay
  coherent and prompt-faithful but diverge from the un-cached seed. The engine ships this **default OFF**
  as a "fast non-reproducible sampling" opt-in, not a transparent accelerator; `PERFORMANCE.md` (retired) §1's
  default-on feature table and §6 experimental-switch table both omit `HARTSY_STEP_CACHE` entirely,
  confirming it is not part of the standard profile.
- **HunyuanVideo 13B and Kandinsky-5.0 T2V Lite have no ComfyUI baseline yet** — per the 07-11 scoreboard
  these are still open rows pending a Comfy Hunyuan T2V workflow and in-engine text-encoder wiring
  (Kandinsky-5) respectively. The numbers shown are engine-side e2e wall-clock only, from their
  2026-07-02 bring-up benchmarks (not re-measured on a later engine build in these sources).
  HunyuanVideo runs at ~2.15 s/step via
  fp8-resident weights + GPU RoPE + `HARTSY_FP8_NATIVE`.
- **LTX-2.3 22B has no comparable Comfy workflow on this box**, so its row is internal-progress-only:
  451 s (2026-07-03) → 95.5 s (07-08) → 42.3 s (07-11), a 10.7× cumulative improvement, block-swap-bound
  (streams ~19 GB/forward on a 24 GB card).
