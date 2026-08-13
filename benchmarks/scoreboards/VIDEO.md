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
| LTX-2.5 22B dev (video+audio, int8-convrot, 30 steps)† | RTX 4090 | 51.33 s | **42.25 s** | 1.21× slower | 2026-08-13 | bench_ltx25.py |
| HunyuanVideo 13B T2V (fp8, 20 steps) | RTX 4090 | 1m26s e2e (~2.15 s/step) | n/a — no Comfy Hunyuan T2V workflow benched yet | n/a | 2026-07-02 | hunyuanvideo_e2e_2026-07-02.md |
| Kandinsky-5.0 T2V Lite (2B, 30 steps) | RTX 4090 | 102.0 s e2e (~2.9 s/step) | n/a — not yet wired through SwarmUI (in-engine text encoders pending) | n/a | 2026-07-02 | kandinsky5_t2v_e2e_2026-07-02.md |

Row count: 8. Bold marks the faster (lower-wall-clock) side of each head-to-head comparison; rows with no
ComfyUI baseline are left unbolded.

## LTX-2.5 22B dev — first Comfy-vs-Hartsy head-to-head (2026-08-12)

† Off-standard workload, not the table's usual 25f/512×320/20-step smoke test — LTX-2.5 is a 22B model
worth a "decent length/quality" pass instead: 768×512, **97 frames** (~4.0 s @ 24 fps), 30 steps, cfg 3.0,
`ltx-2.5-22b-dev-transformer-int8_lean_convrot` (dev, non-distilled, comfy-kitchen int8-convrot on both
DiT and Gemma-4 TE, joint video+audio latent). N=5 warm reps + 1 cold, random seed per gen, same script
family as `swarm_video_bench/bench_t2v.py` (`bench_ltx25.py`, both backends routed through the SwarmUI API
one at a time, model resident). Cold: 79.14 s (Hartsy) vs 71.94 s (Comfy). Peak VRAM 21.5 GiB (Hartsy) vs
24.0 GiB (Comfy) — fully DiT-resident, no block-swap streaming (22B fits in int8 on this 24 GB card). Comfy
needed the `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot` TE Swarm auto-downloads (14.3 GB); Hartsy uses
its own staged files directly.

**Both sides are prompt-faithful — this row IS quality-matched.** An earlier revision of this row carried a
caveat that every Hartsy frame ignored the prompt and blamed the connector bug `5ad864c2` for still being
live at `06fb26c8`. That was wrong, and the cause was a **stale deployment, not a live defect**: the engine
DLLs in the extension's output folder had been built at 14:50, before both `aa8e6bc7` (the Gemma-4 tower
wired into the pipeline, 17:18) and `06fb26c8` (the `prompt_adaln` timestep-scale fix, 19:02). Rebuilt at
HEAD, the same prompt/seed renders an actual lighthouse-at-sunset scene — verified by inspecting frames from
both the CLI and the SwarmUI-produced mp4. **Check the deployed DLL's build time against the engine's HEAD
before attributing a bad generation to a code bug.**

### What moved it from 117.13 s to 56.62 s (2026-08-12 perf pass)

Measured one change at a time on the CLI at the same workload (cold wall 139.69 s → 82.56 s), then confirmed
end-to-end through the SwarmUI API. Every step carried its own correctness gate:

| change | effect | correctness gate |
|---|---|---|
| Rebuild at HEAD (deployment was a 14:50 build) | — | frames prompt-faithful |
| Reflect spatial pad folded into `wan_vae_build_padded` | VAE decode 38.14 s → 3.94 s | all 97 frames + audio **bit-identical** |
| `ApplyKeyframesAbsPos` moved on-device | −82 MB D2H per forward | output changes — the 2.5 keyframe marker had been silently dropped on CUDA |
| F16 block activations (`DitDtype.Act`) + new `ltx2_split_rope_f16` | step 2.563 s → 1.756 s | SSIM 0.9957–0.9966 vs the F32 build across the clip |
| Trim the pool before sizing the resident prefix | warm reps stop alternating 62.6/93.5 s | n/a (residency only) |
| BF16 conv-VAE decode (2 new kernels) | VAE decode 3.94 -> 2.89 s | SSIM 0.9983-0.9986 vs the F32-VAE build |
| Fused QK-norm+RoPE+head-major, and the per-head gate (4 new kernels) | step 1.743 -> 1.629 s | fused vs unfused chain pinned by unit test; SSIM 0.992-0.994 |
| Fused block RMS-norm + AdaLN shift/scale (2 new kernels) | step 1.629 -> 1.593 s | same |
| GELU folded into the int8 dequant epilogue (`LinearGelu`) | step 1.593 -> 1.572 s | fused vs Linear-then-Gelu, max abs err 3.9e-3 |
| Pin the per-generation RoPE tables as resident weights | step 1.572 -> 1.522 s | all 97 frames + audio **bit-identical** |

The reflect-pad one is the headline: `CausalConv3d.ReflectPadSpatial5D` was a scalar per-element C# loop that
read `DataPointer` (draining the whole activation D2H and freeing its device copy) and rebuilt a full-size
host tensor that the next op re-uploaded. Every LTX-2 VAE conv is built with `spatialReflectPad: true`, so all
42 conv forwards paid it, the last up-stage over ~313 M elements. It was invisible to `HARTSY_PROFILE`
because 126 of 189 `CudaBackend` ops had no `NvtxRange` — 40 have been instrumented since, and the rule stands:
**if a profile's op totals do not sum to the phase wall-clock, look for un-scoped ops before concluding anything.**

The pool-trim one is a residency trap worth remembering: `FreeMemoryBytes()` counts pool-retained blocks as
used, so a generation that sized its resident prefix straight after the previous one's VAE decode measured
~5 GB of phantom pressure, pinned 22 of 48 blocks and streamed the rest. Because that generation then never
filled VRAM it did not evict, so the next one pinned all 48 — a stable two-generation ping-pong that cost
~30 s on every other request and is invisible in a single-generation CLI run.

### The remaining gap is glue, not math — measured, not inferred

`Int8ConvRotGemmThroughputTests` times the **whole** resident int8-ConvRot `Linear` (activation ConvRot →
per-row dynamic int8 quant → cuBLASLt IMMA → dequant epilogue) at LTX-2.5's real DiT shapes:

| shape | per call | achieved | vs the 4090's ~330 TOPS dense INT8 peak |
|---|---:|---:|---:|
| FFN up, 4992×16384×4096 | 1.992 ms | 336.4 TOPS | **102%** |
| FFN down, 4992×4096×16384 | 2.003 ms | 334.5 TOPS | **101%** |
| attn q/k/v/o, 4992×4096×4096 | 0.564 ms | 296.8 TOPS | 90% |

**The GEMM path is at the hardware wall**, epilogue included — so the int32 IMMA-accumulator round trip is
already paid for inside those numbers, and a custom fused-dequant IMMA kernel has nothing left to win.
Summed over 48 blocks × 2 CFG branches these shapes come to ~0.81 s/step of irreducible GEMM against a
1.74 s step, so the reachable rest is **non-GEMM glue**. Six fused kernels have since taken the step to
1.593 s by collapsing that glue: `ltx2_qk_norm_rope_headmajor` (RmsNorm → split-RoPE → `Permute0213`, three
full passes over each [S, inner] tensor down to one), `ltx2_head_gate` (the per-head gate was expanded to a
full `[seq, inner]` tensor through a constant 0/1 GEMM and then multiplied — now one in-place broadcast), and
`ltx2_rms_modulate` (the affine-free RMS + AdaLN shift/scale pair each block runs six times). What remains is
~4,600 launches/step of `Modulation` row-slicing and the two surviving permutes per attention.

ComfyUI is doing the same arithmetic on the same silicon — `comfy/samplers.py:610` only skips the uncond
pass at `cond_scale == 1.0`, so at cfg 3 it runs the same 60 forwards — which puts its non-GEMM glue at
roughly 0.4 s/step against our 0.93 s. That difference is the whole remaining gap.

**Where the 1.572 s step stands after the fusion work** (profiled per-op, `HARTSY_PROFILE_SYNC`, so ~19% high):
`Linear` 1064 ms (55%) and `SDPA` 333 ms (17%) are BOTH at their hardware roofline — the GEMM per the table
above, and attention at ~47 TFLOP/step against the 4090's ~165 TFLOPS FP32-accumulate FP16 tensor-core rate.
Together that is ~1.17 s of the 1.593 s real step. The reachable remainder is ~0.42 s: `Ltx2QkNormRopeHeadMajor`
148, `GatedResidual` 65, `H2D_MISS_SMALL` 65 ms over 971 calls/step — **diagnosed and fixed**: `HARTSY_H2D_TRACE=1` (now logging small
misses too, not just megabyte-scale ones) showed the audio RoPE cos/sin tables `[101, 1024]` missing the cache
**1536 times per step**, ~620 MB of pure PCIe traffic. They are built on the HOST, so they are neither a
preloaded weight nor any device op's output, and at ~0.4 MB they are too small for auto-promote to rescue — the
big video tables at `[4992, 2048]` missed only twice. Pinning all six tables with `PreloadWeights` when they are
built (and freeing them on a grid re-size) took the step 1.572 -> 1.522 s with **bit-identical** output, `Permute0213` 50 (the two survivors per attention: V-in and
the output), `SliceRows` 31 (the ~3,850 `Modulation` row-slices), `Ltx2RmsModulate` 33, `Ltx2HeadGate` 27.

### 2026-08-13 pass — residency, and the measured ceiling on the rest

Warm mean **56.62 s → 51.33 s** (N=5, SwarmUI API, same workload/harness); cold 79.14 → 72.94 s. Ratio to
Comfy's 42.25 s: 1.34× → **1.21×**. Both changes are residency-only, so output is unchanged by construction.
Peak VRAM 21992 → 24021 MiB of 24564: the prefix now survives the decode, and that headroom is the price.

| change | effect | correctness gate |
|---|---|---|
| VAE decode evicts only the prefix TAIL it is short of (+1 block margin), not all 48 | −5.3 s warm (combined) | residency only — output unchanged |
| `_onesV`/`_onesA` added to `LtxVideo2Block.EnumerateWeights` | `H2D_MISS_SMALL` 3571 → 2419 calls | residency only — output unchanged |

The two were measured **together**, never separately; the `DiT preload+prime` drop (3494 → ~457 ms) accounts for
~3.0 s of the 5.3 s and the remainder is unattributed. The `ones` fix is also not fully closed: ~403 small misses
per step survive it, from a source not yet identified — a thread worth pulling.

**The +1 block of margin is load-bearing, and finding it needed the journal.** Freeing exactly the computed
deficit hit the warm mean 0.24 s faster (51.09 s) but logged `OOM on async first attempt: requested=594.0 MB,
free=42.9 MB` once per generation, mid-decode. Those retries appear **zero** times in the 2026-08-12 warm-rep
journal for the identical workload, so they were introduced by the change, not pre-existing: `decodeNeed`'s
104 B/px is a bracket that under-estimates the real peak, and the old free-everything eviction had been hiding
that error under ~21 GB of slack. With the extra block, warm reps log none and only the cold generation still
retries. A single-generation CLI run shows neither the win nor the retries — **check
`journalctl --user -u swarmui.service` across warm reps before believing a residency change is clean.**

Also verified, since the TE-miss path is the one 5 identical warm reps never exercise: two generations with
fresh prompts evict the prefix for the 14.2 GB Gemma encode, re-squeeze to 46 blocks, and settle there with no
OOM and no prefix ping-pong.

The eviction one is the same class of trap as the pool-trim above, and bigger. `decodeNeed` is
`max(3 GiB, frames·h·w·104 B)` = 3783 MiB here against 3392 MiB free — a **391 MiB** deficit that was being
paid by freeing the entire 21.5 GB resident prefix, so the next generation spent **3.5 s** in
`DiT preload+prime` re-uploading all 48 blocks to have reclaimed less than one block's worth. Freeing from the
END of the prefix (2 of 48 blocks here) leaves the pin at `_residentPrefixBlocks` describing the survivors, and
the next generation's idempotent `PreloadWeights` tops up only the freed tail. **Invisible in a
single-generation CLI run** — the CLI's `preload+prime` is a cold first load either way; only warm reps through
the API show it.

The `ones` one: the unit weights for the affine-free RMS pre-norms are built on the HOST in the block
constructor and were absent from `EnumerateWeights`, so — exactly like the RoPE tables above — they were
neither a preloaded weight nor any device op's output, and every one of the 8 norms per block per CFG branch
re-uploaded them (768 reads/step).

**The remaining gap is ~0.31 s/step of glue, and that is also the ceiling.** Per-op at the 1.510 s step
(`HARTSY_PROFILE_SYNC`, 1800 ms profiled vs 1510 ms real = 1.19× high; deflated below), the accounting closes
exactly: `Linear` ~850 + `SDPA` ~350 + glue ~311 = 1511 ms. Linear is at 101–107% of the 4090's INT8 peak and
SDPA at ~83% of its FP16/FP32-accum rate, so **glue is the entire addressable budget**. Driving it to *zero*
gives 1.20 s/step → 36 s of denoise + the ~5.8 s of non-DiT that remains after the fixes above = **~41.8 s,
against Comfy's 42.25 s**. So parity is reachable, but only at glue ≈ 0 — it is the asymptote of near-total
glue elimination rather than something any single change buys, which is worth pricing before starting one.
The largest single cluster left is layout, not math: `Ltx2QkNormRopeHeadMajor`'s scatter plus both surviving
`Permute0213`s are ~135 ms/step spent purely converting token-major ↔ head-major for SDPA. `CudnnSdpa` already
sets per-tensor strides (it builds Kᵀ by swapping them), so a strided BSHD descriptor could delete that
cluster without a transpose — the one untested lever with a measured mechanism behind it.

Verified like-for-like while establishing that: `ffprobe` on both backends' benchmark mp4s shows h264 97
frames **plus** an aac track, so Comfy is running the same dual-stream video+audio DiT, not a video-only path.

**Do not re-chase**, all closed by measurement this session:
- **Batching the CFG pair into one batch-2 forward** — the premise was that m=4992 is too small and the
  attention projections' 90%-of-peak is an m artifact. It is not. `Int8ConvRotGemmThroughputTests` at
  m=9984: `attn_qkvo` **1.244 ms / 269.2 TOPS / 82%** against 2×0.564 = 1.128 ms at m=4992 — batching makes
  the attention projections **10% slower**, not faster. The FFN shapes are a wash (107%/107% vs 107%/103%).
  The other two claimed wins do not survive either: row-wise batching leaves every elementwise glue kernel's
  total bytes unchanged (it halves launch *count*, not work — and `GatedResidual` at 62 ms/step is already
  within ~20% of its bandwidth roofline), and halving weight HBM traffic cannot be banked on top of a GEMM
  chain already measuring >100% of compute peak. Large refactor (batch-2 SDPA, `s % S` rope, batch-strided
  head-major emit, VRAM at 23878/24564), negative-to-negligible payoff.
- **Making `ltx2_qk_norm_rope_headmajor` cheaper per-element** — rewritten to hold the RoPE partner lane in a
  register (one thread per pair) and reduce via warp shuffles, dropping the 17 KB dynamic-shared request that
  capped it at ~5 blocks/SM. Passed the unfused-sequence test at both dtypes and bought **7% on the kernel
  (0.075 → 0.069 avg) and 0 ms of step time** (1509.4 vs 1510.6). Reverted. At ~428 GB/s of the 4090's ~1008
  it is not bandwidth-bound: the limiter is the **scattered write** `out[(h·seq+s)·headDim+d]`, which sprays
  32 separate 256 B regions across a 40 MB tensor per token. Anything that does not fix the head-major
  scatter will not move this kernel.
- **L2-sizing the int32 accumulator** — shrinking the row chunk so it fits L2 is monotonically *slower*
  (256/64/32/16 MB → 1768/1854/1920/2070 ms per step); the extra chunks cost more in launches and small-m
  GEMM efficiency than the round trip costs in bandwidth.
- **SageAttention INT8 at this sequence length** — `HARTSY_SAGE_F16_MIN_SKV=1024` to force it on at
  skv 4992 gives 1777 ms/step vs 1749 ms for the cuDNN fused flash path. The default 12288 gate is right.
- **CUDA graphs / launch-overhead work** — the step is GPU-bound, not host-launch-bound (`nvidia-smi dmon`:
  SM 99–100%, memory controller 78–80% throughout the denoise).

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
  broken down in the dedicated section above. Benchmarking it required updating the live ComfyUI backend (was v0.28.0, no LTX-2.5
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
