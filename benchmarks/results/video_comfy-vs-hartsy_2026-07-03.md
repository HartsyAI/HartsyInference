# Video models — HartsyInference vs ComfyUI end-to-end (2026-07-03)

End-to-end wall-clock through the **SwarmUI API** (not the engine test harness): the same SwarmUI
generation request routed first to the ComfyUI backend, then to the HartsyInference backend, on the
**same GPU**. This is the user-perceived latency comparison behind the `BENCHMARKING.md` "within 2× of
ComfyUI" goal. Complements the in-engine kernel/pipeline harness (which times the C# path in isolation).

**Hardware:** RTX 4090 24 GB (CUDA 13.1, `LD_LIBRARY_PATH=~/.local/lib/cuda13`), driver via SwarmUI 0.9.8.1.
**Backends:** ComfyUI self-start (id 1, `GPU_ID=0`) vs HartsyInference (id 2, `GPU_ID=0`, `ComputeBackend=cuda`,
`NativeFp8Gemm=auto`, `CacheWeightCasts=off`). Only one backend enabled at a time so routing is deterministic.
**Workload (identical both backends):** 25 frames, 512×320, 20 steps, cfg 6 (Wan), `videoresolution=Image`,
h264-mp4. Warm = model already resident; **seed randomized per run** to defeat SwarmUI's identical-params
result cache (an identical-seed rerun returns the cached file in ~0.2 s and is NOT a real generation — an
early trap in this measurement). Warm number = avg of 3 real gens (2 for 14B). Wall-clock is full end-to-end
incl. VAE decode + video encode.

## Results — warm generation (model resident)

| Model | Quant | Comfy warm | Hartsy warm | Ratio (Hartsy/Comfy) |
|---|---|---|---|---|
| Wan 2.1 T2V 1.3B | fp16 | **6.28 s** | **67.61 s** | **10.8× slower** |
| Wan 2.2 TI2V-5B | fp16 | **4.52 s** | **37.90 s** | **8.4× slower** |
| Wan 2.1 T2V 14B | fp8_scaled | **30.62 s** | **180.39 s** | **5.9× slower** |

_(14B at 15 steps both backends; the two fp16 models at 20 steps. Same steps per model across backends.)_

Per-run detail (warm): Comfy wan1.3B 6.35/6.11/6.38 s, ti2v5B 4.70/4.42/4.45 s, wan14B 30.61/30.64 s;
Hartsy wan1.3B 67.66/67.88/67.28 s, ti2v5B 38.03/37.96/37.72 s. Variance is <1%, so the gap is structural,
not noise.

## Results — cold (first gen incl. model load + C# checkpoint convert)

| Model | Comfy cold | Hartsy cold |
|---|---|---|
| Wan 2.1 T2V 1.3B fp16 | 8.5 s | 88.8 s |
| Wan 2.2 TI2V-5B fp16 | 7.7 s | 67.8 s |
| Wan 2.1 T2V 14B fp8 | 38.4 s | 205.0 s |

Hartsy's cold path also carries a large safetensors-load + checkpoint-conversion cost (~20–80 s) on top of the
first denoise. Separate lever from steady-state; not the focus here.

## Diagnosis — the gap is host-bound, not raw compute

During Hartsy Wan generation the **4090 sits at ~5–11 % GPU utilization, ~80 W, with only ~1.7 GB VRAM live**
for the 1.3B fp16 model. The GPU is idle >90 % of the time — the pipeline is **host/launch-bound**, not
compute-bound. Root causes in `HartsyInference.Diffusion/Models/Denoisers/DiTBlocks/WanVideoBlock.cs` +
`WanDitOps.cs`:

1. **F32 compute path everywhere.** Every `Linear`/`RmsNorm`/`ScaledDotProductAttention` in the Wan DiT block
   runs in `DType.F32` (weights `LoadF32`'d). No F16 tensor cores — Comfy runs Wan in bf16. ≈2–4× on GEMM
   alone. (Per `docs/Research/CUDA_PERFORMANCE.md` gap table.)
2. **Per-op eager alloc + launch.** Each op does `new Tensor(...)` → one backend kernel → `.Dispose()`. A
   1.3B gen is ~40 forwards × ~30 blocks × ~10 ops = ~12k kernel launches + 12k GPU allocs, each with launch
   latency the tiny F32 ops can't amortize. This is why util is ~5 %: launch/sync latency dominates the actual
   math. Needs op fusion, a memory pool (no per-op alloc/free), and/or CUDA graphs.
3. **Host-side patchify / unpatchify loops.** `WanDitOps` patchify (`gt×gh×gw×inC×pt×ph×pw`) and unpatchify
   run nested `for` loops over `DataPointer` on the CPU, serializing host↔device around each forward.
4. **Timestep-embed per-group host loop** (`Buffer.MemoryCopy` per group) — minor, but same host-bound shape.

The ratio is worst on the smallest model (1.3B, 10.8×) and shrinks as the model grows (5B, 8.4× → 14B 5.9×) —
consistent with fixed per-op overhead being amortized over larger GEMMs.

**Two distinct axes, by VRAM regime:**
- **fp16 models (1.3B, 5B) — host/launch-bound.** GPU ~5–11 % util. The fix is F16 tensor cores + op fusion /
  pooling / CUDA graphs (above). This is the dominant, most-fixable gap.
- **14B fp8 — GPU-bound on redundant re-casts.** Here the GPU runs at **~93 % util, 174 W, 19.4 GB** — but on
  wasted work: `CacheWeightCasts=off` (required so 14B fp8 fits 24 GB; `on` needs ~28 GB → OOM) re-casts every
  fp8 weight to fp16 **every step**. So the 14B isn't launch-starved, it's burning the GPU on per-step casts
  Comfy doesn't pay. Fix axis is different: native fp8 GEMM (`NativeFp8Gemm` only helps when activations are
  fp8, which diffusion's aren't) or a cast cache that fits (fp8-resident + on-the-fly cast into a small reused
  buffer, not a full second fp16 copy).

## Verdict

Fails the "within 2× of ComfyUI" bar for Wan video by a wide margin (8–11×). This is the **optimization target**:
F16 tensor-core path + op fusion / memory pooling / CUDA graphs for the Wan DiT block. The DiT-block device port
that fixed Flux/SD3 (`dit-block-device-port-perf`: 20-min → ~99 s) has **not** been applied to the Wan video
blocks; this is the same class of fix.

## LTX-2.3 22B — block-swap-bound (separate regime)

Not run head-to-head vs Comfy (no comparable LTX-2.3 workflow wired on the Comfy backend here), but the Hartsy
timings frame the constraint. LTX-2.3 is a 22B DiT (~19.2 GB) that **block-swaps**: it streams all 48 blocks from host
RAM every forward, keeping only a ~1.2 GB resident window, so it fits 24 GB but pays host→device weight I/O per
forward.

| Config | Frames | Res | Steps | Wall |
|---|---|---|---|---|
| short clip | 25 | 512×320 | 20 | **451 s** (7.5 min) — completes, coherent video + real 48 kHz stereo audio |
| long/large | 177 | 704×448 | 30 | **did not finish in 30 min** (client timeout; ~60 forwards × 7 k tokens × 19 GB stream) |

Per-forward cost is dominated by the 19 GB stream (roughly constant) plus token-dependent compute, so wall-clock
scales with steps×cfg forwards and with resolution/frames. Practical only for short clips on 24 GB. Levers:
bigger resident window on higher-VRAM cards, or fewer/cheaper forwards.

## Optimization attempt #1 — device-side modulation (engine alpha.43.15-local)

Profiled a real Wan-1.3B gen with `HARTSY_PROFILE=1` (per-op CPU wall-time, dumped on backend Dispose). Pre-fix
top op was **`GatedResidual` 85 s / 7200 calls / 11.8 ms avg** — vs the structurally-identical `AffineBroadcast`
at 0.21 ms. Root cause traced to `GpuTransferHelper.CopyToDevice`: the cache-miss path does a **full
`SyncStream()`** before its H2D, and `WanVideoBlock.Modulation()` built the 6 shift/scale/gate tensors **on the
host** every block → every downstream `GatedResidual(residual, value, gate)` uploaded a fresh host `gate` → cache
miss → full stream drain of the async attention/FFN work. So `GatedResidual` absorbed the drained compute.

**Fix:** rewrote `Modulation()` (G=1 path) to build the 6 tensors **on-device** (`SliceRows` + `GatedResidualLastDim`),
so every downstream modulation upload is a cache HIT. Numerically identical (F32 add, host→device); output verified
frame-coherent.

**Result (Wan-1.3B warm, 4090):**

| | wall | GPU util | power | GatedResidual avg |
|---|---|---|---|---|
| before | 67.6 s | ~5 % | 80 W | 11.8 ms |
| after  | **62.7 s** | **~89 %** | 196 W | 8.3 ms (big adds) |

Only **~7 % wall** — but util went 5 %→89 %, i.e. the fix **saturated the GPU** and moved the system from
(apparently) idle-bound to compute-bound. The small wall gain corrected the diagnosis: the 85 s was **mostly real
async GPU compute drained at the sync point**, not pure sync waste — the CPU-wall profiler charges drained async
work to whichever op syncs. The workload was already largely compute-bound.

**Remaining anomaly (next target):** post-fix, the big `[S,dim]` `GatedResidualLastDim` residual adds are **still
~8.3 ms each** while the same-size all-device `AffineBroadcast` is **0.06 ms** — a ~130× gap that is no longer
input residency (both device). Suspects: (a) the large residual/value activations get **evicted** from the
activation cache under pressure (worsened by the fix's 6 extra cached modulation tensors/block) → re-upload +
`SyncStream`; (b) a slow `dit_gated_residual` kernel. Needs targeted instrumentation.

**Honest verdict on the 10× gap:** it is genuine compute — F32 everywhere (no F16 tensor cores), vanilla SDPA (no
flash-attention), no kernel fusion, plus the residual-add anomaly. F16 alone is ~2× (Linear was only 4.5 s of the
gen, so F16's ceiling here is modest until the residual-add + attention paths are fixed). Closing 5.9–10.8× is a
multi-front engine effort (F16 path for the ~8 F32-only elementwise/norm ops + flash-attention + the residual-add
fix), not a single change. alpha.43.15-local ships the device-modulation fix as step 1.

## Optimization attempt #2 — kill the per-miss stream drain (engine alpha.43.17-local) ★ 2.4×

Instrumented `CopyToDevice`'s cache-miss path with size-bucketed NVTX labels and re-profiled. The truth:

| op (2 gens) | calls | total_ms |
|---|---|---|
| **H2D_MISS_SMALL** | 32,816 | **94,023** |
| H2D_MISS_BIG | 837 | 2,597 |

The bottleneck was **32,816 small-tensor cache misses**, each doing a **full `cuStreamSynchronize`**. The Wan DiT
allocates ~14 small scratch/modulation tensors per block-forward that miss on first upload, and the miss path did
`Allocate (stream-ordered) → SyncStream() → synchronous cuMemcpyHtoD` — the `SyncStream` drained the *entire* async
pipeline on every one. `GatedResidual` looked slow only because it sat right after the attention/FFN work that the
next miss drained.

**Fix (shared, all-arch):** the alloc is already stream-ordered (`cuMemAllocAsync`); make the copy stream-ordered too
(`cuMemcpyHtoDAsync` on the same compute stream) and **drop the `SyncStream()`**. The copy is naturally ordered after
the alloc and before the consuming kernel; no CPU read depends on it here, so only stream order matters — which holds.

**Result (Wan-1.3B, 4090, warm):**

| | H2D_MISS_SMALL avg | GatedResidual avg | warm wall | cold |
|---|---|---|---|---|
| 43.16 (pre) | 2.87 ms | 2.81 ms | 62.7 s | 88.8 s |
| **43.17 (post)** | **0.19 ms** | **0.09 ms** | **28.1 s** | **41.4 s** |

**Wan-1.3B: 67.6 s (baseline) → 28.1 s — 2.4×.** Gap to ComfyUI 10.8× → **4.5×**. Post-fix the top op is finally
`Linear` (the real GEMMs) — the workload is now genuinely compute-bound, so F16 tensor cores are the legitimate next
lever. Output verified frame-identical (Wan) and coherent on an image arch (Z-Image Turbo) — the change is safe beyond
Wan since it's in the shared transfer helper; **every architecture that misses host tensors benefits**.

alpha.43.17-local ships: device modulation (#1) + stream-ordered miss H2D (#2). The device-modulation change (#1) is
retained but was a minor contributor; #2 is the real win.

## Optimization attempt #3 — the real bottleneck is attention, not GEMMs (alpha.43.20/.21)

Added a **sync-profiler** (`HARTSY_PROFILE_SYNC=1`: `cuStreamSynchronize` per NVTX range → per-op timing = true GPU
time, not async-launch cost — the CPU-wall profiler is blind to async GPU work, and Nsight isn't installed here).
True GPU-time breakdown (Wan-1.3B, 2 gens, sync-profiled):

| op | GPU ms | share |
|---|---|---|
| **SDPA** | 63,821 | **~half** |
| Linear (GEMM) | 15,852 | — |
| Conv2D (VAE) | 12,600 | — |
| GatedResidual | 9,710 | — |

Findings that redirected the plan:
- The DiT **GEMMs are already F16 tensor-core** (`ResolveGemmDtype(F32,F16)→F16`; Wan weights are fp16, input cast
  down). TF32 is already on (`_allowTf32`, SM≥8). So F16-on-the-GEMMs was a non-lever — TF32 test = no-op.
- **SDPA is ~half of GPU time**, running at ~5 % of the 4090's FLOPS. Not the GEMMs (already TF32 tensor-core inside
  SDPA) — the cost is the **materialized `[heads,Sq,Skv]` score matrix** (Wan-1.3B self-attn ≈ 963 MB, written by QK
  then re-read by softmax + AV ≈ 4 GB traffic/call) plus a per-head GEMM loop. The engine's flash kernel is a naive
  reference (re-reads K/V per row) — **slower** (forced-flash cold = 159 s), so it's not usable.

**Fix (43.20+):** run the non-tiled no-mask SDPA in **F16** — halves the score-matrix traffic + F16 tensor cores.
**OPT-IN (`HARTSY_SDPA_F16=1`), NOT default.** Verified fast + frame-coherent on Wan (fp16, RMS-normed Q/K → bounded
scores), but a default-on trial produced a **BLACK image on Z-Image fp8** — its unbounded pre-softmax scores overflow
F16. So it needs per-arch / per-call gating (enable only when Q/K are pre-normalized) before it can be default; the
universal safe default stays F32/TF32 attention.

**Result: Wan-1.3B warm 28.1 s → 23.7 s (min 22.6 s).** Cumulative **67.6 → 23.7 s = 2.85×**; gap to ComfyUI 10.8× → 3.8×.

**Shipped safely by default (43.23) via a per-arch gate:** added `bool allowF16=false` to `IBackend.ScaledDotProductAttention`;
the F16 SDPA path fires only when a caller passes `allowF16: true`. `WanVideoBlock` passes true (Q/K are RMS-normed →
bounded scores); Z-Image and other unbounded-score archs don't → they stay F32 → **no black output**. So Wan gets 23.7 s
**by default** while Z-Image stays correct (verified frame-coherent, seed 98765/55511, mean≈138). Env overrides:
`HARTSY_SDPA_F16=1` forces it on for all callers (testing), `HARTSY_SDPA_NO_F16=1` kills it globally.

## Where the remaining ~3.8× lives (next levers)

- **SDPA (still #1 even after F16):** the win-condition is a real **flash-attention** kernel (tiled, tensor-core,
  online-softmax, no materialized score matrix) — the engine's current one is a naive reference. **Detailed plan:
  [`../../docs/Research/FLASH_ATTENTION_PLAN.md`](../../docs/Research/FLASH_ATTENTION_PLAN.md)** — fused WMMA, **TF32-in
  / F32-accumulate** (safe for ALL archs incl. fp8 — architecturally fixes the F16 blackout), ~4 GB→~55 MB HBM traffic
  (~3-5× on SDPA → ~1.6-2× e2e at M1). NOTE: no `nvcc` on this box — PTX builds via `native/cuda/nvrtc_compile`.
- **VAE Conv2D** (12.6 s sync-profiled): separate from the DiT; its own optimization axis.
- **F16 activations end-to-end** to kill the ~25 k per-Linear F32→F16 input casts and halve elementwise traffic
  (needs F16 variants of the ~6 F32-only elementwise/norm ops).

## Coverage / verification context

All video models verified **coherent** through SwarmUI/Hartsy the same day (separate matrix): LTX-2.3 22B
(T2V+audio, real 48 kHz stereo), Wan 2.1 1.3B/14B, Wan 2.2 TI2V-5B / T2V low+high-noise, LTX-0.9 2B;
HunyuanVideo correctly refused. So this is a **speed**, not correctness, gap. (Wan pipeline still logs
"first-run-validation pending — numerics unverified vs reference".)

## Known robustness bug found while benchmarking

Interrupting a **block-swapped LTX-2.3** generation mid-stream (client HTTP timeout / `InterruptAll`) left the
Hartsy backend wedged "in use"; the subsequent backend re-init threw
`NullReferenceException` in `GpuTransferHelper.OnPromotedHostAccess` ← `Tensor.DrainPendingFinalizerGpuCleanup()`
← `CudaStreamingWeightCache..ctor` — stale pending finalizer GPU-cleanups from the disposed context. Only a full
SwarmUI process restart cleared it (`RestartBackends`/toggle could not). File as: draining pending finalizer
cleanups during new CUDA-context construction must be null-safe / context-scoped.

## LTX benchmark + optimization (2026-07-03, 4090)

| model | Comfy warm | Hartsy | notes |
|---|---|---|---|
| LTX-0.9 2B | **2.84 s** | ~89 s → **16 s** (RoPE) → **15 s** (F16 DiT) | 31× → **5.3× gap**; cold 23 s (was 159 s). Now launch-overhead bound |
| LTX-2.3 22B (audio) | n/a (no Comfy workflow) | ~434–551 s, block-swap-bound | split-rope now device-ported; coherent video + real 48 kHz audio ✓. Timing noise (contention) > any rope-level gain |

**Found + fixed (same host-loop patterns as Wan):** both `LtxVideo2Attention` (LTX-2.3) and `LtxVideoBlock` (LTX-0.9)
did the multi-head reshape (`ToBhsd`/`FromBhsd`), AdaLN modulation, shift-scale, and gated-add as **host `DataPointer`
loops**, and called `ScaledDotProductAttention` without `allowF16`. Ported to device ops (`Permute0213`, `SliceRows`+
`GatedResidualLastDim`, `AffineBroadcastLastDim`) + `allowF16: true` (LTX RMS-norms Q/K → bounded scores → safe).
Both verified frame-coherent (LTX-2.3 + real audio).

**LTX-0.9 RoPE device port (2026-07-04, alpha.43.29-local): 89 s → 16 s warm (5.6×).** Sync-profile (`HARTSY_PROFILE_SYNC`)
ranked LTX-0.9's true GPU cost: **H2D_MISS_BIG 26 s (37 589 big-tensor cache misses) > Linear 22 s > SDPA 20 s**. The
misses were the one host loop left in: `LtxRope.ApplyRotary` reads `qn`/`kn` via `DataPointer` (D2H), rotates on the CPU,
then the next device op (`Permute0213`) re-uploads the whole `[S,dim]` Q/K — a big-tensor miss *per attention*, which also
caused the wild wall-clock variance (94–313 s). Fix: LTX's cos/sin already use the duplicated-pair layout
(`cos[2j]==cos[2j+1]`) that the proven `wan_rope_interleaved` device kernel reads (`coff = s·dim + 2i`, identical
rotation), so `ApplyRotary` was swapped from the host loop to `backend.WanRopeInterleaved(x, cos, sin, seq, 1, dim)` —
zero `BuildCosSin` change, Q/K stay GPU-resident. Verified frame-coherent (fox in sunlit snowy forest). Remaining gap to
Comfy's 2.84 s is real compute: materialized SDPA + Linear GEMMs.

**Re-profile after RoPE fix (alpha.43.29):** H2D_MISS_BIG collapsed **37 589→989 (26 s→2 s)** — fix confirmed. New top
GPU costs: **Linear 8.1 s > SDPA 7.2 s** (sync-inflated; ~10 s/gen real GPU vs 15 s/gen warm wall → ~1/3 of wall is
launch/host overhead, not GPU). LTX-0.9 ships **all-F32 weights** (908 tensors), so its GEMMs ran TF32 tensor cores
(~½ F16 throughput). **F16 DiT-weight cast (alpha.43.30):** `LtxVideoBlock` loads the 2D q/k/v/o + ff weights as F16
(→ `ResolveGemmDtype(F32 act, F16 w)=F16`, F32 accumulate = lossless reduction, same as Comfy's bf16; norms/bias stay
F32). Coherent. But only **16→15 s warm** because at 512×320×25f LTX-0.9 is just ~640 tokens → **launch-overhead bound**
(50 k+ tiny kernel launches/gen: 34 k Linear, 27 k Permute0213, 20 k RmsNorm), so halving GEMM throughput barely moves
wall-clock. F16 still worth keeping: **halves DiT VRAM (9.4→4.7 GB)** and the payoff grows with resolution/frames (where
compute dominates). Cold 23→32 s from one-time host F32→F16 weight cast (amortizes — load once, gen many). **Remaining
gap to Comfy is now structural: kernel-launch latency at small token counts → needs CUDA graphs or op fusion (a big
engine change), not more dtype/device-port tweaks.**

**FinalLayer device port (alpha.43.31):** the output `FinalLayer` still did a host `LayerNorm` + a host affine loop over
`[s,dim]` per step (then re-uploaded `normed` for the final Linear). Ported to `backend.LayerNormNoAffine` +
`AffineBroadcastLastDim` (folding the +1 into the scale) — the last per-step host loop in LTX-0.9's forward. Warm stays
14–15 s (no regression; confirms the launch-bound plateau), output coherent. LTX-0.9 forward is now fully device-resident.

**⚠️ GPU_ID footgun (recurred 2026-07-04):** backend #2's `GPU_ID` in `Data/Backends.fds` flipped `0→1` again between
deploys (concurrent edits / Swarm persistence), silently moving Hartsy onto the **RTX 3060** (SM 8.6) — a warm gen read
25 s instead of 15 s purely from the slower card. **Hartsy GPU_ID 0 = 4090, 1 = 3060** (opposite nvidia-smi's index).
Always confirm the init line says `RTX 4090, SM 8.9` before trusting a timing. Fix: `sed -i '92s/GPU_ID: 1/GPU_ID: 0/'
Data/Backends.fds` (kill Swarm first so it can't rewrite the file on exit), then restart.

**LTX-2.3 split-rope device port (alpha.43.32):** wrote a dedicated `ltx2_split_rope_f32` CUDA kernel (per-head cos
`[S,dim/2]`, one angle per `(i,i+headDim/2)` pair — matches `LtxVideo2Rope.Split`; `wan_rope_interleaved`/`dit_rope_f32`
don't) + `IBackend.Ltx2SplitRope` (host reference default, CUDA override); `LtxVideo2Rope.ApplyRotary` now dispatches
Split→`Ltx2SplitRope`, Interleaved→`WanRopeInterleaved`. **Verified: 2/2 gens coherent video + real 48 kHz stereo audio**
(fox in sunlit snowy forest). **Timing inconclusive:** 373 s and 551 s across the two runs — a >170 s swing that straddles
the ~434–451 s baseline, because LTX-2.3 is fundamentally block-swap-bound (fp8 22B streams 19 GB/forward on a 24 GB
card) and the shared-PCIe bus is contended by the 3060 (seen at 100% util). The rope re-uploads the fix removes are a few
hundred MB — negligible against 19 GB/forward — so the port is correct + architecturally cleaner (device-resident RoPE,
less PCIe + host traffic) but cannot produce a measurable wall-clock win here. **The only real lever for LTX-2.3 speed is
the block-swap itself** (streaming/compute overlap + prefetch, or more VRAM), not attention-level ops.

**Env gotcha (2026-07-04):** the NVIDIA driver's max PTX ISA dropped to 9.0 (was ≥9.2 on 07-03). The committed GGUF
matvec PTX (`mul_mat_vec_q4k/q6k/q8_0_f32.ptx`) were `.version 9.2` and are loaded *eagerly* at backend init → CUDA 222
PTX-JIT failure killed the whole backend. Rebuilt to 9.0 with `nvrtc_compile ... compute_80` (same as the flash kernel).
These are **still committed at 9.2 in git** — any fresh checkout will hit this until the 9.0 rebuilds are committed.
