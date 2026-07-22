# Inference Acceleration Grind — toward the hardware ceiling

Status legend: ⬜ todo · 🔧 in progress · ✅ done (CPU-verified) · 📊 GPU-measured · ⛔ blocked

**Goal.** Not "beat ComfyUI" — that bar is already met or nearly met fleet-wide (see
[MODEL_STATUS_IMAGE.md](MODEL_STATUS_IMAGE.md) §perf table). The target is the **hardware ceiling**: for
every hot kernel, measure arithmetic intensity and % of roofline
(`min(peak FLOP/s at dtype, intensity × peak GB/s)`) and close the gap with kernel + algorithmic work.
ComfyUI itself sits nowhere near its own roofline, so "faster than Comfy" and "near the physical limit"
are unrelated claims. Where the literature has no recipe, invent one and document it to paper standard
(per [CUDA_PERFORMANCE_PLAN.md](../Research/CUDA_PERFORMANCE_PLAN.md) methodology — Welch's t-test,
pinned stacks, committed CSVs).

**Approved scope (2026-07-21).** Stay layered on cuBLAS/cuBLASLt/cuDNN for GEMM/conv (a from-scratch
tensor-core GEMM core was considered and explicitly declined); attention is custom-PTX territory anyway
(cuBLASLt has no softmax). Workstreams:

- **A. Quantization on existing GEMM infra** — common-scale fp8 requant (fusion unblock), W8A8 INT8
  IMMA (native on SM 8.6), SVDQuant-style W4A4, Marlin-style fused-dequant W4A16 for GGUF.
- **B. Custom attention PTX** — SageAttention-v1 INT8 flash attention; content-adaptive sparse video
  attention (Wan/LTX); LLM GEMV access-pattern redesign + R4 repack.
- **C. Algorithmic (zero new kernels)** — across-step feature caching (TeaCache/FBCache family),
  limited-interval CFG, solver upgrades. Catalog: [STEP_ACCELERATION.md](../Research/STEP_ACCELERATION.md).

Method is the house rule: one change at a time, isolated microbench + e2e A/B, correctness gate before
speed, negative results recorded. **This box has no GPU** — everything lands CPU-verified + default-off,
and §GPU-HANDOFF below is the exact protocol for the GPU box.

---

## Session log — 2026-07-21 (foundation pass, no GPU)

Five deliverables landed, all CPU-tested, all default-off (unset env ⇒ byte-identical baseline paths).
**35 new tests, all green**; full regression sweep across Diffusion/LLM/Audio/ModelAssets/Core/Cpu shows
**zero regressions** (every failure observed also reproduces on a pristine `git archive HEAD` extract —
see §Pre-existing issues).

### C1 ✅ Across-step feature cache, device-resident (First-Block-Cache family)

The Feature-3 primitives (`FeatureCache`, LCM/TCD schedulers, `CfgHelper.IsGuidanceActive`) existed but
were wired into **zero** pipelines — and the CPU-side `FeatureCache` cannot be used on GPU activations at
all: its `DataPointer` reads evict the CUDA activation cache mid-forward (the Kandinsky
`AddInPlace(temb, pooled)` stream-drain pathology). This session built the device-resident version:

- [`native/cuda/dit/stepcache.cu`](../../native/cuda/dit/stepcache.cu) — `stepcache_rel_l1_f32/f16`:
  grid-stride shared-memory tree reduction computing Σ|a−b| and Σ|b| into 2 floats (8-byte D2H is the
  only sync, once per gated forward). 64-bit indexing (PHASE_3_DEVIATIONS #12), F32 accumulate for F16.
  **Not yet compiled** — no nvcc on this box; `build.sh` updated, compile is GPU-handoff step 1.
- `IBackend.RelativeL1Distance` (default host impl, F32+F16) + `IBackend.SupportsDeviceStepCacheGate`
  (default false; CpuBackend true; CudaBackend true iff `stepcache.ptx` present — graceful when missing).
- `CudaBackend.RelativeL1Distance` override + `CudaKernels` optional-module load +
  `LaunchStepCacheRelL1`.
- [`DeviceFeatureCache`](../../src/HartsyInference.Diffusion/Utilities/DeviceFeatureCache.cs) — gate +
  residual store/apply entirely through IBackend ops (Scale −1 + Add), fresh-tensor snapshots (avoids
  the in-place re-cache pitfall, PHASE_3_DEVIATIONS #17), accumulated-drift threshold + consecutive-reuse
  cap, per-CFG-stream instances.
- **Reference wiring: Qwen-Image** (`QwenImageTransformer.Forward` optional `stepCache` param — block 0
  always runs, its img stream gates; hit = skip blocks 1..59, final = block0 + cached residual;
  `QwenImagePipeline` cond/uncond instances + stats logging). Env: `HARTSY_STEP_CACHE=<threshold|1>`
  (1 ⇒ 0.15), `HARTSY_STEP_CACHE_CAP` (default 3).

**Numerical finding worth recording:** residual reconstruction `in + fl(out − in)` is exact to one float
rounding per element but NOT bit-exact (IEEE 754 does not make `a+(b−a)==b`). Tests pin: miss-path and
recompute-path **bit-identical** to the uncached forward; hit-path within 1e-4 on the synthetic config;
null-cache **byte-identical** code path. `StepCacheAccelerationTests` (24 tests) — includes a tiny-config
(Depth 2, hidden 256) synthetic QwenImageTransformer CPU forward exercising the real block-loop wiring +
anchor lifetime.

Expected (literature, to be GPU-measured): ~1.4–2× at threshold 0.1–0.15 on 20-step models
(TeaCache/FBCache report up to 2× image, 4.4× video at quality-neutral settings).

### C2 ✅ Limited-interval CFG (arXiv:2404.07724, generalizing ACE-Step 1.5's cfg-interval)

- [`GuidanceInterval`](../../src/HartsyInference.Diffusion/Utilities/GuidanceInterval.cs) — normalized-σ
  band record; `Parse`/`FromEnvironment` (`HARTSY_CFG_INTERVAL=lo,hi`; malformed values THROW — a
  silently-ignored perf knob would invalidate an A/B).
- Wired into Qwen-Image (both drain-free and host branches): outside the band the uncond forward is
  skipped and the step runs cond-only (`CfgEulerStep` gw=1). Skip counter logged. The paper measures a
  quality IMPROVEMENT (ImageNet-512 FID 1.81→1.40) plus the compute saving; per-model band tuning is a
  GPU-handoff item (start `0.1,0.85`).
- Interaction with C1 is self-healing: skipped uncond steps stale the uncond cache's indicator → drift
  spikes on re-entry → forced recompute.

### A1 ✅ Common-scale fp8 requant (the QKV/w1w3 fusion unblock)

`CheckpointConvertUtils.RequantizeToCommonFp8Scale(params Tensor[])` — in-place rewrite of a group of
fp8_scaled (E4M3) tensors to one common `Fp8ScaleFactor = max(sᵢ)`, the enabler for concat-fusing
separately-scaled projections into a single GEMM. This was explicitly ruled out fleet-wide ("fused QKV is
OFF the table for fp8-scaled checkpoints: one GEMM alpha can't represent 3 scales" — Chroma round 3;
Ideogram4's "w1/w3 fusion needs common-scale fp8 requant at load" named it as the missing piece).

**The math** (documented in the XML doc + pinned by `Fp8CommonScaleRequantTests`, 6 tests): E4M3 is
floating-point, so relative precision is scale-invariant for values that stay NORMAL after rescaling —
those round-trip within half an E4M3 ulp (amax-normalized error ≤ 1/16, and **exactly 0** for
power-of-two scale ratios). Only weights below `s*·2⁻⁶` go subnormal (progressive loss), below `s*·2⁻¹⁰`
flush to zero — negligible GEMM energy at real checkpoint ratios (≤ ~8×). Returns the introduced
amax-normalized max error so converters can gate.

**Not yet wired into any converter** — per-model fusion (Chroma to_q/k/v, Ideogram4 w1/w3) changes the
transformer forward and needs GPU e2e verification; recipe in §GPU-HANDOFF.

### B1 ✅ (reference + design) INT8 SageAttention — algorithm validated, kernel deferred to GPU box

`SageAttentionReferenceTests` (5 tests, Cpu.Tests) validates the exact numerical scheme the future
`sage_attn_int8` PTX kernel will implement, before any kernel code exists:

1. **K-smoothing is a softmax invariant** — subtracting K's per-channel mean changes q·k logits by a
   per-query constant only (< 1e-4 attention delta, pure re-rounding).
2. **Outlier-channel recovery** — on channel-consistent K outliers (the documented DiT activation
   pathology), plain per-row INT8 collapses; mean-subtraction recovers > 4× of the error and lands under
   the 1e-2 budget at which SageAttention reports metric-neutral e2e results.
3. Holds at head_dim 64 and 128 (the fleet's shapes). CPU-backend SDPA cross-checked against the
   reference (diff-target validity).

Writing the fused flash kernel blind (no nvcc, no GPU) was deliberately NOT done — kernel design notes in
§GPU-HANDOFF; the reference here is its correctness oracle. Ampere-native (INT8 IMMA ≈ 2× FP16 rate on
SM 8.6) — the highest-leverage custom-attention item for the 3060 AND attacks Wan's measured 2.0 s/step
SDPA share (27%).

### Hardening ✅ (found by this session's work)

- `MatMulKernels.LinearTransB` now **fails fast** when `output.ElementCount != M·N` — previously a
  wrong-shaped weight wrote past the output buffer: silent native heap corruption (0xC0000374, crashes
  the test host with no stack). Cost: one multiply+compare per call. This exact failure ate the first
  hour of test debugging (a wrong synthetic modulation-weight shape) and is the same crash signature as
  the SyntheticSmoke-quarantined CPU rollouts — some of those may be THIS bug class, worth re-running
  once fleet CPU paths are exercised.

### Pre-existing issues found (NOT from this session's changes — all reproduce at pristine HEAD)

| Issue | Evidence |
|---|---|
| `HunyuanVideoDitTests.Forward_ProducesFiniteVelocityOfLatentShape` fails: test builds `txt_in.weight` but `HunyuanVideoTokenRefiner.LoadWeights` wants `txt_in.input_embedder.weight` (test/loader key drift) | identical KeyNotFoundException on `git archive HEAD` extract |
| 5 Audio failures (`EnglishG2PTests` ×2, `AudioTextFrontendTests` ×3): IPA rendering + BOS-prepend drift vs expected token ids | identical 5 failures on pristine HEAD |
| Diffusion.Tests run ends "Test Run Aborted" (host teardown crash) even when all tests pass | reproduces on pristine HEAD |

---

## GPU-HANDOFF — exact protocol for the GPU box (4090/3060)

> Prereqs: CUDA toolkit (nvcc) on PATH, the standard model fleet staged, repo built `-c Release`.
> House rules apply: warm A/B same process where possible, seed 42, "A photograph of an astronaut riding
> a horse", 5 trials, record per [PROFILING_METHODOLOGY.md](../Research/PROFILING_METHODOLOGY.md), commit
> results under `benchmarks/results/` (`YYYY-MM-DD_accel_<item>_<gpu>.md` + raw CSVs). Every A/B needs
> its correctness gate BEFORE its speed number. Cloud boxes: [CLOUD_GPU_RUNBOOK.md](../../benchmarks/CLOUD_GPU_RUNBOOK.md).

### H0 — Roofline instrumentation (do first; changes what we prioritize next)

- [ ] `bash benchmarks/run_benchmarks.sh` baseline on this tree (all knobs unset) — confirms zero drift
      vs the committed baselines before any knob flips.
- [ ] **`ncu` access check**: `ncu --version` and a smoke `ncu --set full` on any kernel. On GeForce this
      needs driver perf-counter permission (`NVreg_RestrictProfilingToAdminUsers=0` or admin). This has
      been the LLM GEMV redesign blocker (LLM_DECODE_PERF_GRIND "needs ncu, blocked by GeForce perms") —
      on a cloud box it works out of the box; **capture for the top-5 kernels per model**:
      `dram__throughput.avg.pct_of_peak_sustained_elapsed` (bandwidth roofline %),
      `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` (tensor-core %),
      `sm__throughput.avg.pct_of_peak_sustained_elapsed`. These three numbers per kernel ARE the
      %-of-ceiling table this whole grind optimizes against — put them in the results doc.

### H1 — Step cache (C1) bring-up + measurement

1. [ ] Compile: `cd native/cuda/dit && ./build.sh` (compiles `stepcache.cu` → installs
       `src/HartsyInference.Cuda/Ptx/stepcache.ptx`; csproj globs it automatically). Rebuild.
2. [ ] Gate: run `StepCacheAccelerationTests` on the GPU box — CPU tests must stay green; then verify
       `CudaBackend.SupportsDeviceStepCacheGate == true` (log line appears when `HARTSY_STEP_CACHE` set).
3. [ ] Kernel numerics: A/B `RelativeL1Distance` CUDA vs the IBackend host default on identical random
       F32 + F16 tensors (~1e-6 F32 / ~1e-3 F16 agreement expected; atomic-order nondeterminism is fine
       at that tolerance). Add as a `GpuIntegration` test.
4. [ ] **Qwen-Image A/B** (the reference wiring; warm, 1024², 20 steps, cfg 4, seed 42, ×3):
       - baseline (unset) → confirm ≈ 39.4 s and byte-stable across the 3 runs;
       - `HARTSY_STEP_CACHE=0.1`, then `0.15`, then `0.2`;
       - record: wall, per-stream compute/reuse counts (logged), and **quality**: SSIM vs baseline image
         + eyeball. Acceptance: SSIM ≥ 0.95 at the shipped default; pick the default from the knee.
       - Watch VRAM: the cache holds prevIndicator + residual per stream (~58 MB × 4 at 1024²) — confirm
         peak stays under budget beside the resident DiT.
5. [ ] Replicate the wiring (same pattern: optional `stepCache` param after the first block, per-stream
       instances in the pipeline) to, in order: **Chroma** (biggest open image gap, 1.7×; note its
       persistent CFG-pair step graph must be BYPASSED when the cache is armed — variable per-step
       topology can't replay a fixed graph; eager fallback exists), **HiDream** (25 st, eager),
       **Wan T2V/I2V + LTX-2.3** (the big wins — video steps are 1.8–2.0 s each; TeaCache-class results
       are 2–4.4× on video; wire into `WanVideoBlock`-level forward the same way).
6. [ ] Negative-result discipline: if a model's gate never fires below quality-loss thresholds, record
       that in the results doc with the drift trace (the polynomial-rescaled TeaCache gate is the
       documented upgrade path — per-model coefficient fit, STEP_ACCELERATION §2.3).

### H2 — CFG interval (C2) measurement

1. [ ] Qwen-Image warm A/B: baseline vs `HARTSY_CFG_INTERVAL=0.1,0.85` vs `0.15,0.9` (20 st, cfg 4).
       Expected: wall drops ∝ skipped uncond steps (logged); quality NEUTRAL-OR-BETTER (the paper's
       claim — verify SSIM + eyeball; if quality dips, shrink the excluded tails).
2. [ ] Composability run: interval + step cache together (they compound: interval halves gated steps,
       cache skips block stacks on the rest).
3. [ ] Replicate to HiDream (cfg 5 → biggest absolute saving per step) and Wan (2-forward CFG at
       1.8 s/forward — the largest per-step win in the fleet). Same self-healing note re: C1.

### H3 — fp8 common-scale fusion (A1) wiring + measurement

1. [ ] **Chroma QKV**: in `ChromaCheckpointConverter`, after the fp8 scale pre-pass, call
       `RequantizeToCommonFp8Scale(to_q.weight, to_k.weight, to_v.weight)` per block (log the returned
       error; gate < 1/16), then enable the existing fused-QKV path (ZetaChroma's split-attention fuse
       shipped `44.53` — same transformer family) for fp8-scaled checkpoints. Gate: seed-42 A/B image
       SSIM ≥ 0.99 vs unfused (the requant error bound predicts visually-identical output; verify).
       Measure: step time (3 GEMM launches → 1, larger-N GEMM efficiency).
2. [ ] **Ideogram4 w1/w3**: same recipe on the FFN pair (its qkv is already fused in the checkpoint);
       this was its named next lever. Also try its remaining menu (BSHD strided cuDNN SDPA) same session.
3. [ ] If the SSIM gate fails anywhere: record which blocks carried extreme scale ratios (the helper
       returns per-group error — log it per block) and fall back per-block (fuse only clean groups).

### H4 — INT8 SageAttention kernel (B1) — the build item

Design (validated by `SageAttentionReferenceTests`, which is the diff oracle):

- **Precompute per head, per forward**: K per-channel mean over the sequence (one reduction kernel or
  folded into the projection epilogue); subtract during the K-quant pass.
- **Quant granularity**: per-row (token) absmax → int8, scales in F32. Q rows and K̄ rows.
- **Kernel**: non-causal flash loop (the diffusion simplification — no mask, fixed seqlen, see
  DEEP_KERNEL_OPTIMIZATION §2): tile Q rows per block, stream K̄/V tiles; QK^T via `mma.sync.m16n8k32`
  s8×s8→s32 (`dp4a` fallback for odd tiles), dequant `s32 · qScale·kScale·softmaxScale` into the online
  softmax running max/sum in registers; PV in F16 with F32 accumulate. SMEM XOR-swizzle; `cp.async.cg`
  2-stage. Target shapes first: head_dim 128, seq 4–17k (Qwen/Wan/LTX), then 64.
- **Placement**: `native/cuda/attention/sage_attn_int8.cu` (the dir PHASE_B reserved), new optional
  module in `CudaKernels` (same pattern as stepcache), dispatch in
  `CudaBackend.ScaledDotProductAttention` behind `HARTSY_SAGE_ATTN=1` (default off), shape-gated
  (no-mask + head_dim ∈ {64,128}), falling through to cuDNN otherwise.
- Gates, in order: (1) kernel vs `SageAttentionReferenceTests`'s int8 reference math on identical inputs
  (exact int32 dots → ~1e-6 agreement of dequant scores; e2e attention ≤ 1e-2 vs F32 on outlier
  distributions); (2) `SdpaGpuBenchmarks` A/B vs the cuDNN path per shape (claim requires t-test α=0.01);
  (3) e2e: Wan T2V 14B step time (SDPA is 27% → success ≈ −10–14% step) + SSIM/eyeball; Qwen/Chroma next.
- Expected: ~2× attention-op speedup (SageAttention v1 measured 2.1× over FA2 on SM 8.6-class hardware).

### H5 — Roofline follow-ups (prioritize by H0's measured table)

- [ ] **LLM GEMV redesign** with `ncu` finally in hand (cloud box): capture the Q4_K GEMV's
      `dram__throughput` % — the doc's standing estimate is ~22% of bandwidth vs llama.cpp's ~80% on
      Mistral. The scoped attack: R4 row-interleaved repack (4 rows' quant blocks contiguous → coalesced
      warp reads + input-vector reuse) — design in LLM_DECODE_PERF_GRIND "dotLLM techniques". Derive the
      thread→byte mapping from the measured transaction sizes, don't guess.
- [ ] **W8A8 IMMA path** (ViDiT-Q recipe, QUANTIZATION_LOW_PRECISION_INFERENCE §5/§Top-recs Phase B):
      needs the COL32/COL4_4R2_8C layout plumbing + timestep-aware calibration harness — biggest
      remaining quantization build; only start after H1–H4 land (they're cheaper per unit speedup).
- [ ] **Sparse video attention** (Wan/LTX): first MEASURE per-layer attention-score concentration on
      real generations (dump per-layer attention entropy / top-k mass on a few steps — cheap
      instrumentation pass) — the content-adaptive budget is the novel claim vs fixed Radial/STA
      patterns; design after the measurement, not before.
- [ ] Wan2.2-Lightning / LTX-distilled loadable accelerators (STEP_ACCELERATION §5.1): LoRA merge +
      4-step scheduler + CFG off — near-free 4–20× for users who accept distilled outputs; wire loaders.

---

## Results ledger (GPU box fills; keep negative results)

| Date | GPU | Item | Config | Baseline | New | Δ | Quality gate | Result dir |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
