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

- [x] `bash benchmarks/run_benchmarks.sh` baseline on this tree (all knobs unset) — confirms zero drift
      vs the committed baselines before any knob flips. *(🔧 2026-07-22: launched `--skip-python
      --skip-e2e --trials 5 --tag h0_baseline_3060` on the 3060 — results land under
      `benchmarks/results/run_*_h0_baseline_3060/`; drift check + the per-model ncu top-5 capture remain
      open. Formal BDN SDPA table (incl. the new Sage INT8 column) recorded in
      [2026-07-22_accel_sageattn_3060.md](../../benchmarks/results/2026-07-22_accel_sageattn_3060.md).)*
- [x] **`ncu` access check**: `ncu --version` and a smoke `ncu --set full` on any kernel. On GeForce this
      needs driver perf-counter permission (`NVreg_RestrictProfilingToAdminUsers=0` or admin). This has
      been the LLM GEMV redesign blocker (LLM_DECODE_PERF_GRIND "needs ncu, blocked by GeForce perms") —
      on a cloud box it works out of the box.
      *(2026-07-22: UNBLOCKED on this box. ncu 2026.2.1 user-local (deb-extract, no system install);
      driver has `RmProfilingAdminOnly:1` so counters need root → scoped NOPASSWD sudoers for the ncu
      binary (`sudo -n ncu ...`), plus `/etc/modprobe.d/nvidia-profiling.conf` flips the restriction off
      at the next reboot. Smoke capture on `stepcache_rel_l1_f32` (3060, 4096×1536 F32): **93.0% of peak
      DRAM bandwidth**, 157 µs, tensor-pipe 0% — the gate kernel is already AT the bandwidth roofline;
      no kernel work warranted. Profiled app runs as root: `chown` report files back after capture.
      Recipe: `sudo -n $NCU --kernel-name regex:<pat> --set <set> -o <rep> --target-processes all
      /usr/bin/env LD_LIBRARY_PATH=$HOME/.local/lib/cuda13 CUDA_VISIBLE_DEVICES=<n> dotnet test ...`.)*
      **Capture for the top-5 kernels per model**:
      `dram__throughput.avg.pct_of_peak_sustained_elapsed` (bandwidth roofline %),
      `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` (tensor-core %),
      `sm__throughput.avg.pct_of_peak_sustained_elapsed`. These three numbers per kernel ARE the
      %-of-ceiling table this whole grind optimizes against — put them in the results doc.

### H1 — Step cache (C1) bring-up + measurement

1. [x] Compile: `cd native/cuda/dit && ./build.sh` (compiles `stepcache.cu` → installs
       `src/HartsyInference.Cuda/Ptx/stepcache.ptx`; csproj globs it automatically). Rebuild.
       *(2026-07-22, 3060 box. **TRAP for anyone recompiling PTX here:** the pip-wheel toolchain must be
       version-locked — `nvidia-nvvm` (ships `cicc`, which stamps the PTX `.version`) floats to 13.3 even
       when `nvidia-cuda-nvcc==13.0.88` is pinned, and ISA 9.3 PTX fails driver JIT (`CUDA error 222:
       Unsupported .version 9.3; current version is '9.0'`) on this driver (580.159, CUDA 13.0). Install
       set that works: `pip3 install --target=~/.local/cuda-tools-13.0 nvidia-cuda-nvcc==13.0.88
       nvidia-nvvm==13.0.* nvidia-cuda-crt==13.0.* nvidia-cuda-runtime==13.0.* nvidia-cuda-cccl==13.0.*
       nvidia-cuda-nvrtc==13.0.*` then put `.../nvidia/cu13/bin` on PATH; verify every emitted PTX starts
       `.version 9.0` before shipping.)*
2. [x] Gate: run `StepCacheAccelerationTests` on the GPU box — CPU tests must stay green; then verify
       `CudaBackend.SupportsDeviceStepCacheGate == true` (log line appears when `HARTSY_STEP_CACHE` set).
       *(24/24 green on this box; gate=true asserted by the new GPU test below.)*
3. [x] Kernel numerics: A/B `RelativeL1Distance` CUDA vs the IBackend host default on identical random
       F32 + F16 tensors (~1e-6 F32 / ~1e-3 F16 agreement expected; atomic-order nondeterminism is fine
       at that tolerance). Add as a `GpuIntegration` test.
       *(`tests/HartsyInference.Cuda.Tests/StepCacheKernelTests.cs`, 4 tests, run on the 3060: F32 rel-err
       4.5e-7, F16 5.4e-7, identical-tensors ⇒ 0, gate=true. All pass.)*
4. [x] **Qwen-Image A/B** (the reference wiring; warm, 1024², 20 steps, cfg 4, seed 42, ×3):
       - baseline (unset) → confirm ≈ 39.4 s and byte-stable across the 3 runs;
       - `HARTSY_STEP_CACHE=0.1`, then `0.15`, then `0.2`;
       - record: wall, per-stream compute/reuse counts (logged), and **quality**: SSIM vs baseline image
         + eyeball. Acceptance: SSIM ≥ 0.95 at the shipped default; pick the default from the knee.
       - Watch VRAM: the cache holds prevIndicator + residual per stream (~58 MB × 4 at 1024²) — confirm
         peak stays under budget beside the resident DiT.
       *(📊 2026-07-22, 4090 — [results](../../benchmarks/results/2026-07-22_accel_stepcache_qwen_4090.md):
       baseline 40.17 s byte-stable ✓; 0.1 → 35.14 s (1.14×) SSIM 0.9552 ✓; 0.15 → 30.25 s (1.33×)
       SSIM 0.9189 eyeball-clean but below gate; 0.2 → 27.22 s (1.48×) SSIM 0.8744 visible
       simplification. **Shipped default moved 0.15 → 0.10** (the SSIM≥0.95 knee). Cached runs
       deterministic. VRAM ≈15.3 GB ✓. Harness: `StepCacheQwenAbTests`. Negative-result: 1.14× at gate
       is below the 1.4–2× literature headline — the plain accumulated-rel-L1 gate is the limiter;
       polynomial TeaCache gate (STEP_ACCELERATION §2.3) is the upgrade path before the video ports.)*
5. [ ] Replicate the wiring (same pattern: optional `stepCache` param after the first block, per-stream
       instances in the pipeline) to, in order: **Chroma** (biggest open image gap, 1.7×; note its
       persistent CFG-pair step graph must be BYPASSED when the cache is armed — variable per-step
       topology can't replay a fixed graph; eager fallback exists), **HiDream** (25 st, eager),
       **Wan T2V/I2V + LTX-2.3** (the big wins — video steps are 1.8–2.0 s each; TeaCache-class results
       are 2–4.4× on video; wire into `WanVideoBlock`-level forward the same way).
       *(📊 2026-07-22 — **Wan DONE, NEGATIVE for the pinned gate**: no plain-gate threshold passes
       SSIM≥0.95 at the verified 50-step config (0.03→0.88 … 0.2→0.72) — video trajectories diverge
       under ANY reuse; outputs stay eyeball-clean (different-but-coherent samples) at 1.2–1.55×.
       Honest ship-frame: "fast non-reproducible sampling" opt-in, default OFF; polynomial gate worth
       one try but per-seed SSIM may be the wrong acceptance metric for video (use FVD).
       [Full write-up](../../benchmarks/results/2026-07-22_accel_stepcache_wan_4090.md) — incl. two
       pre-existing standalone-Wan-test bugs found (FlowShift 5-vs-8; real-length embed slice vs the
       engine's 512-row + ZeroPaddedRows). Infra added: `IBackend.PinActivation` (cache state survives
       per-step FreeActivations). **LTX-0.9 WIRED + smoke-verified** (2026-07-22: `LtxVideoTransformer.
       Forward/ForwardPaired` optional caches force the eager path + skip `PrepareGraphLatent` — armed
       cache can't replay the CFG-pair graph; smoke @0.05/12st: armed, graph bypassed, clean run,
       0 reuses — LTX gate needs per-model threshold tuning like Wan; quality A/B requires the
       engine-anchored treatment first, same as Wan's ground-truth pass). Chroma/HiDream wiring
       DEFERRED — no weights on disk to verify, and Chroma's default-ON graph makes blind wiring a
       regression risk.)*
6. [ ] Negative-result discipline: if a model's gate never fires below quality-loss thresholds, record
       that in the results doc with the drift trace (the polynomial-rescaled TeaCache gate is the
       documented upgrade path — per-model coefficient fit, STEP_ACCELERATION §2.3).

### H2 — CFG interval (C2) measurement

1. [x] Qwen-Image warm A/B: baseline vs `HARTSY_CFG_INTERVAL=0.1,0.85` vs `0.15,0.9` (20 st, cfg 4).
       Expected: wall drops ∝ skipped uncond steps (logged); quality NEUTRAL-OR-BETTER (the paper's
       claim — verify SSIM + eyeball; if quality dips, shrink the excluded tails).
       *(📊 2026-07-22, 4090 — **NEGATIVE RESULT at the paper's bands.** Both bands: 33.8 s vs 40.1 s
       baseline (−16%, 6/20 uncond skips ✓ arithmetic) BUT SSIM 0.35/0.38 and the eyeball shows a
       CATEGORY flip: "A photograph of…" renders as a stylized flat illustration. Skipping guidance on
       the EARLY high-noise steps (normalized t > 0.85) abandons prompt-style adherence — the paper's
       ImageNet class-conditional FID result does NOT transfer to text-prompt fidelity on Qwen-Image
       @cfg 4/20 steps. Composability run (cache 0.1 + interval): 27.5 s, mechanically compounds ✓
       (uncond 10 computes/4 reuses — the self-healing works) but inherits the same style flip.
       Per-protocol remedy tested: late-only bands (`0.05,1` / `0.1,1` / `0.15,1` — guidance kept early,
       uncond skipped only at low noise): quality-safe (SSIM 0.981–0.996, eyeball-identical) but
       MARGINAL (−3…−5%) — this scheduler only puts 1–2 of 20 steps below t=0.15. C2 stays default-off;
       `0.15,1` is the sane opt-in. Full write-up:
       [2026-07-22_accel_cfginterval_qwen_4090.md](../../benchmarks/results/2026-07-22_accel_cfginterval_qwen_4090.md).)*
2. [x] Composability run: interval + step cache together (they compound: interval halves gated steps,
       cache skips block stacks on the rest). *(Done above — compounds mechanically; quality verdict
       tracks whichever interval band is used.)*
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

> **Status 2026-07-22 (v0 landed: correct, not yet fast).** Built `native/cuda/attention/sage_attn_int8.cu`
> (4 kernels: K channel-mean, Q/K per-row INT8 quant with folded attn scale + K-smoothing, fused flash loop
> with wmma s8 m16n16k16 QK^T + TF32 PV), optional-module plumbing (`HasSageAttentionKernels`,
> `HARTSY_SAGE_ATTN=1` dispatch in the F32/no-mask block, shape-gated D∈{64,128} ∧ Sq%32==0), and
> `SageAttnKernelTests` — **gate (1) PASSED on the 3060**: maxErr 3.3e-4 vs CPU F32 with channel-consistent
> K outliers (30× under the 1e-2 budget), clean-input control passes, Skv-tail correct, bit-exact
> run-to-run. **Gate (2) FAILED honestly**: `SageSdpaMicroBench` (3060) — v0 is **0.17×** vs the default
> materialized-TF32-cuBLAS F32 path (250 ms vs 43 ms @ [1,24,4096,128]; 2.31 s vs 0.41 s @ [1,24,12288,128];
> Welch |t|>47). Root cause: v0 cloned `flash_attn_v2_tf32.cu`'s structure, and wmma's UNDEFINED fragment
> layouts force O/softmax through SMEM with serial per-thread D-loops, BC=16 micro-tiles (Skv/16 sync-heavy
> iterations), Q8 refetched from global per step, 2-warp blocks — this is also why HARTSY_SDPA_V2 never
> became a default. **v1 plan (the real build)**: raw `mma.sync` PTX asm (m16n8k32 s8s8s32 QK^T, m16n8k8
> tf32 or m16n8k16 f16 PV) whose lane↔element layout is ARCHITECTURALLY DEFINED → register-resident O
> accumulator + per-lane m/l online-softmax state with shuffle row-reductions, BR=64/BC=64, `cp.async.cg`
> 2-stage K8/V staging, SMEM XOR swizzle. Only that shape of kernel can meet the ~2× target vs cuDNN.
> The v0 parity tests remain the correctness oracle for v1 (identical contract).
>
> **v1 layout worksheet (worked out 2026-07-22, use as-is):** 4 warps, warp owns 16 query rows; BC=64 ⇒
> QK^T = 8× `mma.sync.m16n8k32.s32.s8.s8.s32` per D/32-chunk (A=Q8 row-major, B=K8 col-major). C-frag
> (m16n8, s32): lane holds rows {g, g+8} (g=lane/4) × cols {t·2, t·2+1} (t=lane%4) — each row spread over
> 4 lanes ⇒ row max/sum = 2 shuffle-xor hops (offsets 1,2) within the quad. KEY ALIGNMENT: that C layout
> is register-identical to the m16n8k16-f16 A-frag layout, so S→P after softmax is pure reg renaming +
> `cvt.rn.f16x2.f32` packing — zero shuffles, zero SMEM round-trip. PV: P(16×BC f16, A) × V(BC×D f16, B
> via ld.shared, k-chunks of 16) → O = D/8 = 16 accum tiles × 4 f32 = 64 regs/lane; m/l = 2+2 regs
> (2 rows/lane); online rescale multiplies the 64 O regs by corr[row-half] — per-lane, no comms.
> Budget ≈ 110–130 regs/thread ⇒ 128-thread blocks still 2+ blocks/SM on GA102. V must be staged as F16
> in SMEM (cast on cp.async ingest or pre-pass) for the f16 PV mma; K8 B-frags via manual ld.shared.b32
> (ldmatrix is b16-only). Sq%64 gate replaces Sq%32 at BR=64 (or keep a BR=32 variant for the tail).
>
> **v1 BUILT + measured (2026-07-22, `sage_attn_int8_v1.cu`, entries `sage_attn_int8_v1_d{128,64}_f32`,
> preferred by `LaunchSageAttnInt8` when present; `HARTSY_SAGE_V0=1` forces the wmma v0):** fully
> row-guarded (ANY Sq — v0's Sq%32 gate excluded Wan's 3510-token stream), parity green (maxErr 3.2e-4,
> bit-deterministic). Iteration log (3060, `SageSdpaMicroBench` [1,24,12288,128] vs default cuBLAS path):
> v1 BR=64/BC=64 naive → **1.26×** (t=7.7); +XOR swizzle alone → 0.92× (conflicts −80% but regs 168→221
> lost a resident block — occupancy 24→16%); +`launch_bounds(128,3)` → 1.24× (cap-induced spills ate the
> win); **+BC=32 → 1.36×** (t=9.9; 168 regs, 0 spills, 3 blocks/SM). `asm volatile`→`asm` on the mma
> wrappers: no change (ptxas already scheduled maximally). Current ncu: latency-bound — 69.3% of stalls
> are short-scoreboard on SMEM fragment loads, SM 22%, mem 18%, IPC 0.81, occupancy 24.25% (theoretical
> 25 at 12 warps/SM, reg-capped). Note the honest frame: the paper's 2.1× was vs FA2;
> our baseline is the materialized cuBLAS-TF32 pipeline, a different (strong) incumbent, and the F16-in
> cuDNN fused path (Hunyuan/Kandinsky class) is not yet comparable — Sage takes F32 in/out today.
>
> **v1.3 LANDED (2026-07-22): 5.55× at [1,24,12288,128] (74.0 ms vs 411 ms, t=31.5), 4.57× at
> [1,24,4096,128] — parity green incl. Skv tail, 96 regs, 0 spills.** The two changes: (1) `sage_v_f16t`
> one-shot V transpose+cast to [B,H,D,skvPad] f16 (the per-block SMEM re-transpose was Sq/64-redundant —
> the dominant win); (2) cp.async.cg 16B double-buffered K8/Vt staging (2 stages ≈ 24.8 KB < 48 KB
> default). Trap fixed: cp.async src must be 16B-aligned — per-head kscale bases (head×Skv floats) are
> NOT for odd Skv → CUDA 716 MISALIGNED_ADDRESS; ks tile now stages with scalar loads. ~60% of the
> mixed-precision roofline (PV f16→f32 dominates the remaining floor). Full log:
> [2026-07-22_accel_sageattn_3060.md](../../benchmarks/results/2026-07-22_accel_sageattn_3060.md).
> Remaining H4 gates: (2-formal) BDN `SdpaGpuBenchmarks` run + 4090 numbers; (3) e2e Wan step + SSIM.
>
> **NEXT KERNEL UNIT — F16-ingest/F16-out Sage (designed 2026-07-22, build next session):** the image
> fleet survey shows nearly every DiT calls SDPA no-mask+allowF16 and the deciding variable is tensor
> dtype at call time. F32 callers: Sage already wins 1.15–1.25× ≥2048 (crossover bench vs the cast-
> paying cuDNN branch). F16-NATIVE callers (Qwen-Image, Hunyuan/Kandinsky class) hit branch 1
> (cast-free cuDNN) where Sage-from-F32 can't compete below ~16k. Build: (a) prologue variants reading
> `__half` — template `sage_quant_row` on SrcT, `sage_k_mean_f16`, `sage_v_f16t` from-f16 (pure
> transpose/pad, no cast); (b) `OUT16` epilogue knob → `sage_attn_int8_v1_*_f16io` entries (pair only
> with f16acc); (c) dispatch in the NATIVE-F16 branch behind the env knob with an initial Skv≥12288
> gate, then re-measure the crossover (cheaper prologue should pull it below 16k; kernel-floor work —
> `ldmatrix` fragment loads — is what turns 4k-seq parity into wins). Ideogram4 D=256 support is a
> separate instantiation (32 o-tiles blows the register budget — needs BC=16 or split-D two-pass).
>
> **D=256 SCOPED OUT (2026-07-22 decision, recorded):** 32 output tiles ⇒ 128 f32 accumulator regs
> (unbuildable without heavy spills) or a two-block D-split that RECOMPUTES the full-D QK^T (+50%
> total flops at D=256 where QK≈PV). Against cuDNN's NATIVE D=256 fused support (Ideogram4's current
> path, already perf-passed at 19.5 s), a +50%-flops handicap has no realistic win. Ideogram4's Sage
> story is the w13 fusion (shipped, awaiting weights) — not attention. Revisit only if a D=256 model
> shows up whose incumbent is the materialized path. (F16-ingest e2e note: `ldmatrix` PV variant was
> measured SLOWER and reverted — see results doc — so "ldmatrix flips 4k parity" above is DEAD; the
> 4k-seq flip now depends on prologue-cost reduction and deeper pipelining.)

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

- [x] **LLM GEMV redesign** with `ncu` finally in hand: MEASURED 2026-07-22 (4090, real Qwen3 Q4_K
      decode) — the standing "~22% everywhere" estimate is WRONG. `mul_mat_vec_q4k_f32` DRAM%: 1216
      blocks → **77.7%** (already llama.cpp-class), 512 → 70.0%, 320 → 57.4%, **128 (K/V projections)
      → 27.5%**. The deficit is small-N wave-filling, NOT access pattern — which the LLM-side QKV/gate-up
      fusion (in progress, uses this branch's FuseSwiGluPairs/ConcatRowsHost utils) captures by moving
      those bytes into the large-shape class. **R4 repack DEPRIORITIZED: its remaining ROI after fusion
      is the 512-block mid-shapes' ~10-20% — revisit only if post-fusion decode profiles still show a
      GEMV gap.** (Measure-first strikes again: this reprioritization cancels a large planned build.)
      **Independent confirmation (LLM agent, 2026-07-22, 3060, post-QKV/gate-up-fusion shapes)**: converges
      with the above from a different angle. Fresh `ncu --set full` at Qwen3-4B's real `ffn_gate` shape
      (K=2560 N=9728) shows the kernel compute/memory co-limited (68.65% DRAM, 74.6% ALU) rather than
      bandwidth-bound — a perfect coalescing fix could only reclaim a fraction of the flagged 41.5%
      isolated-instruction "Est. Speedup". SASS-level source correlation (`ncu --page source --print-source
      sass`) localized the flagged excess-sector load to the activation-vector reads (`float4 xa`/`xb2`), not
      the scale-byte unpack — root cause is `WARPS_PER_BLOCK=8` warps per block redundantly re-reading the
      same input row (L1 hit rate 94.83% already absorbs most of it). Built and measured the direct fix
      (per-block shared-memory staging of the input row, gated on a 96 KB opt-in budget, verified bit-exact
      against all 7 `FusedGemvGroundTruthTests`): **net 11% regression** (63.05us → ~70.2-70.7us/call,
      reproduced 3×) — the `__syncthreads()` barrier serializing all 8 warps costs more than the redundant
      reads it removes. Reverted. Full writeup: `docs/Checklists/LLM_DECODE_PERF_GRIND.md`, 2026-07-22 R4
      status block. Net: two independent measurements now agree R4 isn't worth building at current shapes.
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
| 2026-07-22 | RTX 3060 | LLM GEMV R4 shared-mem-staging (input-vector-reuse half only) | Q4_K decode GEMV, Qwen3-4B ffn_gate shape K=2560 N=9728 | 63.05us/call | ~70.2-70.7us/call | **-11% (regression, reverted)** | 7/7 `FusedGemvGroundTruthTests` bit-exact before revert | `docs/Checklists/LLM_DECODE_PERF_GRIND.md` (2026-07-22 R4 block) |
