# W8A8 IMMA Lane — Handoff (2026-07-23)

Continuation doc for the next agent. Read `docs/CODE_STYLE.md` + `docs/Agents/AGENTS.md` first, then
this. Companion context: `docs/Checklists/INFERENCE_ACCEL_GRIND.md` (H5 W8A8 item + results ledger),
`docs/Research/QUANTIZATION_LOW_PRECISION_INFERENCE.md` (§5 IMMA, SmoothQuant, ViDiT-Q).

## HARD CONSTRAINTS — read before touching anything

1. **Git is READ-ONLY. No commit / merge / branch / tag / push / worktree / stash — EVER.** The user
   commits everything manually. Deliverable = working-tree edits + a precise changed-file list.
   This directive was breached on 07-23 and user trust is damaged; do not re-litigate it.
   (Memory: `no-worktrees-no-commits-directive`.)
2. Generations must run through SwarmUI, never standalone CLI/test gens (memory
   `gens-must-run-through-swarm`) — kernel/block micro-benches via `dotnet test` are fine.
3. GPUs are shared: 4090 (`CUDA_VISIBLE_DEVICES=0`) belongs to Swarm/the user; prefer the 3060
   (`CUDA_VISIBLE_DEVICES=1`) — it is also the W8A8 target class (Ampere SM 8.6, no fp8 MMA).
   Never restart Swarm mid-generation; check `nvidia-smi` before touching anything.
4. An LLM-decode agent works concurrently in the SAME main checkout — check `git status` before
   editing shared files (`CudaBackend.cs` especially) and never stash/revert their dirty files.
5. Flagship regression gate after ANY engine deploy: Krea2-Turbo < 6.5 s AND Z-Image-Turbo ≤ 3.2 s —
   **ENGINE-INTERNAL log times** (`Z-Image txt2img complete in NNNNms`), NOT API wall (wall runs
   1.5–2.5 s higher under load and false-fails the gate). Open the output PNGs — timings lie.

## State as of handoff (all code in local main, user handles commits/push)

Landed and test-gated, everything opt-in behind `HARTSY_W8A8=1` (`CudaBackend.EnableW8A8`):

| Stage | What | Numbers (RTX 3060) | Tests |
|---|---|---|---|
| 1 | `src/HartsyInference.Cuda/Int8GemmExecutor.cs` — cuBLASLt int8 TN, **plain layouts (NO COL32/COL4_4R2_8C needed on cuBLAS 13.1)**, int32 out | raw **3.2–3.7×** over the F16 GEMM at DiT shapes; 80–96 TOPS ≈ 78–94 % of INT8 peak (F16 arm at ITS peak too) | `W8A8ImmaGemmTests` (exact vs CPU int32 ref) |
| 2 | `native/cuda/dequant/w8a8.cu` → `w8a8.ptx` (per-row dynamic int8 act quant + int32→F16/F32 dequant+bias); optional module `CudaKernels.HasW8A8Kernels` | full chain **2.56–2.57×** vs F16+fused-bias; **relL2 5.5e-3** pre-smoothing | same file, chain tests |
| 3 | `CudaBackend.LinearImpl` dispatch: gates `M≥32 ∧ K%4 ∧ N%4`, F16/BF16/F32 weight, F16/F32 act/out; per-channel host weight quant (Parallel.For, once) cached POOL-resident as `[int8 N·K | pad256 | F32 wScale[N]]`; weight `Fp8ScaleFactor` (branch-damp alpha) folds into wScale; cache freed in `FreeAllDeviceMemory`/`FreePreloadedWeights`/`Dispose` | integrated Linear **2.46×** at 4608×12288×3072 warm; **relL2 3.0e-3** vs baseline Linear (both act dtypes); cache-hit call 9.25 ms | `W8A8LinearTests` |

Regression safety: 35 adjacent GEMV/fp8 ground-truth tests green; W8A8 off by default leaves the
Linear path byte-identical (gate short-circuits).

Bench invocations (Category-gated, excluded from sweeps):
```
CUDA_VISIBLE_DEVICES=1 dotnet test tests/HartsyInference.Cuda.Tests/HartsyInference.Cuda.Tests.csproj \
  -c Release --filter "FullyQualifiedName~W8A8" --logger "console;verbosity=detailed"
```
PTX rebuild (if `w8a8.cu` changes): pinned toolchain at `~/.local/cuda-tools-13.0/nvidia/cu13/bin`
(PATH-prepend), `nvcc -ptx -arch=sm_80`, verify output starts `.version 9.0` (nvvm 13.3 emits 9.3
which the 580.159 driver refuses to JIT — memory `cuda-toolchain-perfgrind-setup`), install to
`src/HartsyInference.Cuda/Ptx/`.

## Traps discovered this session (will bite you again if unknown)

- **cuBLASLt bias epilogue reads bias in the OUTPUT dtype.** F16-out GEMM + F32 bias buffer = silent
  NaN garbage, no error.
- **Never mix synchronous `cuMemAlloc` with the engine's deferred-free world.** A sync alloc can
  receive a VA whose earlier `cuMemFreeAsync` (transient weight cast) hasn't executed; the late free
  then destroys the new buffer and the eventual free double-frees (INVALID_VALUE). All caches go
  through the stream-ordered pool (`GpuTransferHelper.AllocateDevice`/`FreeDevice`).
- **No mid-forward `Tensor.DataPointer` reads on device-cached tensors** — the lazy-sync consume
  races the transfer caches and the outer finally double-frees `pWeight`. The W8A8 gate is hoisted
  ABOVE the uploads for exactly this reason, and the F16 weight upload is SKIPPED on the W8A8 path
  (int8 replaces it entirely — also faster and saves VRAM).
- H3.1 postscript (closed NEUTRAL, don't reopen): GEMM-count consolidation does nothing at ≥4k-token
  image shapes — the GPU is GEMM-saturated. Third convergent negative (Ideogram4 w13, LLM R4).
  `HARTSY_CHROMA_FUSED_QKV` exists opt-in as the measurement record.

## Session update (2026-07-23, continuation): stage 4a/4b measured, 4c BLOCKED on a real e2e crash

Real-model target used: **Kandinsky-5.0 T2V Lite (2B)**, already staged (no download, no disk-pruning
needed) at `Models/Stable-Diffusion/Kandinsky5/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers/`, verified
working per `docs/Checklists/MODEL_STATUS_VIDEO.md`. Two new test files, both `[Trait("Category",
"W8A8Bench")]` in `tests/HartsyInference.Video.Tests/`:

- **`Kandinsky5W8A8MeasurementTests.cs`** (block-level, stage 4a/4b): calls `Kandinsky5Transformer.
  ForwardVideo` directly at 5 probe points across the real 30-step flow-match schedule, REAL loaded
  weights + REAL pre-computed Qwen/CLIP text conditioning (not synthetic), W8A8 on vs off, relL2 of the
  velocity output. Includes a determinism-floor control (two identical W8A8=off forwards) — floor
  measured **0.000e+00** (cuDNN fused SDPA is bit-deterministic on this D=64 path), so the signal below
  is real quantization error, not noise. **Result: relL2 ranges 1.35e-2 (mid-schedule) to 2.52e-2
  (t=147, low-noise tail)** — U-shaped across the schedule (elevated at both t=1000 and t=147, not a
  monotonic drift), and **3-5× worse than the existing synthetic-activation benchmarks (relL2 3-5.5e-3
  in `W8A8ImmaGemmTests`)**, confirming item 1's prediction that real checkpoint activations are uglier
  than synthetic. Still inside the existing 3e-2 gate at every probed point, but nowhere near the
  "near-lossless" ViDiT-Q regime — real signal that SmoothQuant (item 1) is worth building, but see the
  blocker below before spending more velocity-relL2 cycles: velocity relL2 is a proxy for the real ship
  gate (SSIM on decoded frames), not the gate itself.

- **`Kandinsky5W8A8SsimAbTests.cs`** (e2e, stage 4c, the SSIM ≥ 0.95 gate every other fleet feature
  ships against): **RESOLVED (2026-07-23, same continuation session) — real result obtained, two
  separate findings:**
  1. **The e2e crash was NEVER a W8A8 bug.** Root cause: `CausalConv3d.Forward`'s batched fast path
     (`src/HartsyInference.Diffusion/Models/Vae/CausalConv3d.cs`, the `batch == 1 && !DisableBatchedPath`
     branch) has a genuine memory bug — `Tensor.Dispose()` throws `CUDA_ERROR_ILLEGAL_ADDRESS` inside it,
     reproducing on BOTH the 3060 and 4090, with W8A8 fully OFF, with 17+ GB free VRAM confirmed via a
     1 Hz background `nvidia-smi` sampler through the crash. **Setting the class's own test-only escape
     hatch `CausalConv3d.DisableBatchedPath = true` (falls back to the older, still-correct per-frame
     loop) makes the crash disappear completely** — all three arms (floorA/floorB/w8a8) then complete
     cleanly end-to-end. This is a real bug in a decoder shared by Wan/HunyuanVideo/LTX/Kandinsky5's 3D
     causal-conv VAE path (docstring: "Reusable by any 3D causal VAE"), independent of W8A8 — it just
     happened to be what this session's W8A8 e2e test tripped over first. **FIXED (2026-07-23, same
     session).** Installed `compute-sanitizer` (not previously on this box — pip wheel `nvidia-cuda-
     sanitizer-api==13.0.85` into `~/.local/cuda-tools-13.0/`; the wheel ships an incomplete file layout,
     needed manual symlinks for `bin/x86/lib{TreeLauncherTargetInjection,TreeLauncherPlaceholder,
     TreeLauncherTargetUpdatePreloadInjection}.so` and `bin/{TreeLauncherSubreaper,
     TreeLauncherTargetLdPreloadHelper}` sourced from the already-installed Nsight Compute bundle at
     `~/.local/cuda-tools/nsight-compute/.../target/linux-desktop-glibc_2_11_3-x64/`, plus the three
     LD_PRELOAD injection libs symlinked directly into `bin/` from `../lib/`; if this box's sanitizer
     install ever needs reinstalling, redo those symlinks or the tool silently no-ops with "Target
     application terminated before first instrumented API call"). `--tool memcheck` against the crashing
     `floorA` arm (batched path enabled, reduced footprint `K5V_STEPS=2 K5V_FRAMES=9` to keep runtime
     down) found the exact fault: **an out-of-bounds `__global__` write in the `wan_vae_build_padded`
     kernel** (`native/cuda/wan/wan_vae_conv3d.cu`), 4421 occurrences, writes landing ~7.5 KB past a
     184,320-byte allocation. Root cause traced to `CausalConv3d.Forward` (lines ~137-141): when
     `_spatialReplicatePad` is set (HunyuanVideo/Kandinsky5's shared VAE convs — Wan's zero-pad convs and
     LTX's `reflectPre` path were never affected), the code allocated the `padded` working tensor at the
     UNPADDED spatial size (`hp = h, wp = w`) but then told `BuildPaddedFrames` to pad internally
     (`padH=_padH, padW=_padW` passed to the kernel) — the kernel's `Hp = H + 2·padH` write range then
     exceeded the buffer's actual allocated size. **Fix**: `hp`/`wp` now compute the post-pad size for
     the `spatialPre` case too (`h + 2·_padH`), not just the pre-existing `reflectPre` case. Verified:
     compute-sanitizer rerun on the fixed code (full memcheck pass, no early exit) confirmed clean;
     normal (non-instrumented) runs of all three arms (floorA/floorB/w8a8) now pass reliably on both
     GPUs, and MUCH faster than the `DisableBatchedPath` workaround (~37-56s vs ~170-360s per arm) —
     the fast path's whole performance point is finally realized correctly.
  2. **Final, confirmed SSIM answer** (fix in place, batched path enabled — no workaround needed):
     `Ssim_Compare_Dumps` — determinism floor (floorA vs floorB, both W8A8=off) **SSIM = 1.0000** (bit-
     identical, perfect control), **W8A8 vs floorA SSIM = 0.9211 — still below the fleet's 0.95 gate**,
     essentially unchanged from the pre-fix workaround measurement (0.9235) as expected — the VAE crash
     and the W8A8 quality gap were always two independent issues; fixing the crash does not touch W8A8's
     own quantization error. Eyeballed frames 0/12/24 of all three arms (converted BMP→PNG, viewed
     directly): same coherent snow-leopard-on-ridge subject and composition in every arm, no color
     shift, no anatomical break, no category flip — but the W8A8 arm is visibly, consistently softer in
     fine fur texture and eye/whisker detail than baseline across all three sampled frames. This is a
     real, measurable (not catastrophic) quality regression that **confirms SmoothQuant (item 1 below)
     is NEEDED, not merely "worth building"** — it converges with the stage 4a/4b block-level finding
     (real-checkpoint relL2 3-5× worse than synthetic) into one consistent story: current per-row/per-
     channel W8A8 without channel smoothing does not clear this repo's real-model quality bar on
     Kandinsky5. `CausalConv3d.DisableBatchedPath` in `Kandinsky5W8A8SsimAbTests.cs`/`K5V_DISABLE_
     BATCHED_CONV` is no longer needed and was left at its default (false) in the test file.
  - Repro / re-run commands (both GPUs confirmed to reproduce; poll `nvidia-smi --query-compute-apps` in
    a loop first — this box also runs a concurrent LLM-decode agent):
    `CUDA_VISIBLE_DEVICES=<1 for 3060, 0 for 4090> K5V_ARM=<floorA|floorB|w8a8>
    K5V_DISABLE_BATCHED_CONV=1 dotnet test
    tests/HartsyInference.Video.Tests/HartsyInference.Video.Tests.csproj -c Release --filter
    "FullyQualifiedName~W8A8_E2E_SingleGen" --logger "console;verbosity=detailed"`, then `dotnet test
    tests/HartsyInference.Video.Tests/HartsyInference.Video.Tests.csproj -c Release --filter
    "FullyQualifiedName~Ssim_Compare_Dumps"`. Each single-gen run takes ~6 min with the workaround
    (~2× the ~3 min with the batched path, since that path exists for VAE-decode performance).

  <details><summary>Debugging history (how the crash was localized) — kept for anyone auditing the
  investigation or hunting the exact CausalConv3d line</summary>

  Design note load-bearing for whoever picks this up: run exactly ONE full generation per PROCESS (not
  per backend instance) — an earlier version ran floorA/floorB/candidate back-to-back on one
  `CudaBackend` and the exception on generation 2 poisoned interpretation of generation 3; a fresh
  process gives a fresh CUDA context so a crash in arm N can't taint arm N+1. Each single-gen run
  (`K5V_ARM=floorA|floorB|w8a8`) dumps raw RGB frames to `/tmp/k5w8a8_ab/<arm>/`; a separate CPU-only
  `Ssim_Compare_Dumps` test SSIM-compares them once all three exist.
  - **First pass (initial code): floorA/floorB (W8A8=off) both PASSED** on the 3060 (~167-170s), **w8a8
    (W8A8=on) CRASHED** — `CUDA_ERROR_ILLEGAL_ADDRESS` at `CudaMemory.FreeAsync` inside
    `CudaBackend.Dispose()` → `FreeW8A8Cache()`, every time. This looked like a genuine W8A8 bug (wrong
    — see below). A hypothesized fix (evict `_w8a8WeightCache` entries inside `FreeWeights` too, not
    just `FreeAllDeviceMemory`/`FreePreloadedWeights`/`Dispose`) produced a bit-identical crash after
    rebuild — reverted, since it changed nothing (not safe to leave unconfirmed changes in shared CUDA
    code). `CUDA_LAUNCH_BLOCKING=1` also didn't move the crash site — **CORRECTION: this conclusion was
    drawn BEFORE the exception-masking fix below and is now known WRONG.** The masked trace always
    reported `Dispose()` regardless of where the real fault was, so "didn't move" proved nothing; the
    de-masked trace (see below) shows the real site is inside `DecodeTiled` → `CausalConv3d.Forward`,
    i.e. DURING decode, not after it. Ruling out "an earlier faulting kernel" was therefore also wrong —
    re-test with launch-blocking AND de-masking together to actually pin the exact kernel (not yet done
    as of this line; see the RESOLVED summary above for the follow-up).
  - **The user asked to rerun on the 4090** (hypothesis: Kandinsky5 too large for the 3060's 12GB).
    Result: **floorA (W8A8=OFF, pure baseline) crashed too, identically**, on a 4090 with 23 GB free and
    zero OOM warnings in the log (full 50s runtime, all 30 steps completed). This refutes BOTH the "too
    big for 3060" theory AND the "genuine W8A8 bug" theory in one shot: `EnableW8A8=false` means
    `_w8a8WeightCache` is never populated, so `FreeW8A8Cache()` is a no-op — it cannot be the cause of a
    crash that also fires with the cache empty. **The crash lives in the Kandinsky5 T2V / HunyuanVideo-
    VAE-decode path itself (or its `CudaBackend`/`GpuTransferHelper` interaction), independent of W8A8;
    W8A8 was only ever shifting timing, not causing it.** Re-ran floorA again on the 3060 with identical
    code: crashed again there too. The two originally-clean floorA/floorB runs were not representative —
    this box also runs a concurrent LLM-decode agent that cycles its own GPU tests on the 3060 (and
    sometimes CVD=0, which off this box's default FASTEST_FIRST ordering lands on the 4090 too) with no
    external signal besides polling `nvidia-smi --query-compute-apps`; several retries were confounded
    by genuine contention-driven OOM (visible as `CUDA_ERROR_OUT_OF_MEMORY` with an `OOM after async
    sync+pool-trim retry` warning line and a FAST, sub-15s failure before step 1) — those are noise,
    discard them. **The 4090 crash (23 GB free, full 30-step + VAE-decode duration, zero OOM lines) is
    not noise.**
  - **Exception-masking discovered and fixed**: every crash so far reported its stack trace as
    `CudaMemory.FreeAsync` ← `CudaBackend.Dispose()` ← the test method, with NOTHING in between (no
    `GenerateFromEmbeddings`/`RunDenoise`/VAE frames) — and no `[arm] N frames in Xs` output line, no
    dump directory ever created. That combination means the REAL exception was thrown **inside**
    `GenerateFromEmbeddings` and never returned; the test's `using CudaBackend backend = ...` then ran
    `Dispose()` during stack unwind on the now-broken CUDA context, Dispose threw its OWN exception
    (the FreeAsync one), and .NET reports the LATER exception, silently discarding the original — the
    `Dispose`-site trace was a mask the whole time, not the real crash site. **Fixed in
    `Kandinsky5W8A8SsimAbTests.cs`**: `CudaBackend` is now constructed explicitly (not `using`) inside a
    try/catch that logs the real exception (plus `FreeMemoryBytes()` at throw time) before rethrowing,
    with `Dispose()` moved to a `finally` that itself catches and separately logs any masking exception.
    One supporting clue already in hand from BEFORE this fix: a `floorA` retry on the 3060 (this same
    de-masking change not yet applied, but the OOM case happens to carry its own full trace since OOM is
    thrown at the allocation site, not deferred) surfaced a RICH stack trace through
    `HunyuanVideoVaeDecoder.Decode` → `Upsample` → `CausalConv3d.Forward` → `Conv2D` →
    `GpuTransferHelper.AllocateDevice` → `CudaMemory.AllocateAsync` — i.e. VAE-decode-tail memory
    pressure is already a proven real failure mode on this exact pipeline; the illegal-address crashes
    may well be the same VAE-decode-memory story with the error type flipping under different
    timing/contention (OOM when caught cleanly at the alloc site, illegal-address when a fault happens
    mid-kernel and only surfaces later). **Not yet re-run with the de-masking fix** — that is the
    immediate next step for whoever continues (command below), and it should finally name the real call
    site.
  - **Next steps, in order** (none require git — Dispose fix is already in place, working-tree edit):
    1. Re-run with de-masking on a verified-idle GPU (poll `nvidia-smi --query-compute-apps` in a loop
       first, same pattern used this session): `CUDA_VISIBLE_DEVICES=1 K5V_ARM=floorA dotnet test
       tests/HartsyInference.Video.Tests/HartsyInference.Video.Tests.csproj -c Release --filter
       "FullyQualifiedName~W8A8_E2E_SingleGen" --logger "console;verbosity=detailed"`. Read the "REAL
       EXCEPTION (pre-Dispose)" line — if none appears and the test PASSES, the crash was contention
       after all and the fix's diagnostic value was moot (good outcome, means 4c/4d are actually
       unblocked); if a real exception appears, its stack trace is authoritative.
    2. If VAE-decode memory is confirmed as the culprit, try `K5V_FRAMES=9` (small, valid: `(9-1)%4==0`)
       to shrink the 3D-decode memory peak — if a tiny-footprint run on a verified-idle GPU STILL
       crashes, that is a real bug independent of memory pressure and a clean minimal repro worth
       `compute-sanitizer` (not installed on this box — would need `pip3 install
       nvidia-cuda-sanitizer-api`, surface to the user first given the ~96%-full disk).
    3. Cross-check against the pre-existing, documented-passing `Kandinsky5_Gpu_T2V_ShortClip` (in
       `Kandinsky5VideoGenerationTests.cs`, the file this test's loading code was copied from) in the
       same environment — if IT also crashes now, the bug is environmental/pre-existing in the pipeline,
       not anything introduced by this session's test files; if it passes, diff against
       `Kandinsky5W8A8SsimAbTests.cs` for what differs (this test's `EnableW8A8`/`arm` plumbing, the
       explicit-Dispose change, or genuinely nothing — in which case suspect timing/contention again).
    4. Once a clean run produces frames for all three arms, `Ssim_Compare_Dumps` (already written) gives
       the real 4a/4c answer (SSIM ≥ 0.95 vs the floorA/floorB determinism floor) — that result, not the
       velocity-relL2 number above, should decide whether SmoothQuant (item 1) actually needs building.
    5. Bar for touching shared CUDA code once the real site is named: confirmed root cause + the fix
       verified by this same test going green, not a plausible theory (this repo also has a concurrent
       LLM-decode agent in the same checkout most of the time — check `git status` and `nvidia-smi
       --query-compute-apps` in a loop, not a single point-in-time check, before running anything on
       either GPU or editing `CudaBackend.cs`).

  All five steps above were completed this session (de-masking landed, real site named as
  `CausalConv3d`'s batched fast path via the `DisableBatchedPath` toggle, SSIM comparison run) — see the
  RESOLVED summary above this collapsed section for the final numbers.

  **Further localization attempt (still same session) — negative result, worth recording.** Re-ran
  `floorA` with de-masking AND `CUDA_LAUNCH_BLOCKING=1` together (the combination not tried before the
  masking fix landed) WITHOUT `K5V_DISABLE_BATCHED_CONV`, on a verified-idle 3060. This time it surfaced
  a clean `CUDA_ERROR_OUT_OF_MEMORY` (not illegal-address) with a full trace: `Conv2D` ->
  `GpuTransferHelper.AllocateDevice` -> `CudaMemory.AllocateAsync`, inside `CausalConv3d.Forward`'s
  batched path, called from `HunyuanVideoVaeDecoder.Upsample` -> `Decode` -> `DecodeTiled`.
  `FreeMemoryBytes at throw: 724 MB` — genuinely low on the 3060's 12 GB, unlike the 4090 run (23 GB
  free, no OOM). Two different symptoms (OOM on the 12 GB card, illegal-address on the 24 GB card) from
  the same code path suggested a memory-accounting problem (leak or over-allocation) in the batched path
  rather than a one-shot logic bug, so traced the ownership contract for every `GpuTransferHelper.
  CopyToDevice`/`FreeDevice` pair the batched path's helpers use (`CudaBackend.BuildPaddedFrames`,
  `FillBias`, `AccumulateTap`, and `Conv2D` itself): **`CopyToDevice` can return either a fresh
  caller-owned transient pointer OR an existing cache-hit pointer the caller must not free** (confirmed
  at `GpuTransferHelper.cs:214-238`, `WeightCache`/`ActivationCache` priority lookup) — this looked like
  the exact hazard class (a `finally` block unconditionally calling `FreeDevice` on a `CopyToDevice`
  result that might be cache-resident, e.g. `AccumulateTap`'s `pConv` from `CopyToDevice(convDt)`, where
  `convDt` was just written+cached by `Conv2D`). **Investigated and RULED OUT**: `GpuTransferHelper.
  FreeDevice` (line 350) is itself cache-aware — `if (gpuPtr != 0 && !s.CachedPointers.Contains(gpuPtr)
  && !IsArenaPtr(s, gpuPtr)) CudaMemory.FreeAsync(...)` — calling it on a still-cached pointer is a
  documented-safe no-op by design, and this pattern (unconditional `FreeDevice` on a `CopyToDevice`
  result inside a `finally`) is used identically and correctly throughout `Conv2D` itself (`pInput`/
  `pWeight`/`pBias`) and every other kernel wrapper in this class — so it is NOT the bug. **Net: the
  exact statement/kernel is still not pinned.** Candidates not yet individually verified: (a) something
  in `wan_vae_build_padded`/`wan_vae_fill_bias`/`wan_vae_accumulate_tap`
  (`native/cuda/wan/wan_vae_conv3d.cu`) — read closely this session, indexing math looks bounds-correct
  by construction given `tout = (paddedT - kt) / strideT + 1` (max `srcT` in `accumulate_tap` derives to
  <= `paddedT - 1` algebraically), so no OOB found by inspection, but not verified empirically; (b) a
  memory-growth pattern specifically across the MANY repeated `CausalConv3d.Forward` calls within one
  `DecodeTiled` invocation (multiple tiles x multiple resnet blocks x 2 convs/block) — not instrumented
  this session; a per-call `FreeMemoryBytes()` log across the decode loop would show whether usage climbs
  monotonically (leak signature) or jumps once (single bad allocation); (c) something specific to how
  `CudaKernels.LaunchWanVae*` marshals args differs from other kernel launches in this file — not
  compared line-by-line. **This is now the right place to stop without `compute-sanitizer`** — two
  plausible theories were investigated and both cleanly refuted, which is real progress (rules out
  the free-pattern and the kernel-index-math classes of bug), but continuing to guess a third theory
  without a tool that can just show the faulting address is the "blind patch" trap this session's
  discipline exists to avoid. The `DisableBatchedPath=true` workaround remains fully verified (SSIM
  comparison above ran on it) and is the practical unblock in the meantime.
  </details>

## What's left on this lane (stage 4+, in order)

1. **SmoothQuant — BUILT, kernel/plumbing VALIDATED, but does NOT clear the SSIM gate as calibrated.
   Next lever is per-group weight quant, not more SmoothQuant tuning.** (2026-07-24 session, full
   writeup: `docs/Checklists/W8A8_HANDOFF.md` session log below + memory
   `hartsy-inference-smoothquant-e2e-regression`.)

   **What was built** (all present, all opt-in, zero effect on production unless called):
   - `native/cuda/dequant/w8a8.cu` `w8a8_quant_rowwise_{f16,f32}` — new optional `invScale[K]` param
     (0/null = no-op, same convention as the dequant epilogue's `bias` arg). X_hat = X·invScale computed
     BEFORE the per-row absmax, so the row's own dequant scale reflects the smoothed magnitudes.
     Recompiled to `w8a8.ptx` — **PTX toolchain changed this session**: the box's nvcc is now CUDA 13.3
     (pip `nvidia-cuda-nvcc`), which emits PTX ISA 9.3 — too new for this box's driver (580.159.03,
     JIT ptxas caps at 9.0). Had to pip-install a pinned `nvidia-cuda-nvcc==13.0.88` +
     matching `nvidia-nvvm==13.0.88` (the nvvm codegen backend is the ISA-version-determining piece, NOT
     the nvcc frontend — installing nvcc alone still pulls latest nvvm as a transitive dep) into an
     isolated `pip install --target` dir to get ISA 9.0 output. Regression tests confirm the recompile is
     numerically neutral vs the prior CUDA-11.5-compiled `w8a8.ptx` (chain relL2 5.53e-3 → 5.52/5.53e-3;
     e2e no-smooth SSIM 0.9211 prior-session vs 0.9210 this-session, see control below) — but it IS a
     shipped-artifact change, flag it if `w8a8.ptx` provenance ever matters.
   - Correctness gate: `tests/HartsyInference.Cuda.Tests/W8A8ImmaGemmTests.W8A8QuantRowwise_InvScale_MatchesCpuReference`
     — exact match vs CPU reference, with and without invScale.
   - `src/HartsyInference.Cuda/CudaBackend.cs`: `QuantizeWeightForW8A8` folds a per-weight host `s[K]`
     (W_hat = W·s per input channel) into its existing per-output-channel host quant, if set. New public
     API `SetW8A8SmoothingScale(Tensor weight, ReadOnlySpan<float> s)` — stores host `s` + uploads a
     device `1/s` buffer, evicts any already-cached quantized weight so it re-quantizes smoothed on next
     use. `LinearImpl`'s W8A8 dispatch passes the stored invScale device pointer into
     `LaunchW8A8QuantRowwise`. **Calibration (deciding what `s` should be) is deliberately NOT in
     CudaBackend** — advisor-directed: storage + application only, don't build a permanent calibration
     API before SSIM proves the mechanism earns a place. Calibration lives in test/harness code
     (`Kandinsky5W8A8SsimAbTests.CalibrateSmoothQuant`).
   - Integration gate: `tests/HartsyInference.Cuda.Tests/W8A8LinearTests.Linear_W8A8_SmoothingScale_ReducesErrorVsExactReference`
     — synthetic outlier-channel activation through the FULL production `Linear()` path: relL2 vs exact
     F32 reference drops 2.409e-2 → 7.120e-3 (70%) with smoothing. **Mechanism is proven correct.**

   **Offline gate before building (advisor-directed measure-first, cheap and decisive — see
   `tests/HartsyInference.Video.Tests/Kandinsky5W8A8OperandAblationTests.cs`)**:
   - `W8A8_OperandAblation_ActivationVsWeight`: fake-quant A-only vs W-only on a real captured, genuinely
     deep (post-adaLN, mid-network) Kandinsky5 Linear (1792×1792) — activation dominates (A-only
     1.098e-2 vs W-only 5.413e-3, ratio 2.03), confirming SmoothQuant has headroom and won't structurally
     backfire. (Shallower captures — the t-invariant text-embed and visual patch-embed projections — gave
     misleading/degenerate ratios and were explicitly excluded once identified; see the test's comments
     on the M/K filtering.) **W-only floor ≈ 5.4e-3 for this layer — this is the part smoothing cannot
     touch, and it's the binding constraint below.**
   - `W8A8_ActivationChannelOutlier_Stability`: per-channel activation absmax Pearson r drops to 0.43
     between schedule extremes (t=1000 vs t=147) — outlier channel IDENTITY drifts across the schedule,
     not just magnitude. **Single-sample calibration is invalid**; must max-aggregate actMax across
     several timesteps spanning the schedule.
   - `W8A8_SmoothQuant_OfflineGate`: alpha sweep {0.3,0.5,0.7,0.8,0.9} on both captured layers — alpha≈0.7
     is the local-error optimum for both (attn: 1.72e-2→1.02e-2; FFN: 1.83e-2→1.08e-2), ~40% reduction.

   **What actually happened when built and measured e2e** (`Kandinsky5W8A8SsimAbTests`, real Kandinsky5
   T2V, 512×512×25f×30steps, 3060):
   - Calibration: 3 forward passes (t≈1000/833/147, mirrors the offline gate's schedule spread),
     always-re-arming capture hook sees all 335 W8A8-eligible Linears, max-aggregates per-channel
     actMax/wMax, computes `s_j=(actMax_j/wMax_j)^0.7` clamped [1e-3,1e3] per weight, calls
     `SetW8A8SmoothingScale` for all 335. ~126s one-time cost. **Caveat**: calibration necessarily runs
     with `EnableW8A8=true` (the capture hook piggybacks on the same eligibility gate as the real W8A8
     dispatch — no separate calibration-mode bypass exists), so captured activations reflect layers AFTER
     unsmoothed-int8 noise from earlier blocks in the SAME forward — a minor accepted approximation
     (doesn't change WHICH channels are outliers, adds jitter to magnitude).
   - **`W8A8_SmoothQuant_AllLayers_OfflineGate`** (same calibration hook, but computes per-layer fake-quant
     relL2 with/without smoothing instead of just applying it — zero extra GPU passes): of 335 layers,
     **281 helped >2%, 38 hurt >2%, 16 neutral; aggregate local relL2 dropped 29% (sum unsmoothed 4.995 →
     sum smoothed 3.561)**. Uniform alpha=0.7 is net-positive at the local level, not a "gate the hurt
     layers" situation.
   - **e2e SSIM regressed anyway: unsmoothed (this session's build) = 0.9210, smoothed = 0.9144.**
     Confirmed NOT a build-drift artifact — ran a same-build no-smooth control
     (`K5V_SKIP_CALIBRATION=1`, `Ssim_Compare_NoSmoothControl`) specifically to separate "smoothing hurt
     e2e" from "the w8a8.ptx recompile shifted the baseline" (advisor-caught confound, same class of
     mistake as the earlier `CUDA_LAUNCH_BLOCKING` false conclusion this session): 0.9210 ≈ the prior
     session's pre-recompile 0.9211, so the build is neutral and the 0.9210→0.9144 drop is a REAL
     smoothing effect.
   - **Conclusion**: local per-layer relL2 improvement (even net-positive in aggregate) does not survive
     propagation through 32 transformer blocks + VAE decode into e2e perceptual quality — smoothing
     redistributes error into channels/layers the downstream is more sensitive to than raw relL2 captures.
     Combined with the offline gate's W-only floor (≈5.4–6.6e-3 per layer, unsmoothed e2e was ALREADY
     0.9210 < 0.95 before SmoothQuant touched anything): **the weight-side quantization floor is the
     binding constraint, not activation outliers.** SmoothQuant attacks the activation half of the error
     and structurally cannot reach the weight half.

   **Disposition**: infra is complete, tested, and CORRECT — it's just not the lever that closes this
   gate with a uniform-alpha, all-layers policy. It is fully dormant in production (opt-in API, no
   caller in SwarmUI/the real pipeline sets `SetW8A8SmoothingScale`) — **do not revert it**, it may be
   useful for a smarter per-layer policy (e.g. gate to only the 281 layers the offline gate says helps,
   or a properly-clean F16-only calibration pass) if revisited.

   **Recommended next lever**: per-group (not per-output-channel) weight quantization — attacks the
   W-only floor directly, which is the thing actually keeping unsmoothed e2e under the gate. Not started
   this session; fresh multi-hour effort, needs its own measure-first pass before building.
2. **Timestep-aware calibration check (NDTC-style)**: per-row dynamic act quant already adapts per
   step, so full calibration may be unnecessary for W8A8 — MEASURE first: run a real DiT block/model
   with W8A8 on/off across the timestep schedule and track per-step relL2 drift before building any
   calibration harness (the measure-first lesson: H2, H3, R4 all cancelled builds).
3. **Block-level e2e on a real model (3060-fit)**: smallest wins-relevant target on disk/catalog —
   e.g. SD15/SDXL attention Linears or a small DiT; step-time A/B + seed-fixed SSIM ≥ 0.95 + eyeball.
   Note Chroma/Qwen-class checkpoints were disk-pruned; re-staging is a multi-GB download on a
   chronically ~98 %-full disk — prune after, surface first (memory `disk-cleanup-authorization`).
4. **Swarm e2e A/B** on a 3060-routed workload (audio models route to the 3060 via
   `HARTSY_AUDIO_CUDA_DEVICE`; image/video would need explicit device routing — discuss with user).
   Payoff model: 3060-class DiT GEMMs are 58–68 % of step time ⇒ ~1.5–1.9× e2e ceiling at chain 2.5×.
5. Optional perf polish, only if profiles demand: fuse the dequant into a custom epilogue, persistent
   act-quant scratch (the dp4a `EnsureDp4aScratch` pattern), skip-quant for repeated activations.

## Other open backlog (not this lane)

- Wan-14B 832×480 VAE-decode headroom estimator undershoots → OOM beside resident expert pair
  (serving backlog, `2026-07-22_accel_sageattn_3060.md`).
- HunyuanVideo I2V conditioning (recipe TODO; T2V is production-verified through Swarm as of 07-23,
  512×320 and 720p, memory `hunyuan-video-swarm-production-0723`).
- Sparse video attention: MEASURE per-layer attention-entropy concentration before designing
  (H5 item), sage v1 spill-free register reduction (deprioritized, ~60 % roofline).
- Wan2.2-Lightning / LTX-distilled loadable accelerators (H5 item).
- User pushes local main to origin manually; local main also carries the merged perf-grind history
  and a leftover `backup/performance-grind-2026-07-23` tag the user may delete.
