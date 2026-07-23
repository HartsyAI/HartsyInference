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

## What's left on this lane (stage 4+, in order)

1. **SmoothQuant α≈0.5 channel smoothing** (`docs/Research/QUANTIZATION_LOW_PRECISION_INFERENCE.md`
   §SmoothQuant): activations carry fixed-channel outliers; migrate difficulty into the weight via
   per-channel factor s_j at LOAD time (divide act channel, multiply weight row — needs a smoothing
   hook where the activation producer is known, i.e. recipe/converter level, or fold s into the
   preceding norm's affine like SmoothQuant does). Expect it to cut relL2 further and protect W8A8
   on real checkpoints whose activations are uglier than the bench's synthetic rows.
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
