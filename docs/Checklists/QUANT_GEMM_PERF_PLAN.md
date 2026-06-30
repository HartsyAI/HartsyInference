# Quant + GEMM Performance Plan (measure, flip, fix)

> **Scope**: the GEMM / quantization hot path only (FP8, GGUF quants, tensor-core GEMM,
> attention). This is the pragmatic "cheap wins first, measure every step, never regress"
> thread. It sits **next to** the broad [`PHASE_B_GPU_PERFORMANCE.md`](PHASE_B_GPU_PERFORMANCE.md)
> (the full paper-grade microbench effort) and reuses its harness; it does not replace it.
>
> **Execute at will**: every stage below is self-contained. Run a stage, record numbers in the
> ledger, accept or revert based on the gate. Stop any time; the ledger is the durable record.

---

## Why this exists (findings from the kernel/quant audit)

Traced on 2026-06-30. The engine already has the right machinery, but the fast paths are
**switched off by default** and the on-the-fly quant path is wired for LLMs only.

| Capability | State today | File |
|---|---|---|
| FP8 scaled (diffusion) | native FP8 in VRAM, scale folded into cuBLAS `alpha` (free); default `Linear` **caches** the FP8→F16 cast (re-expands VRAM after warmup) | [CudaBackend.cs:287](../../src/HartsyInference.Cuda/CudaBackend.cs#L287) |
| Native FP8 GEMM (Ada/Hopper, `cublasLtMatmul`) | **built, default OFF** (`EnableNativeFp8Gemm=false`, "not validated on Ada in CI") | [CudaBackend.cs:57](../../src/HartsyInference.Cuda/CudaBackend.cs#L57) |
| Tensor-core HGEMM (`hgemm_mma_sm80.ptx`) | **built, default OFF** (`EnableTensorCoreGemm=false`, PTX fragment layout unverified) | [TensorCoreGemm.cs](../../src/HartsyInference.Cuda/TensorCoreGemm.cs) |
| cuBLASLt epilogue fusion (bias in GEMM) | **built, default OFF** (`EnableEpilogueFusion=false`) | [CudaBackend.cs:77](../../src/HartsyInference.Cuda/CudaBackend.cs#L77) |
| GGUF on-the-fly dequant GEMM (`QuantizedMatMul`) | wired for **LLM only**, gated by `LowVramQuant`; no diffusion/audio/video uses it | [GenericTransformer.cs:48](../../src/HartsyInference.LLM/Transformer/GenericTransformer.cs#L48) |
| GGUF for diffusion / T5 | dequant to F16/F32 **at load** (no inference VRAM saving) | [GgufModelLoader.cs:95](../../src/HartsyInference.ModelHandler/Gguf/GgufModelLoader.cs#L95) |
| Vulkan int8 GEMM (`matmul_int8`) | kernel exists, **not wired to any model's weights** | [VulkanBackend.cs:612](../../src/HartsyInference.Vulkan/VulkanBackend.cs#L612) |
| Attention (`lm_flash_attn_f32`) | correct but does a full block barrier+reduction **per key** (algorithmic loss) | [flash_attn_f32.cu:52](../../native/cuda/lm/flash_attn_f32.cu#L52) |

Reference reality check: ComfyUI-GGUF dequantizes per op then runs a normal matmul (identical to
`QuantizedMatMul`); ComfyUI `--fast` on Ada+ uses native FP8 GEMM (our `EnableNativeFp8Gemm`). So
the goal is to turn on what we already have, prove it, and close the diffusion-quant gap, not to
invent new quant math.

---

## Guardrails (apply to every stage)

1. **Correctness gate before speed.** A flag only stays on if its gate test passes on the target
   GPU. Gates that already exist:
   - `EnableTensorCoreGemm` → [`TensorCoreGemmTests`](../../tests/HartsyInference.Cuda.Tests/TensorCoreGemmTests.cs) (diffs vs cuBLAS through `Linear`)
   - `EnableEpilogueFusion` → [`EpilogueFusionTests`](../../tests/HartsyInference.Cuda.Tests/EpilogueFusionTests.cs)
   - `EnableNativeFp8Gemm` → `Fp8Executor.IsSupported` + a new diff test (Stage 4)
   Tolerance: F16 path avg_err < 1e-3 vs cuBLAS reference; F32 < 1e-5.
2. **No end-to-end regression.** Existing Generation / SSIM / parity tests must stay green with the
   flag on. The authority is [`PARITY_VERIFICATION.md`](PARITY_VERIFICATION.md).
3. **Measure on vs off, same process, same shapes.** Every claim is an A/B on identical inputs.
   Use Welch's t-test (already in [`analyze.py`](../../benchmarks/analyze.py)); accept a speedup only
   at α = 0.01.
4. **Record everything in the ledger** (bottom of this doc) and commit the result dir under
   `benchmarks/results/run_post_{tag}_{gpu}/`. A win we did not save did not happen.
5. **One change at a time.** Never flip two flags in the same measurement. Compounding comes after
   each is individually proven.
6. **Revert cleanly.** If a stage fails its gate or regresses, leave the flag default-off, write the
   negative result in the ledger (negative results are data), move on.

---

## Measurement protocol

Reuse the Phase B harness; do not build a new one.

- **Microbench (C#)**: `dotnet run -c Release --project benchmarks/HartsyInference.GpuBenchmarks -- --filter '<glob>'`
  - GEMM shapes: the ones the real models hit (SDXL/Flux/SD3.5/Z-Image/Qwen) already encoded in [`MatMulGpuBenchmarks.cs`](../../benchmarks/HartsyInference.GpuBenchmarks/MatMulGpuBenchmarks.cs).
  - Attention: [`SdpaGpuBenchmarks.cs`](../../benchmarks/HartsyInference.GpuBenchmarks/SdpaGpuBenchmarks.cs).
- **Python baseline (parity ceiling)**: [`benchmarks/python-baseline/`](../../benchmarks/python-baseline/) (`bench_pytorch_matmul.py`, `bench_pytorch_sdpa.py`).
- **End-to-end**: `bash benchmarks/run_benchmarks.sh` (it/s for diffusion, tok/s for LLM), plus VRAM
  high-water mark from the `nvidia-smi` poller for any quant/VRAM stage.
- **Per-op attribution**: `HARTSY_PROFILE=1` (NVTX dump) or `nsys` via [`benchmarks/profile.sh`](../../benchmarks/profile.sh)
  to get the dequant-vs-GEMM split and kernel-launch counts.
- **Metrics to log per run**: median time, p95, GFLOP/s (GEMM) or it/s & tok/s (e2e), VRAM peak,
  the gate accuracy number, the full flag config, GPU, driver/CUDA version, git SHA.

Devices, in priority order: RTX 3060 (SM 8.6, dev box, the primary reference) → an Ada card
(SM 8.9, L40S/4090, the only one that exercises native FP8) → A100/H100 if available.

---

## Stage 0 — Lock the baseline (do this first, always)

- [ ] Confirm clean tree, record git SHA + driver/CUDA/GPU fingerprint.
- [ ] Run full microbench + e2e with **all flags default (off)**.
- [ ] Run the Python baseline on the same box for the parity ceiling.
- [ ] Commit to `benchmarks/results/run_baseline_{date}_{gpu}/`. This is the immutable reference for
      every later "Nx" number. Fill the ledger Baseline rows.

Acceptance: harness runs clean end to end; baseline numbers committed.

---

## Stage 1 — Env-var wiring for the flags (cheap enabler, do before any flip)

Today the four flags are public properties set only inside tests. To A/B them in real pipelines and
in `run_benchmarks.sh` without recompiling, wire them to env vars (convention: `HARTSY_*`, matching
`HARTSY_PROFILE`). Default stays off when the var is unset.

- [x] In `CudaBackend` construction, read and apply (via `EnvFlag()`, [CudaBackend.cs](../../src/HartsyInference.Cuda/CudaBackend.cs)):
  - `HARTSY_EPILOGUE_FUSION=1` → `EnableEpilogueFusion`
  - `HARTSY_TENSORCORE_GEMM=1` → `EnableTensorCoreGemm`
  - `HARTSY_FP8_NATIVE=1` → `EnableNativeFp8Gemm`
  - `HARTSY_HIGH_PRECISION_GEMM=1` → `HighPrecisionGemm`
  - `HARTSY_LOWVRAM_QUANT=1` → default for `TransformerConfig.LowVramQuant` (env default on the `init` property)
- [x] Log the resolved flag set once at startup (`[Cuda] perf flags: …`).
- [x] `run_benchmarks.sh` result naming: `--tag` (→ `run_<tag>_<utc>_<gpu>`); slug honors the *visible* GPU
      under `CUDA_VISIBLE_DEVICES`; resolved flags + `LD_LIBRARY_PATH` recorded in `software.txt`.

Acceptance: **met** — setting a var flips the path (verified via the startup flag log); unset reproduces
Stage 0 (smoke fingerprint shows all `HARTSY_*=0`).

> **Stage-0 prerequisite: the benchmark harness was non-functional and produced silent `NA`.** Five bugs,
> all of which blocked *any* real microbench number, fixed in `run_benchmarks.sh` / the bench project
> (2026-06-30, RTX 3060):
> 1. **Missing cuBLAS path** — no system CUDA toolkit; the engine loads `libcublas`/`libcublasLt` by bare
>    soname from `LD_LIBRARY_PATH` (a torch venv's `nvidia/cublas/lib`). BDN runs each benchmark in a
>    **child process**, which without the path threw `CUDA is not available` → every result `NA`. Harness
>    now preflights cuBLAS and auto-detects/prepends a bundle (or fails loud).
> 2. **Multi-target run** — bench project is `net8.0;net10.0`; `dotnet run/build` needs `-f net10.0`.
> 3. **`--exporters json,markdown,csv`** was one comma-joined token (BDN rejects it) → space-separated.
> 4. **`find … | head` under `set -o pipefail`** — SIGPIPE 141 silently aborted the run mid-way.
> 5. **Stray git worktree** (`.claude/worktrees/…`) holds a duplicate bench `.csproj`; BDN's by-name
>    project search then aborts. Harness hides duplicates for the run and restores on exit (incl. the
>    success path — do **not** `trap - EXIT`).
> Net: identical inputs now yield real timings (3060: MatMul_F32 ≈ 5.2 ms, F16 ≈ 2.1 ms at one shape),
> 0 `NA`, worktree untouched.

---

## Stage 2 — Epilogue fusion (cheapest win, lowest risk)

Hypothesis: folding bias into the cuBLASLt epilogue removes one BiasAdd launch + one output-sized
HBM round-trip per Linear. Pure upside where a bias exists.

- [ ] Gate: `EpilogueFusionTests` green on the target GPU.
- [ ] A/B microbench: GEMM-with-bias shapes, `HARTSY_EPILOGUE_FUSION` off vs on.
- [ ] A/B e2e: a bias-heavy model (any DiT). Watch launch count drop in the NVTX timeline.
- [ ] Gate e2e correctness (SSIM / parity green).
- [x] If win confirmed (t-test α=0.01) and no regression → record; consider making default-on in a
      follow-up once it has run clean across devices. **Result: NOT a win** — see ledger. The flag swaps
      the GEMM backend (cublasGemmEx→cublasLtMatmul); deltas swing −41%…+42% by shape on the two GPUs,
      so it stays default-off. Gate (correctness) passes on both 3060 and 4090.
- [x] Ledger filled. (No `run_post_epilogue_*` committed — A/B was a focused `Linear_F16_Bias` probe,
      not a full-grid run; raw logs in the session scratchpad.)

---

## Stage 3 — Tensor-core HGEMM (F16)

Hypothesis: the hand-written `hgemm_mma_sm80` beats or matches the default `cublasGemmEx` F16 path on
aligned shapes. Risk: PTX fragment layout was authored from docs, never hardware-verified, so the
gate test is load-bearing.

- [ ] Gate: `TensorCoreGemmTests` green on an SM 8.0+ GPU (diffs vs cuBLAS).
- [ ] A/B microbench across the aligned GEMM shape grid, off vs on. **Also compare against cuBLAS**:
      if cuBLAS already wins on these shapes, the honest result is "keep cuBLAS, leave off" and that
      is fine to record.
- [ ] A/B e2e on an F16 model; gate SSIM/parity.
- [ ] Decision: default-on only where it measurably beats cuBLAS; otherwise document as available-but-off.
- [ ] Commit `run_post_tcgemm_{date}_{gpu}/`, fill ledger.

---

## Stage 4 — Native FP8 GEMM (Ada/Hopper)

Hypothesis: `cublasLtMatmul` FP8 tensor-core GEMM (our ComfyUI `--fast` equivalent) is faster than
cast-FP8→F16-then-HGEMM for the FP8 diffusion models, and saves the cached-F16-weight VRAM. Only runs
where `Fp8Executor.IsSupported` (SM 8.9+).

- [x] **New gate test** `Fp8NativeGemmTests` ([tests/…/Fp8NativeGemmTests.cs](../../tests/HartsyInference.Cuda.Tests/Fp8NativeGemmTests.cs)):
      diffs the `EnableNativeFp8Gemm` path vs the cast-then-F16 path through `Linear` on identical FP8
      operands. **PASS on 4090** (rel_err 7–8e-5, ≪ 1e-2 tol — first real-Ada validation of `Fp8GemmExecutor`,
      previously "untested locally"); **correctly SKIPS on 3060** (SM 8.6). Required the cublasLt resolver fix.
- [x] A/B microbench: `Linear_FP8_Native` ×10 shapes, off vs on, 4090. **Median 1.19× (best 1.96×),
      8/10 shapes faster.** Net win.
- [ ] A/B e2e: Flux Dev FP8 / a Chroma or Lumina2 FP8 checkpoint. Log **both** it/s and VRAM peak
      (this stage is where the cached-F16-cast VRAM cost should disappear). **PENDING — e2e harness not
      wired; needs a real FP8 checkpoint + the generation path. This is where the VRAM win is quantified.**
- [ ] Gate image quality (SSIM vs the F16-cast path). **PENDING (with e2e).**
- [ ] Commit `run_post_fp8native_{date}_{ada-gpu}/`, fill ledger. (Microbench rows filled; raw in scratchpad.)
- [x] 3060 (SM 8.6): unsupported, recorded "N/A, pre-Ada", gate SKIPS — confirmed.

**Decision pending (user):** default-on for Ada FP8 (SM 8.9+)? Net win + VRAM saving + matches ComfyUI
`--fast`; the two regressing shapes (s6/s8) argue for either accepting them or a per-shape guard. A
clean-box re-measure would firm up the marginal shapes before flipping the default.

---

## Stage 5 — Attention kernel (`lm_flash_attn_f32`)

The one algorithmic loss no flag fixes. Two steps, smallest first.

- [ ] **5a, low-risk quick win**: replace the bottom of the per-key tree reduction (`s < 32`) with
      `__shfl_down_sync`, removing ~5 of ~8 `__syncthreads()` per key. Gate: bit-close to current
      output (avg_err < 1e-5 F32). A/B `SdpaGpuBenchmarks` vs sequence length.
- [ ] **5b, real fix**: keys-in-parallel rewrite (warp owns a strip of keys, full Q·K dot in
      registers, single warp reduction for the online-softmax V accumulation). This aligns with the
      Phase B4.1 FA2 plan; coordinate so we do not write the kernel twice. Gate vs the materialize-S
      reference and vs `F.scaled_dot_product_attention` at model shapes.
- [ ] A/B e2e: LLM tok/s (decode + prefill), long-context especially.
- [ ] Commit `run_post_attn_{date}_{gpu}/`, fill ledger.

---

## Stage 6 — Diffusion quant VRAM path (policy + wiring)

Question to settle: **do we want low-VRAM quantized diffusion?** Today only LLMs get on-the-fly
dequant; diffusion quants are load-size-only. If yes, the backend method already exists.

- [ ] First **measure the cost** of the dequant-per-op design on the existing LLM path: at M=1
      (decode), time `QuantizedMatMul`'s dequant kernel vs the GEMM (`HARTSY_PROFILE=1`). If dequant
      dominates, that caps the win and informs whether a fused dequant-GEMM is worth it later.
- [ ] Decision point (record in the Decision Log below):
  - **(a)** Wire `QuantizedMatMul` into the diffusion `Linear` path behind a low-VRAM flag (mirror
    the GGUF-on-LLM pattern). Measure it/s cost vs VRAM saved on a Flux GGUF/FP8 checkpoint.
  - **(b)** Leave as-is; document that diffusion quants shrink disk/load only, not inference VRAM.
- [ ] If (a): gate parity, A/B it/s + VRAM peak, commit `run_post_diffquant_{date}_{gpu}/`.

Deferred (note, do not start unless measurement justifies): fused dequant-into-GEMM, and wiring the
Vulkan int8 GEMM to real weights. Neither ComfyUI nor diffusers does these for diffusion; only pursue
if Stage 6 measurement shows dequant overhead is the real ceiling.

---

## Results ledger

One row per measured A/B. `Δ%` is (new − baseline)/baseline; negative time = faster. Link the result
dir. Keep negative results.

| Date (UTC) | GPU | Stage | Config (flags) | Shape / Model | Metric | Baseline | New | Δ% | Gate | Result dir | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-30 | 3060 | 0 baseline | all off | 86-row microbench grid | mean ms | (ref) | — | n/a | n/a | `run_baseline_2026-06-30T080551Z_nvidia-geforce-rtx-3060` | MatMul_F16 2.2–15.7 ms; Sdpa_F16 2.2–25.9 ms (s1=long-seq). 0 NA. SHA 6bba440 + dirty. |
| 2026-06-30 | 4090 | 0 baseline | all off | 86-row microbench grid | mean ms | (ref) | — | n/a | n/a | `run_baseline_2026-06-30T080914Z_nvidia-geforce-rtx-4090` | Only **~1.3–1.6× faster than 3060** on GEMM/SDPA → likely **host/launch-bound**, not compute-bound, at these shapes. |
| | | 0 caveats | all off | — | — | — | — | — | — | — | Box contaminated (ComfyUI+rustdesk resident, accepted); 5 trials → some shapes ±20–36% (e.g. MatMul_F32 s1). Marginal wins on noisy shapes won't clear α=0.01. Python parity ceiling not run (no torch; use `--py-venv`→ComfyUI venv). |
| 2026-06-30 | 3060 | 2 epilogue | EPILOGUE on | Linear_F16_Bias ×10 | mean ms | off | mixed | −41%…+10% | **gate PASS** | `scratchpad ab_epi` | no consistent win |
| 2026-06-30 | 4090 | 2 epilogue | EPILOGUE on | Linear_F16_Bias ×10 | mean ms | off | mixed | −30%…**+42%** | **gate PASS** | `scratchpad ab_epi` | shapes 6/7/8 regress |
| | | 2 epilogue | — | — | finding | — | — | — | — | — | Flag swaps cublasGemmEx→**cublasLtMatmul** (different per-shape algo heuristics), not just bias-fusion. Pure bias saving < noise. **Keep default-off**; only worth per-shape selection, not a global flip. Eyeball deltas (8 iters, contaminated box) — a clean-box Welch t-test could refine per-shape, but sign-flips rule out default-on. |
| | | 3 tcgemm | TENSORCORE on | aligned GEMM | GFLOP/s | | | | | | vs cuBLAS too |
| 2026-06-30 | 4090 | 4 fp8native | FP8_NATIVE on | Linear_FP8_Native ×10 | mean ms | cast-F16 | native | **median −16% (1.19×), best −49% (1.96×)** | **gate PASS (rel_err 7e-5)** | `scratchpad ab_fp8` | 8/10 shapes faster; s6 (SD3.5) 0.72×, s8 (Lumina2) 0.98× regress. **Net win** — recommend on for Ada FP8. |
| 2026-06-30 | 4090 | 4 fp8native | FP8_NATIVE on | (structural) | VRAM | +2 B/param | +0 | weight stays FP8-only | n/a | — | Native path returns BEFORE the cached-F16-weight block → no resident F16 cast. Not visible in single-weight microbench; quantify on a full FP8 DiT e2e (it/s + peak VRAM). |
| 2026-06-30 | 3060 | 4 fp8native | — | — | — | — | — | N/A | n/a | — | SM 8.6 (pre-Ada): native path unsupported, gate SKIPS, dispatch falls back to cast-F16. |
| | | 5a attn | — | SDPA vs Skv | ms | | | | | | warp-shuffle |
| | | 5b attn | — | LLM decode | tok/s | | | | | | keys-parallel |
| | | 6 diffquant | LOWVRAM diff | Flux GGUF | it/s / VRAM | | | | | | policy (a) |

---

## Decision log

| Date | Decision | Rationale | By |
|---|---|---|---|
| 2026-06-30 | Plan created; cheap-switch-first ordering | fast paths already built but default-off; measure before building anything new | |
| 2026-06-30 | Stage 1 done; harness repaired (5 bugs) before Stage 0 | `run_benchmarks.sh` produced silent `NA` (BDN child had no cuBLAS path) — had to fix it to get any baseline | |
| 2026-06-30 | Both 3060 **and** 4090 are on the dev box | Stage 4 native FP8 (SM 8.9) is runnable locally; plan's "Ada may be unavailable" no longer applies | |
| 2026-06-30 | Stage 0 scope = C# microbench both GPUs | no torch on box (python parity ceiling deferred → point `--py-venv` at a ComfyUI venv); C# e2e not implemented in harness | |
| 2026-06-30 | Fixed engine bug: `CudaLibraryResolver` didn't resolve `cublasLt` | `[LibraryImport("cublasLt")]` needs the versioned soname (`libcublasLt.so.12`); no unversioned `.so` exists, so epilogue/FP8/Lt-GEMM threw `DllNotFoundException`. Dormant because all are default-off. Stage 2 gate was failing on this, not on math. | |
| 2026-06-30 | Added `Linear_F16_Bias` microbench | existing grid only calls raw `MatMul` (no bias) → epilogue fusion never engaged; needed a bias-GEMM probe for the Stage 2 A/B | |
| | Stage 6 (a) wire diffusion quant **/** (b) leave load-only | (fill after Stage 6 measurement) | |

---

## Quick reference: run order

```
# 0. baseline (flags off)
bash benchmarks/run_benchmarks.sh                       # commit run_baseline_*

# 1. wire env vars (code change), then per stage:
HARTSY_EPILOGUE_FUSION=1   dotnet test tests/HartsyInference.Cuda.Tests --filter EpilogueFusion
HARTSY_EPILOGUE_FUSION=1   bash benchmarks/run_benchmarks.sh           # commit run_post_epilogue_*
HARTSY_TENSORCORE_GEMM=1   dotnet test ... --filter TensorCoreGemm
HARTSY_TENSORCORE_GEMM=1   bash benchmarks/run_benchmarks.sh           # commit run_post_tcgemm_*
HARTSY_FP8_NATIVE=1        dotnet test ... --filter Fp8Native          # Ada box only
HARTSY_FP8_NATIVE=1        bash benchmarks/run_benchmarks.sh           # commit run_post_fp8native_*
# 5/6: kernel + wiring changes, A/B each, commit run_post_*
```
