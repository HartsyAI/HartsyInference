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
- [ ] If win confirmed (t-test α=0.01) and no regression → record; consider making default-on in a
      follow-up once it has run clean across devices.
- [ ] Commit `run_post_epilogue_{date}_{gpu}/`, fill ledger.

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

- [ ] **New gate test** `Fp8NativeGemmTests`: diff the `EnableNativeFp8Gemm` path against the
      cast-then-F16 path through `Linear` on real FP8 weights (avg_err at FP8 noise level, ~1e-2 rel).
- [ ] A/B microbench: FP8 GEMM shapes (Flux 3072-wide etc.), off vs on, on the Ada box.
- [ ] A/B e2e: Flux Dev FP8 / a Chroma or Lumina2 FP8 checkpoint. Log **both** it/s and VRAM peak
      (this stage is where the cached-F16-cast VRAM cost should disappear).
- [ ] Gate image quality (SSIM vs the F16-cast path).
- [ ] Commit `run_post_fp8native_{date}_{ada-gpu}/`, fill ledger.

Note: on the 3060 (SM 8.6) this path is unsupported; record "N/A, pre-Ada" and skip.

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
| | | 2 epilogue | EPILOGUE on | DiT Linear+bias | ms/op | | | | pass/fail | | |
| | | 3 tcgemm | TENSORCORE on | aligned GEMM | GFLOP/s | | | | | | vs cuBLAS too |
| | | 4 fp8native | FP8_NATIVE on | Flux FP8 | it/s | | | | | | + VRAM peak |
| | | 4 fp8native | FP8_NATIVE on | Flux FP8 | VRAM GB | | | | | | |
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
