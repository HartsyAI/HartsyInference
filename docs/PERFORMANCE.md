# Performance Guide

HartsyInference is engineered so that **every install reproduces the published benchmark times out of the
box**. The optimizations that produce those times are not tuning suggestions — they are the engine's
defaults, applied identically whether the engine runs inside the SwarmUI backend extension, a sample CLI,
or your own host application. This document specifies those defaults, the hardware and native-library
requirements behind them, how to verify the active configuration, and how to reproduce the published
numbers.

---

## 1. The standard performance profile

The profile is defined **inside the engine** (single source of truth; see
`HartsyInference.Core.Runtime.EnvSwitch`). Each feature is a tri-state environment switch:

| Value | Meaning |
|---|---|
| *(unset)* | The documented default below — what every fresh install runs |
| `1` / `true` | Force the feature on |
| `0` / `false` | Kill-switch — force the feature off (debugging, constrained hardware) |

Every feature degrades gracefully: a machine that cannot run a given fast path loses **speed only, never
correctness** — the engine falls back to the reference implementation and logs what happened.

| Feature | Switch | Default | Requires | Effect | Fallback when unavailable |
|---|---|---|---|---|---|
| cuDNN fused flash attention | `HARTSY_SDPA_CUDNN` | **On** | cuDNN ≥ 9.21 (see §3) | Attention (D ∈ {64, 128, 256}, unmasked or `[B,1,Sq,Skv]`-broadcast additive F32 mask) runs as one fused kernel — no materialized score matrix. ~34× per call at 4608-token self-attention; the single largest fleet win | Materialized cuBLAS QKᵀ→softmax→PV path; self-disables per session (missing library) or per head-dim (engine rejection) |
| cuDNN convolution forward | `HARTSY_CONV_CUDNN` | **On** | cuDNN ≥ 9.21 (see §3) | F16/BF16 NCHW convolutions run cuDNN tensor-core implicit-GEMM/Winograd engines instead of materializing an im2col matrix (a kernel-area-times-input-sized HBM round-trip per conv). The dominant conv win on UNet models (SDXL) and every VAE | im2col→cuBLAS GEMM; self-disables per session on any cuDNN failure |
| cuBLASLt bias-epilogue GEMM | `HARTSY_EPILOGUE_FUSION` | **On** | cuBLASLt | Biased Linear layers fold the bias add into the GEMM epilogue, removing a separate kernel + an output-sized HBM round-trip per Linear (~700/step on SDXL) | GemmEx + separate BiasAdd kernel |
| fp8 tensor-core GEMM | `HARTSY_FP8_NATIVE` | **On** when SM ≥ 8.9 (Ada/RTX 40xx+), **Off** otherwise | Ada-generation GPU | fp8-weight models run activation-quant (F16→e4m3) GEMMs on fp8 tensor cores; weights stay packed (no VRAM increase) | F16-cast GEMM |
| F16 DiT activations | `HARTSY_DIT_F16` | **On** | Per-architecture code opt-in | Audited DiT block loops run F16 activations (half the HBM traffic of the bandwidth-bound norm/modulate/gate/attention kernels). The switch alone never flips an un-audited model — F16 safety is verified per architecture (QK-normed attention, bounded FFN intermediates) before a model opts in | F32 activations |
| Resident DiT weights | `HARTSY_KEEP_MODELS` | **On** | — | DiT weights stay GPU-resident across generations (skips the per-generation free + ~2 s re-upload). VRAM-aware by construction: on a prompt-cache miss the pipeline evicts the DiT before loading the text encoder, so smaller cards remain viable | Free after each generation, re-upload on the next |
| Warm activation pool | `HARTSY_MEMPOOL_KEEP` | **On** | — | Freed activation buffers stay in the CUDA stream-ordered pool (release threshold raised), so per-op buffer reuse is instant instead of a driver round-trip (~13 s/gen on large-activation models). OOM-retry and explicit trim paths still return memory on demand | Threshold-0 pool (every free returns memory to the driver) |

**Compatibility note.** Versions before `1.0.0-alpha.45` shipped these features opt-in (`=1` required).
The SwarmUI extension pins the exact engine version, so extension users are always consistent; direct NuGet
consumers upgrading across that boundary get the profile enabled by upgrading alone.

### Overriding

```bash
# Disable one feature for a session (Linux)
HARTSY_SDPA_CUDNN=0 dotnet run ...

# Windows (PowerShell)
$env:HARTSY_FP8_NATIVE = "0"
```

There is nothing to enable — a fresh install is already running the full profile.

---

## 2. Hardware baseline

Published numbers are measured on an **NVIDIA RTX 4090 (24 GB, SM 8.9)**. Expectations by tier:

| Hardware | What changes |
|---|---|
| RTX 40xx / Ada or newer | Full profile, matches published times at equal VRAM |
| RTX 30xx / Ampere | fp8 tensor-core GEMM auto-disables (no fp8 units) → fp8-weight models pay an F16-cast per layer; everything else identical |
| Older / smaller VRAM | Same correctness; pipelines evict weights instead of keeping them resident when memory pressure demands it |

---

## 3. Native library requirements and resolution

The engine is pure C# — GPU work goes through the CUDA **driver** API plus three NVIDIA userspace
libraries loaded at runtime. None of them ship in the NuGet packages.

| Library | Linux soname | Windows DLL | Needed for | If missing |
|---|---|---|---|---|
| CUDA driver | `libcuda.so.1` | `nvcuda.dll` | Everything (comes with the GPU driver) | No CUDA backend |
| cuBLAS | `libcublas.so.13` (or `.12`/`.11`) | `cublas64_13/12/11.dll` | All GEMMs | No CUDA backend |
| cuBLASLt | `libcublasLt.so.13` (or `.12`/`.11`) | `cublasLt64_13/12/11.dll` | fp8 / epilogue-fusion GEMM paths | Those paths disabled |
| cuDNN ≥ 9.21 | `libcudnn.so.9` | `cudnn64_9.dll` | Fused flash attention | Materialized attention (slower, identical output quality) |

**Resolution order** (per library, newest version first):

1. `$HARTSY_CUDA_LIB_DIR` — explicit override, checked first
2. `~/.local/lib/cuda13` — the conventional user-local install location (Linux)
3. The system loader — `LD_LIBRARY_PATH`, `ldconfig` cache, `PATH` on Windows

NVIDIA's redistributable `.so`s carry `$ORIGIN` RUNPATH, so placing `libcublas.so.13`,
`libcublasLt.so.13`, and `libcudnn.so.9*` together in one directory is sufficient — siblings resolve each
other. No `LD_LIBRARY_PATH` export is required when using locations 1 or 2.

cuDNN must be **9.21 or newer**: the fused attention graph uses the unified `SOFTMAX` backend operation
introduced in that release. The engine was validated against cuDNN 9.24 (CUDA 13).

---

## 4. Verifying the active configuration

The engine self-documents at startup and on first use. Check the log for:

```
[Cuda] perf flags: SdpaCudnn=True NativeFp8Gemm=True MempoolKeep=True ...
```

— the resolved profile, printed once per backend construction. Every published benchmark states the flag
set it ran under; a bug report or benchmark without this line is not comparable.

```
[cuDNN SDPA] fused flash-attention engaged (D=128, cuDNN 92400)
```

— printed on the first fused attention call. If instead you see
`[cuDNN SDPA] disabled for the session (init failed)`, cuDNN was not found (§3) and attention is running
on the materialized fallback: output is identical, generation is slower.

---

## 5. Reproducing the published benchmarks

**Methodology.** End-to-end wall clock through the SwarmUI API (identical request routed to each backend),
1024×1024, RTX 4090. One cold generation (model load) then three warm generations; the **warm median** is
the headline number. Seeds are randomized per run (a fixed seed times the result cache, not the engine).
Nothing else may use the GPU during a run — contention poisons numbers. Every result must be
**visually verified coherent**; a fast broken kernel is not a result.

**Harness.**

```bash
cd benchmarks/swarm_image_bench
python3 bench_t2i.py --backend hartsy --config models.json --out results.json --only Krea2-Turbo
```

**Benchmark request parameters** (from `models.json`; all 1024×1024):

| Model | Steps | CFG |
|---|---:|---:|
| Krea2-Turbo | 8 | 1.0 |
| Z-Image-Turbo | 8 | 1.0 |
| Boogu-Turbo | 4 | 1.0 |
| Boogu-Base | 20 | 4.0 |
| Qwen-Image | 20 | 2.5 |
| Ideogram4 | 20 | 4.0 |
| Chroma1-HD | 20 | 4.0 |
| Flux-Dev | 20 | 1.0 (guidance 3.5) |
| Flux-Schnell | 4 | 1.0 |
| Flux.2 Klein 4B (distilled) | 4 | 1.0 |
| SDXL | 20 | 7.0 |
| ERNIE-Image | 20 | 4.0 |
| AuraFlow-0.3 | 20 | 3.5 |

**Current scoreboard** — RTX 4090, warm median, engine `1.0.0-alpha.45` + in-flight `44.x-local` optimization rounds, 2026-07-09. ComfyUI
column is the same request on the same GPU through the ComfyUI backend. The optimization grind is ongoing
and tracked in [`benchmarks/results/`](../benchmarks/results/); this table is a snapshot, updated as
rounds land:

| Model | HartsyInference | ComfyUI | Status |
|---|---:|---:|---|
| Z-Image-Turbo | **2.76 s** | 3.1 s | Faster than ComfyUI |
| Krea2-Turbo | **4.52 s** | 6.5 s | Faster than ComfyUI |
| Krea2-Base | 30.3 s | — | CFG path validated 07-10 (28 st/cfg 4.5); no Comfy baseline yet |
| Qwen-Image | **40.9 s** | 54.8 s | Faster than ComfyUI |
| ERNIE-Image | **20.0 s** | 23.9 s | Faster than ComfyUI (was 49.6 s / 2.1× slower) |
| Ideogram4 | 19.5 s | 17.0 s | 1.15× — optimization queued |
| Boogu-Turbo | 3.26 s | 2.54 s | 1.28× — was 48.9 s (15× in two rounds); optimization in progress |
| Boogu-Base | 26.5 s | 17.8 s | 1.49× — was ~6 min (~13×); optimization in progress |
| Chroma1-HD | 28.5 s | 16.6 s | 1.7× — was 3.7× (round 3: F16 blocks, persistent CFG-pair CUDA graph, context trim); batched CFG queued |
| AuraFlow-0.3 | **13.93 s** | 14.0 s | Tied with ComfyUI (was 31.4 s) |
| Flux-Dev | **16.05 s** | 12.5 s | 1.28× — Flux-family kit transplant done (`44.38-local`: F16 residual damp + persistent cross-generation step graph + rope/prompt caches); remaining gap is per-step GPU compute |
| Flux-Schnell | **3.6 s** | 3.04 s | 1.18× — Flux-family kit transplant done (`44.38-local`) |
| Flux.2 Klein 4B | **2.36 s** | 1.85 s | 1.28× — Flux-family kit transplant done (`44.38-local`). Distilled variant: 4 steps/CFG 1 official |
| SDXL | **2.93 s** | 3.7 s | Faster than ComfyUI (was 33.9 s / 9.2× slower two rounds ago) |

---

## 6. Experimental and diagnostic switches

Everything below is **default-off**, strict opt-in (`=1` only), and not part of the supported profile.
Semantics may change between versions.

| Switch | Purpose |
|---|---|
| `HARTSY_DIT_GRAPH` | CUDA-graph capture of the denoise step. Tri-state: architectures where the per-generation graph is a validated win run it **by default** (Chroma — the full CFG pair replays as one `cuGraphLaunch`, self-disables on capture failure); other opted-in models (Z-Image, Krea2) stay opt-in (`=1`). `=0` kills it everywhere |
| `HARTSY_SDPA_V2`, `HARTSY_SDPA_FORCE_FLASH`, `HARTSY_SDPA_FORCE_TILED` | Alternate attention kernels, validation only |
| `HARTSY_SDPA_F16` / `HARTSY_SDPA_NO_F16` | Force/kill the F16 SDPA path for **all** callers (per-call `allowF16` is the supported mechanism) |
| `HARTSY_TENSORCORE_GEMM`, `HARTSY_FP8_F16`, `HARTSY_FP8_F32`, `HARTSY_HIGH_PRECISION_GEMM`, `HARTSY_NO_TF32` | GEMM A/B-benchmarking toggles |
| `HARTSY_PROFILE`, `HARTSY_PROFILE_SYNC`, `HARTSY_PROFILE_OUT` | Op-level profiler (serializes ops — never profile and benchmark in the same run) |
| `HARTSY_VAE_STATS`, `HARTSY_LOWVRAM*`, model-specific `*_DEBUG`/`*_DUMP` switches | Targeted diagnostics |

---

## 7. Reporting performance issues

Include: GPU model and VRAM, the `[Cuda] perf flags:` log line, whether
`[cuDNN SDPA] ... engaged` appears, engine version, the exact request parameters, and cold vs warm
timings. A performance report is actionable only when the active profile is known.
