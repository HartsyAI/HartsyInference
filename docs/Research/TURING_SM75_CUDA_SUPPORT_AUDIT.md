# Turing (sm_75) CUDA support: audit and recommended path

Scope: why the engine's CUDA backend refuses compute capability 7.5 cards (RTX 2060 SUPER here, 8 GB), what
actually blocks it, how much kernel work it takes, and what to build first. Audit only; no kernels changed.

## TL;DR

- **CUDA itself is not the blocker.** The 2060 runs CUDA 13 fine. The blocker is our build policy: every shipped
  kernel is compiled to `sm_80` PTX, and PTX JIT only moves forward, so an sm_75 device cannot load it.
  `BackendFactory` then refuses the whole CUDA backend and `auto` falls to Vulkan.
- **Most of the code already compiles for sm_75.** A per-file `nvcc -arch=sm_75 -cubin` audit of all 96 `.cu`
  kernels: **90 compile for sm_75, 6 do not**. All 96 compile for sm_80 (control), so the 6 are real sm_75
  gaps, not header or toolchain noise.
- **The 6 are the tensor-core / async-copy kernels**: flash-attention F16 (two files), INT8 sage/v2 attention,
  INT8 MMA GEMM, and MoE prefill. They use `cp.async` and `mma.sync.m16n8k16` / `m16n8k32`, which need sm_80.
- **No new CUDA kernels are needed to get Turing running.** Step 1 is a build-list change plus per-kernel sm_75
  variants. Only the 6 need real rewrites, and those rewrites use the Turing-legal `m16n8k8` MMA and synchronous
  loads in place of `cp.async`. They are mechanical and parity-testable.
- **Recommended:** make sm_75 a supported CUDA baseline (step 1), keep Vulkan as the fallback for Pascal and older,
  and measure CUDA-on-2060 against Vulkan-on-2060 before deciding more. Details and estimates below.

## 1. Root cause, traced in the code

| Where | What it does |
|---|---|
| `src/HartsyInference.Cuda/Kernels/build_common.sh` | Compiles every `KERNELS` list for `sm_80`. `KERNELS_SM75` exists only for the dequant and MoE expert domains. |
| `src/HartsyInference.Cuda/CudaKernels.cs:503` | `BaselineSm = CudaArch.Ampere` (sm_80). Modules whose `.target` is newer than the device are skipped, not bound. |
| `CudaKernels.cs:515` `PtxPath` | Picks the file whose `.target` header is the highest the device can run, among `{kernel}.ptx` and `{kernel}.sm*.ptx`. A tuned variant above sm_75 needs an **exact** SM match. |
| `src/HartsyInference.Engine/BackendFactory.cs:143` | `CudaMeetsKernelBaseline` returns false below sm_80, so automatic selection passes CUDA over and takes Vulkan. |
| `CudaBackend.cs:1445` | `SupportsBF16 = Sm >= Ampere`. |
| `CudaBackend.cs:1344` | TF32 and "fast F16 GEMM" are gated to Ampere. |

So the sm_80 number is a **packaging decision**, not a hardware requirement for most of the code. The hardware
requirement is confined to the instructions in §2.

Confirmed from the troubleshooting log: `CUDA error 218: PTX JIT failed … requires .target sm_80` is the expected
symptom if the gate is bypassed.

## 2. What sm_75 cannot do (hardware/ISA, checked with nvcc)

| Feature | sm_75 | Evidence |
|---|---|---|
| `cp.async` (async global to shared copy) | **No** (sm_80+) | per-file audit; ptxas error `Feature 'cp.async' requires .target sm_80` |
| `mma.sync.m16n8k16` (any dtype) | **No** (sm_80+) | ptxas error `Feature '.m16n8k16' requires .target sm_80` |
| `mma` with `.bf16` operands | **No** | ptxas error `Feature '.bf16' requires .target sm_80` |
| `mma.sync.m16n8k8` F16 | **Yes** | passed in the combined probe; also `docs/Research/DEEP_KERNEL_OPTIMIZATION.md:49` |
| `ldmatrix` | **Yes** | passed in the combined probe |
| BF16 storage and conversion | Yes, software-emulated | 90 files incl. `dit_bf16.cu` compile for sm_75; BF16 arithmetic is slow, no BF16 tensor cores |
| Shared memory per SM | 64 KB (opt-in above 48 KB per block) | `CudaDriverApi.cs:88` comment assumes sm_89 limits; the engine reads per-device limits |

Caveat on the probe: it was one file, so ptxas reported several errors at once. The per-file audit in §3 is the
stronger evidence. Not verified individually: `m8n8k16 s8` and `m16n8k32 s8` on sm_75. The INT8 kernels fail
anyway, and their failure cause is not itemized yet (see §3 table).

## 3. Kernel audit (96 files)

Method: for each `src/HartsyInference.Cuda/Kernels/**/*.cu`, `nvcc -arch=sm_75 -cubin -std=c++17 -I<dir>`, and the
same with `-arch=sm_80` as control. nvcc 11.5 (`/usr/bin/nvcc`) was used for feasibility. Its PTX ISA differs from the
shipped 13.0.88 bytes, so this is a compile-feasibility result, not a shipping result. Raw output:
`/tmp/claude-1000/-home-kalebbroo-Desktop-Projects-HartsyInference/4d755220-a2ce-49d0-96ee-c3b9c320e7e6/scratchpad/audit/results.txt`

| Result | Files | Count |
|---|---|---:|
| sm_75 OK, sm_80 OK | every other file | 90 |
| sm_75 FAIL, sm_80 OK | see below | 6 |
| sm_80 FAIL (control) | none | 0 |

The five sm_75 failures, with the sm_80 feature each one needs:

| File | Needs sm_80 for | Fix class |
|---|---|---|
| `attention/flash_attn_f16.cu` | `cp.async` (4), `m16n8k16` F16 (3) | Synchronous loads + `m16n8k8` MMA |
| `lm/flash_attn_causal_f16.cu` | `m16n8k16` F16 (2) | `m16n8k8` MMA; check smem staging |
| `attention/sage_attn_int8.cu` | `cp.async` (15), `m16n8k16` F16 PV (3) | Synchronous loads; INT8 `m16n8k32` must be checked (see below) |
| `attention/int8_attn_v2.cu` | `cp.async` (3), `m16n8k16` (3) | As above; this file is `NVCC_BUILT` |
| `dequant/int8_mma_gemm.cu` | `cp.async` (16), 128-register accumulators | Synchronous loads; the smem swizzle comments are sm_80-tuned |

`moe/moe_prefill.cu` is not a failure. It compiles and assembles for sm_75 (nvcc and nvrtc compute_75, then ptxas), so
it stays on the sm_80 build only until a Turing run verifies its grouped-prefill path.

Caveat: "90 compile" is not "90 run correctly". The check verifies that the code is legal for sm_75. It does not
check runtime behavior, the shared-memory opt-ins, `__shfl_sync` widths, or performance. Each of the 90 still needs a
parity run on the 2060.

Two files use BF16 but no BF16 MMA: `dit/dit_bf16.cu` and `dequant/dequant_recipe_to_bf16.cu`. They pass the sm_75
compile, so they can be shipped to Turing. The open question is numerics: BF16 is software-converted on sm_75, so
speed suffers, and the memory notes show F16 clips above 65504. Models whose activations need BF16 range need an F32
or BF16-emulated path on Turing. This is a correctness-and-speed question for parity, not a build blocker.

## 4. Fallbacks already in place

The attention dispatch is a capability-row planner (`CudaBackend.AttentionRows.cs`). A row is admitted only if its
module is loaded (`HasFa2Causal`, etc.), so a Turing device that cannot load the F16 flash module should fall through
to another row. Two things to verify before trusting that:

- The fallback rows that do not need sm_80 are not all efficient. `host.reference` (`AttentionRows.cs:429`) is a
  host-side F32 path, which would be very slow on GPU. A non-MMA device row (for example `flash_attn_f32.cu`, which has
  no MMA) has to be confirmed as an admitted row on Turing. Not yet verified at runtime.
- GEMM already goes through cuBLAS/cuBLASLt (`CudaBackend.cs` header comment). cuBLAS supports sm_75, so the matmul
  path is probably not the blocker. The `_gemmFast16`/TF32 gates just turn off on Turing. Needs a runtime check.

## 5. Performance: why Vulkan is not the answer on its own

From `benchmarks/scoreboards/VULKAN.md` (2026-07-28/29), CUDA ÷ Vulkan on the same shape grid:

- GEMM F16: Vulkan 30× to 157× slower (SDXL QKV 32×, Flux FFN 143×).
- SDPA: 36× to 182× slower at model shapes; 430× at video-DiT scale.
- Norm/elementwise: 14× to 220× slower.

Caveat: the table does not record which CUDA card produced its numbers, and the Vulkan numbers are not from the 2060.
Treat the ratios as the gap on a fast card, not on Turing. The document itself says the Vulkan attention kernel is
deliberately correctness-first (Br=1, no coopmat). So Vulkan is a working floor for old cards, not a performance
target.

Decision implication: for GPU-poor users, a CUDA sm_75 path (cuBLAS for GEMM plus the sm_75 kernels) is the
high-value target. Vulkan stays the fallback for Pascal and older, and for any card where the CUDA path is not yet
validated.

## 6. Recommended approach

### Step 1: make sm_75 a CUDA baseline (small, days)

1. Build the 90 compile-clean kernels for `sm_75` PTX as the baseline (`KERNELS` → `-arch=sm_75`). Forward JIT means
   sm_80, sm_86, sm_89, and Blackwell still load them.
2. For the 6 sm_80-only kernels, keep the sm_80 PTX as an exact variant (`{kernel}.sm80.ptx`) and add a
   Turing-legal `{kernel}.sm75.ptx` variant. The existing `PtxPath` already picks exact variants. It does not yet
   fall back to a baseline when no exact variant exists, so the baseline must be a file every device can load. Plan
   to confirm the exact-match behavior in `CudaKernels.cs:515` before relying on it.
3. Set `BaselineSm = CudaArch.Turing` and update `BackendFactory.CudaMeetsKernelBaseline`, the gate messages, and the
   `build_common.sh` comments. Keep sm_70 and older on Vulkan.
4. Run the existing CUDA test suite on the 2060 with the 6 sm_80-only kernels temporarily excluded, to confirm
   nothing else depends on sm_80.

Effort estimate (mine, rough): 2–4 days including the build changes and the test run.

### Step 2: port the 6 sm_80-only kernels (weeks)

Replace `cp.async` with register-staged `ld.global`/`st.shared` (or double-buffered synchronous loads). Replace
`m16n8k16` with two `m16n8k8` per K-step: the A fragment changes from 4 to 2 registers, so the fragment-load code
changes, not just the asm line. The INT8 kernels need a separate check: `m16n8k32 s8` may or may not be legal on sm_75,
and the audit did not isolate it. `moe_prefill` compiles for sm_75, so it is not part of that open question.

Each kernel needs a parity test against its sm_80 output on the same inputs, plus a timing check against the
current Vulkan path. Effort (mine, rough): 1–2 weeks per attention kernel including parity, 1 week for the INT8 GEMM
and the MoE file. Total on the order of 4–6 weeks of focused work.

### Step 3: measure before expanding

On the 2060, measure:

- CUDA (sm_75 baseline + the six ported kernels) against Vulkan, on the same shapes as `VULKAN.md`.
- Whether cuBLAS GEMM alone recovers most of the gap, given the GEMM ratios are the largest.

The result decides whether the remaining Vulkan work (ROADMAP Phase 5 coopmat) or more CUDA tuning is worth the time.

### What not to do

- Do not write new CUDA kernels to replace the 90 that already compile. The work is the build list and runtime
  validation, not rewriting.
- Do not expect Pascal or older to run CUDA. CUDA 13 (and I believe the CUDA 13 release notes) drop compile targets
  below sm_75. I did not verify that against NVIDIA's documentation during this audit, so check it before quoting it.
  Those cards stay on Vulkan.

## 7. Separate axes (do not conflate)

- **Compute capability** (this document): whether the kernels can run on the card at all.
- **VRAM** (8 GB on the 2060 SUPER): orthogonal. Whisper large-v3 ran out of memory on Vulkan on this card, and it
  will run out on CUDA too if the model does not fit. Fixing sm_75 does not fix that. Options are quantized Whisper,
  offloading, or CPU transcription, as the user already does.
- The PyTorch CUDA reference in the pasted note: the Python baseline is a separate question. Current CUDA PyTorch
  wheels should include sm_75. Check with `torch.cuda.get_arch_list()` before installing. It does not change the
  engine's kernel build.

## 8. Open questions to settle with GPU runs

1. Resolved: `moe_prefill.cu` compiles for sm_75. Open: whether its grouped-prefill path is correct on Turing (needs a run).
2. Do the INT8 `m16n8k32` and `m8n8k16` shapes compile for sm_75 in isolation? (single-instruction probe)
3. On the 2060, does the planner pick a usable non-MMA attention row, or `host.reference`?
4. Does `CudaKernels` gate on exact-variant match in a way that breaks a sm_75 baseline plus sm_80 variant layout?
5. What is the CUDA-vs-Vulkan ratio on the 2060 for GEMM, SDPA and decode? (this is the number the decision needs)
