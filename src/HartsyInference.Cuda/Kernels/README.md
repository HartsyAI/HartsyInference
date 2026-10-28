# CUDA kernel sources

The CUDA C++ (`.cu`) source for this package's GPU kernels, organized by domain
(`attention/`, `conv/`, `dequant/`, `dit/`, `lm/`, `vision/`, `wan/`, `audio/`). These compile to PTX,
and the **shipped, packaged artifacts live in [`../Ptx/`](../Ptx)** — the `.csproj` packs `Ptx/*.ptx` into
the NuGet and the runtime loads them from `BaseDirectory/Ptx` (never embedded resources).

## Two authoring styles — read this before editing a kernel

CUDA kernels here come in two forms, and **not every shipped `.ptx` has a `.cu` in this folder**:

1. **`.cu` source → compiled** (most kernels). Edit the `.cu` here, recompile, copy the `.ptx` to `../Ptx/`.
2. **Hand-written PTX** (~28 kernels, no `.cu` anywhere). The fundamental ops — `softmax`, `groupnorm`,
   `layernorm`, `geglu`, `cast_*`, `elementwise_*`, `transpose`, `broadcast_add`, `spatial`, and
   **`hgemm_mma_sm80`** — were authored directly as `.ptx` and live **only** in `../Ptx/`. To change one you
   edit the PTX by hand.

`hgemm_mma_sm80` (tensor-core `mma.sync` HGEMM) is **intentionally** hand-written and should stay that way —
hand PTX is the right tool for MMA instruction selection. The rest are historical; backfilling `.cu` source
for them is tracked as a maintainability item in [`docs/Checklists/ROADMAP.md`](../../../docs/Checklists/ROADMAP.md).
When you next touch one of those, write the `.cu`, verify the new PTX matches within tolerance, then delete
the hand-PTX.

## Compiling

Each domain folder has a `build.sh` holding only its kernel lists; the build itself is `build_common.sh`, which
every domain script sources. It compiles the `.cu` and copies the `.ptx` into `../Ptx/`:

```bash
src/HartsyInference.Cuda/Kernels/dequant/build.sh                # compile + install into ../Ptx
src/HartsyInference.Cuda/Kernels/dequant/build.sh --no-install   # compile only
HARTSY_NVCC=<CUDA 13.0.88>/bin/nvcc src/HartsyInference.Cuda/Kernels/lm/build.sh   # also the NVCC_BUILT kernels

# By hand, with the committed nvrtc helper (dlopens libnvrtc.so):
cc -O2 -o src/HartsyInference.Cuda/Kernels/nvrtc_compile src/HartsyInference.Cuda/Kernels/nvrtc_compile.c -ldl        # build the helper once
LD_LIBRARY_PATH=~/.local/lib/cuda13 src/HartsyInference.Cuda/Kernels/nvrtc_compile in.cu out.ptx compute_80 "$TINC"
cp out.ptx src/HartsyInference.Cuda/Ptx/
```

Target `sm_80` minimum (forward-JIT-compatible). Every `KERNELS` entry not in `SM80_ONLY` also ships a `<kernel>.sm75.ptx`
floor build, which a Turing card loads; `./build.sh --floor-only` builds only those floor variants and leaves the baselines
untouched. A kernel with arch-conditional device code is listed in its
domain's `ARCH_VARIANTS`; `./build.sh --arch sm_120a` (or a plain `--arch sm_89`) then also emits `<kernel>.sm120.ptx`
(`.sm89.ptx`). `CudaKernels.PtxPath` ranks each file by its `.target` header, not its name: a tuned variant above sm_75
loads only on that exact compute capability (family-specific PTX is not forward-compatible), and the baseline serves
every other card. A kernel in `PTXAS_SPILL_CHECK` fails the build when ptxas
reports a spill for any entry on sm_80, sm_86, sm_89 or sm_120 (a variant: its own arch). The backend logs `PTX variants: …` when it picks one. Verify emitted PTX ISA compatibility with the deployment driver (see the dated toolchain notes in TROUBLESHOOTING). The ISA ceiling is driver-dependent. Validate a new/changed kernel
against the CPU reference within tolerance before shipping (`docs/Agents/KERNEL.md`).

### The shipped PTX must reproduce from source — check it

Every `build.sh` compiles through `../nvrtc_compile`, so **every domain builds on a box with no CUDA toolkit**
(the `NVCC_BUILT` kernels below excepted). They did not always: `attention/`, `audio/` and `lm/` were nvcc-only, so on a
toolkit-less box they failed outright — and because nobody could rebuild them, their shipped PTX silently
drifted from their `.cu`. Eight kernels were found compiled from sources up to five weeks older than the
committed source, including `flash_attn_f32`/`_split` (source a week newer than the PTX) and both
`sage_attn_int8` variants. Editing a `.cu` and not shipping the `.ptx` is invisible at runtime — the old
kernel just keeps running — so treat a rebuild that changes `../Ptx/` as a bug report, not noise:

```bash
md5sum src/HartsyInference.Cuda/Ptx/*.ptx > /tmp/ptx.md5
for d in src/HartsyInference.Cuda/Kernels/*/; do (cd "$d" && ./build.sh); done
md5sum -c /tmp/ptx.md5 | grep -v ': OK'      # anything listed is stale or compiler drift
```

`CUDA_INC` must point at a **complete** header set — `mma.h` (attention) includes `crt/mma.h`, which the
`lib/cuda13/include` set lacks. The scripts auto-pick the first candidate that has it; override `CUDA_INC`
explicitly for a different toolkit.

**Each kernel has one compiler.** nvcc and nvrtc of the same release stamp the same `// Cuda compilation tools`
line, yet emit different PTX for many kernels (instruction selection, register numbering), and neither reproduces
the whole tree. Most kernels ship as nvrtc output. Those in a domain's `NVCC_BUILT` list — the LLM decode GEMVs,
norms, top-k and flash attention in `lm/`, the expert GEMVs in `moe/` — ship as CUDA 13.0.88 nvcc output, the bytes
their decode timings were taken on. `build.sh` therefore uses nvrtc for every other kernel even when nvcc is on PATH,
and builds `NVCC_BUILT` kernels only with nvcc (`HARTSY_NVCC`, else PATH, else `~/.local/cuda-tools-13.0`); unless
that nvcc is the 13.0.88 release at ISA 9.0 it reports them `UNVERIFIED` and leaves them as shipped. A pip-installed nvcc may lack CCCL (`<nv/target>`, which
`cuda_fp16.h` needs); the build then takes it from `$CUDA_INC`. Moving a kernel between lists changes its codegen,
so do it only alongside a perf run.
