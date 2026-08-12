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

Each domain folder has a `build.sh` that compiles its `.cu` and copies the `.ptx` into `../Ptx/`:

```bash
# With nvcc on PATH (the normal case):
src/HartsyInference.Cuda/Kernels/dequant/build.sh        # compile + install into ../Ptx
src/HartsyInference.Cuda/Kernels/dequant/build.sh --no-install   # compile only

# No nvcc on the box — use the committed nvrtc helper (dlopens libnvrtc.so):
cc -O2 -o Kernels/nvrtc_compile Kernels/nvrtc_compile.c -ldl        # build the helper once
LD_LIBRARY_PATH=~/.local/lib/cuda13 Kernels/nvrtc_compile in.cu out.ptx compute_80 "$TINC"
cp out.ptx src/HartsyInference.Cuda/Ptx/
```

Target `sm_80` minimum (forward-JIT-compatible). Verify every emitted PTX starts `.version 9.0` — the driver
JIT caps there (see `docs/Checklists/TROUBLESHOOTING.md` § CUDA toolchain). Validate a new/changed kernel
against the CPU reference within tolerance before shipping (`docs/Agents/KERNEL.md`).

### The shipped PTX must reproduce from source — check it

Every `build.sh` falls back to `../nvrtc_compile` when nvcc is absent, so **all eight domains build on a box
with no CUDA toolkit**. They did not always: `attention/`, `audio/` and `lm/` were nvcc-only, so on a
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

Two kernels, `lm_f32` and `mul_mat_vec_q6k_q8_1`, are **nvcc-built and do not reproduce under nvrtc** — the
sources are current, only register allocation differs. They are deliberately left as-is: both are LLM-decode
hot paths whose throughput was tuned against llama.cpp, and swapping codegen without a decode benchmark
would risk that silently. Regenerate them only alongside a perf run.
