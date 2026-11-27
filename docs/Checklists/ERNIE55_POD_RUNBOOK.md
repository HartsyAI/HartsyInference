# ERNIE-5.5-Flash-Base: one-pod real-weight run

This is the runbook for the first real-weight run of `baidu/ERNIE-5.5-Flash-Base` on one rented CUDA pod. What the run is
and what it cannot show is in [ERNIE55_REAL_WEIGHTS.md](ERNIE55_REAL_WEIGHTS.md). The steps below are wrapped by
`scripts/ernie55_pod_run.sh`, which stops on the first failure. No CUDA run has been made for this runbook yet.

## 0. Decisions that are yours, before any spend

- **License gate (not decided here).** `Ernie5Catalog.cs` records the license `ernie-5.5-flash-base-model-license-v1.0`. It grants
  use only outside the EU, the UK and South Korea, and it needs separate authorization for commercial use above US$50M annual
  revenue (LICENSE 1.11, 2, 4.3, 5.3). Confirm this, for the place and the use you run it under, before the download. The script
  refuses to download until you export `ERNIE_CONFIRM_LICENSE=1`.
- **Download and rental.** About 288 GB moves onto a network volume, and the pod is billed while it runs. Confirm the budget.
- **No chat template.** The pinned tree has no `chat_template.jinja`, and the Text path requires one. The run therefore feeds
  token ids (`ERNIE_REAL_TOKENS`, default `1,2,3,4,5`) and no prompt text. This is a base model; the test checks that the forward is
  finite and non-constant, not that the output is correct text.

## 1. Pod spec

- **GPU:** one NVIDIA GPU with at least 24 GB free. The script gates on 24,576 MiB free and on no other compute process.
- **VRAM need, from the code.** The harness sets `ResidentWeights = false`, so the packed weights stay in host memory and each
  forward uploads what it needs. No VRAM budget is written in the code, and no peak has been measured. Reading the CUDA path:
  - The LM head is BF16 `[201216, 4096]`: 1,648,361,472 bytes (1.54 GiB). `Ernie5Model.Logits` calls `Linear` on it.
  - `CudaBackend.LinearImpl` casts a BF16 weight that is not in the cache to the GEMM dtype for that call. The test uses
    `HighPrecisionGemm = true` (F32 GEMMs), which makes that cast 3,296,722,944 bytes (3.07 GiB). That is an estimate from the
    code, not a measurement.
  - The largest FP8 linear is the dense FFN, 14336 x 4096, 58,720,256 bytes as FP8, about 235 MB if expanded to F32.
  - Working estimate: about 5 GB of device memory at peak. 24 GB is enough for that estimate, with the activations and the
    allocator on top. This is not measured. Record the peak (section 8).
- **Comfortable option:** a 48 GB card (L40S, RTX 6000 Ada, A6000) or an 80 GB card (A100 80 GB, H100 80 GB). It leaves room
  for the measured peak to exceed the estimate. The RTX 3060 on the home box (12 GB) is below the floor and cannot run this.
- **Compute capability:** Ampere (sm_80) or newer. Turing (sm_75) has only the floor build of some kernels; it is not verified
  for this model and the script does not target it.
- **Driver:** 580.x or newer. The shipped PTX is ISA 9.0, which the driver must JIT. This is the same requirement as
  [DSV41_RENTAL_RUNBOOK.md](DSV41_RENTAL_RUNBOOK.md) section 2 and the PTX trap in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
- **Host RAM and CPU:** not derived from the code. Record the peak RSS of the test process (section 8).
- **Image:** .NET 10 SDK, git, `curl`, the Hugging Face CLI (`hf`, from `huggingface_hub`) and `nvidia-smi`. `nvcc` is not
  needed to run the test.

## 2. Network volume

- **Bytes.** The 128 shards total 288,434,400,632 bytes (268.6 GiB), per the pinned revision's tree listing. The index is
  13,757,080 bytes and `config.json` is 3,814 bytes, so the 130 files the script checks total **288,448,161,526 bytes**. The
  index's own `metadata.total_size` (288,416,689,496) is not used.
- **Size.** Provision **400 GB** for one copy, which leaves about 110 GB for the build, logs and packages. If the hub client
  keeps a second copy in its cache, the volume needs about twice the checkpoint (about 577 GB). Check after the first few
  shards (`du -sh "$CKPT"` against the count of shards written) and stop if the copies are doubling.
- **Steps.**
  1. Create the network volume in the same region as the pod. A volume cannot follow a pod across regions.
  2. Mount it (for example at `/workspace`). Then:
     ```bash
     export CKPT=/workspace/ERNIE-5.5-Flash-Base
     export HF_HOME=/workspace/hf-cache      # keep the hub cache on the volume
     df -h /workspace                        # at least 400 GB free before the download
     ```
  3. Clone the repository and check out the branch under test. Record the commit (`git rev-parse HEAD`).

## 3. Download (after section 0 is confirmed)

```bash
export ERNIE_CONFIRM_LICENSE=1
export CKPT=/workspace/ERNIE-5.5-Flash-Base
hf download baidu/ERNIE-5.5-Flash-Base --revision 028544b2ffceea189b2d8768ed2b9d09b06f2343 \
  --local-dir "$CKPT" --include "*.json" --include "*.safetensors"
```

The repeated `--include` works in `hf` 2.4.0: a `--dry-run` with the same flags listed both the JSON files and the shards. Another
`hf` version may treat a repeated flag differently, so run `hf download … --dry-run` with the same flags first and check that the
listing has the shards and the JSON files before the 288 GB transfer. The script's `download` stage runs `verify` afterward, which
would also catch a missing index or config, but only after the transfer.

The tree API answered without a token when this runbook was written. If `hf` asks for a login, export `HF_TOKEN` for this
session only, and never commit it or paste it into a log. The script wraps this command as `scripts/ernie55_pod_run.sh download`,
which then runs the verification below.

Expect 133 files: 128 shards, `model.safetensors.index.json`, `config.json`, and the tokenizer and generation JSON files. The
`assets/` folder is not needed.

## 4. Verify the transfer

Size and count, from the manifest `scripts/ernie55-flash-base-028544b2.manifest` (130 files, with the SHA-256 the hub lists for
each LFS file):

```bash
export ERNIE_CKPT_DIR="$CKPT"
scripts/ernie55_pod_run.sh verify                      # sizes, counts, and the index names exactly 128 shards
ERNIE_VERIFY=sha256 scripts/ernie55_pod_run.sh verify  # also hashes every LFS file; run it once per new volume
```

Without the script:

```bash
ls "$CKPT"/model-*-of-00128.safetensors | wc -l                                                # 128
du -cb "$CKPT"/model-*.safetensors "$CKPT"/model.safetensors.index.json "$CKPT"/config.json | tail -1   # 288448161526
```

`config.json` has no LFS hash, so it is checked by size only. The SHA-256 pass reads the whole volume and can take a long time on
a network volume. It is the only check that catches a silently corrupted shard, so run it once.

The test's own shard check (`RealShardsFact`) only checks that each named file exists. It does not check sizes; the script does.

## 5. PTX and architecture

- **Shipped PTX.** `src/HartsyInference.Cuda/Ptx` holds 209 committed files. Check the ISA with
  `src/HartsyInference.Cuda/Kernels/check_ptx_isa.sh`. The script runs it in the `ptx-check` stage, which also requires
  `fp8_block_linear.ptx`, `mul_mat_vec_f16_bf16_f32.ptx` and `lm_f32.ptx`.
- **Arch targets that exist** (from the build lists in `src/HartsyInference.Cuda/Kernels`):
  - Baseline `sm_80`: the `KERNELS` lists, for example `lm/build.sh` (`lm_*`, `flash_attn_*`, `mul_mat_vec_*`). A `.target sm_80`
    file loads on any device at or above sm_80 through JIT.
  - Turing floor `sm_75`: `<kernel>.sm75.ptx` for each `KERNELS` entry that is not `SM80_ONLY`, and the `KERNELS_SM75` entries
    such as `fp8_block_linear` (its shipped header is `.target sm_75`).
  - Tuned variants that are shipped: `block_quant.sm120.ptx` (sm_120a) and `int8_attn_v2.sm89.ptx` (sm_89). There is no
    `block_quant.sm89.ptx`, and no sm_86 or sm_90 variant.
- **Pod GPU mapping.** None of these is a blocker:
  - sm_86 (A10, A40, A6000, RTX 3090): the sm_80 baseline and the sm_75 `fp8_block_linear` both load.
  - sm_89 (L4, L40S, RTX 4090): the baseline loads. `block_quant` uses the baseline, not a tuned file, so this is a speed
    question, not a correctness one.
  - sm_90 (H100): the sm_80 baseline is JIT-compiled forward. Nothing targets sm_90 specifically.
  - sm_120 (RTX 50 series): `block_quant.sm120.ptx` loads.
  - **The real blocker** is a driver older than 580, which cannot JIT ISA 9.0.
- **Build step.** The run needs no build. Do not rebuild on the pod for this run. The lists in `build_common.sh` need the pinned
  CUDA 13.0.88 `nvcc` for the `NVCC_BUILT` kernels, and without it those are reported UNVERIFIED and left as shipped. An
  arch-specific build is only needed to add a variant that is not shipped, for example
  `src/HartsyInference.Cuda/Kernels/dequant/build.sh --arch sm_89`. That is optional, and if you do it, record the output.

## 6. GPU-free check

Run it in the same command as the launch, which `scripts/ernie55_pod_run.sh all` does:

```bash
scripts/ernie55_pod_run.sh gpu-check
```

Expected: `GPU free: <name>, <MiB> MiB free, no compute process`. Any other output stops the run. Manually:
`nvidia-smi --query-gpu=name,memory.free --format=csv` (one row) and `nvidia-smi --query-compute-apps=pid,process_name --format=csv`
(no rows).

Most rented pods do not show processes from other containers in `--query-compute-apps`, so an empty list there is weak evidence.
The free-MiB reading is the check that matters; the process list only catches processes in the same container.

## 7. The test

```bash
export ERNIE_REAL_DIR="$CKPT" ERNIE_REAL_DEVICE=cuda ERNIE_REAL_TOKENS=1,2,3,4,5
dotnet test tests/HartsyInference.Engine.Tests --filter "FullyQualifiedName~Ernie5RealCheckpointTests" \
  --logger "console;verbosity=normal"
```

The whole sequence is `scripts/ernie55_pod_run.sh all`, which runs preflight, verify, gpu-check, ptx-check and test in that
order, and never the download. It writes the log, a 0.5 s VRAM sample and `summary.txt` to `ERNIE_RUN_DIR`.

Only `test` can be rehearsed without a GPU (`ERNIE_REAL_DEVICE=cpu scripts/ernie55_pod_run.sh test`, on a synthetic checkpoint). `all`
runs `gpu-check` and `ptx-check`, which need the GPU.

**Expected output**

- `verify`: `sizes ok: 130 manifest files, 288448161526 bytes, 128 shards`.
- `gpu-check`: `GPU free: ...`.
- `ptx-check`: `ptx ok: ISA 9.0 throughout; targets: ...`.
- Test: `Real_IndexMapsEveryKey` passed and `Real_PrefillAndDecode_ProduceFiniteLogits` passed. The timing line is
  `ERNIE real run on cuda: load <s>, prefill 5 tokens <s>, decode 1 token <s>, argmax after prefill <id>.` The verbosity=normal
  summary is `Total tests: 2`, then `Passed: 2`, with no `Failed:` or `Skipped:` line. (The minimal-verbosity line
  `Failed: 0, Passed: 2, Skipped: 0` is not printed at this verbosity.)
- **A run that did not validate:** any `SKIPPED` line, fewer than 2 passed, or no timing line. The script treats each as a
  failure. Under the repository's rules a run like that is reported as not run.

The two facts together make about two full passes over the volume (one prefill, one decode), so budget tens of minutes of I/O
on a network volume. That is an estimate, not a measurement.

## 8. What to record in the PR

- The branch commit (from `environment.txt` in the run folder), and the pod's region and image.
- Driver version, CUDA version (the `CUDA Version` line of `nvidia-smi`), GPU model and compute capability.
- Load time, prefill time for 5 tokens, and decode time for 1 token, from the timing line.
- Peak VRAM from `vram_mib.log` (the maximum of the 0.5 s samples) against the total on the card. Record peak host RSS too if
  you measure it (for example `/usr/bin/time -v`).
- The `summary.txt` and the test log, attached or linked. The counts: 2 passed, 0 skipped, 0 failed.
- The verification result: size pass, and the SHA-256 pass if you ran it.
- What is not verified, copied into the PR: the indexer ReLU, real-weight numerics against a reference run, and production
  precision (TF32 and F16 GEMMs are not measured by this harness, see [ERNIE55_REAL_WEIGHTS.md](ERNIE55_REAL_WEIGHTS.md)).

## 9. Teardown

Stop the pod. The network volume keeps the checkpoint and costs storage while it exists. Decide whether to keep it. This runbook
does not delete it.
