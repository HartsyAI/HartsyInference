# DeepSeek-V4.1-Flash: rented hardware runbook

Read this before renting anything. It says what each rented lane can show today, what to prepare, and what to send back.
The rig order for the home box is in [MOE_RIG_READINESS.md](MOE_RIG_READINESS.md). The gates are frozen in
[DSV41_CAMPAIGN_FREEZE.md](DSV41_CAMPAIGN_FREEZE.md). Every lane runs through `tests/dsv41-certification.sh`, which refuses
a lane it cannot run honestly.

## 1. What a rented box can show today

| Lane | Rent | What it can show now | What it cannot show |
|---|---|---|---|
| `cpu-oracle` | A CPU host with 512 GB or more of RAM and at least 1 TB of NVMe. No GPU needed. | The real checkpoint against the unmodified upstream model: 40 layers in structural mode with 16 teacher-forced steps (`DSV41_FULL_RUN=1`), DSpark on real weights, the vision tower, the tokenizer and parser on the real tokenizer, and the Engram shards. | Any GPU behaviour. These are CPU reference results. |
| `gpu-expert` | One SM 8.0 or newer GPU per run. | The six existing CUDA suites for the expert cache, MoE primitives and quant workspace (step 3 of [MOE_RIG_READINESS.md](MOE_RIG_READINESS.md)), on that card. | V4.1 generation on a GPU. The GPU executors are not built (plan PRs 14 and 24). The full GpuIntegration category (its step 4) needs the other models, so it stays with the home rig. |
| `gpu-native` | SM 10.0 (datacenter Blackwell) or SM 12.0 (RTX 50xx) on every device. | The block-scaled FP4 suites through cuBLASLt: NVFP4 GEMM and its resident parity, the derivative-format dequantization, and EXL3. | Nothing is recorded for either SM in this repository, so the first run is the first evidence. V4.1 itself is not run here. |
| `multi-gpu`, `two-node`, `offload`, `vulkan-amd` | Nothing yet. The runner marks these lanes BLOCKED. | Nothing. | Plan PRs 20 to 22 (collectives, expert parallel, multi-node), PR 14 (home-lab first token), PR 24b (V4.1 on Vulkan). |

Renting a GPU today certifies the existing CUDA suites on that card. It does not certify V4.1 on a GPU.
Start with `cpu-oracle` for new real-weight evidence. Use `gpu-expert` to confirm the CUDA code on another SKU, and
`gpu-native` only when a Blackwell card is the point.

## 2. Preparation checklist

- **Disk.** The pinned checkpoint is 48 shards totalling 510,296,708,312 bytes (475.3 GiB), plus the index, config and
  tokenizer. Allow 600 GB for the checkpoint alone. The oracle dumps are not sized yet, so measure the first dump before
  you plan the rest.
- **Driver and PTX.** The shipped PTX is PTX ISA 9.0 (80 files) and 7.0 (28 files). The driver must JIT ISA 9.0, which
  means driver 580.x or newer. The "PTX ISA version trap" in [TROUBLESHOOTING.md](TROUBLESHOOTING.md) has the details.
  Check the image with `nvidia-smi` before anything else.
- **Compute capability.** The baseline kernels target SM 8.0 and newer. The block-scaled FP4 suites need SM 10.0 or 12.0.
- **Software.** .NET 10 SDK, git, and Python 3 for the oracle environment.
- **Secrets.** A Hugging Face token never goes into the repository or into chat. Export it for the session, only while
  downloading.
- **Commit.** Record the commit of `main` you run. Every evidence file must name it.

## 3. Get the weights onto the box

Pick one. The costs differ, and the choice is yours.

1. **Download on the host.** `hf download deepseek-ai/DeepSeek-V4.1-Flash --revision dba1be0a40aa45a94ad051997016db3960a90277 --local-dir <checkpoint dir>`,
   with `HF_TOKEN` set for the session. The cost is host download time and any egress fees the provider charges.
2. **Copy from the home RAID.** Copy `/mnt/model-storage/Models/llm/deepseek-v4.1-flash` with `rsync`. Measure the upload
   first: 475 GiB over a home connection can take days.

Verify the transfer:

- 48 `*.safetensors` shards and `model.safetensors.index.json`, with the byte total above.
- The 36 non-weight files against the pin:
  `~/dsv41-ref/.venv/bin/python tests/python-reference/deepseek_v41/fetch_upstream.py --verify-only`.

## 4. Set up the code and the oracle environment

```bash
git clone <repo-url> HartsyInference && cd HartsyInference && git checkout <commit on main>
tests/python-reference/deepseek_v41/setup_env.sh   # venv at ~/dsv41-ref/.venv; DSV41_REF_ROOT overrides it
```

`setup_env.sh` fetches the upstream non-weight files. If the host has no internet access, copy the upstream folder in,
then run `fetch_upstream.py --verify-only`. The runner builds each test project itself.

## 5. Make the oracle dumps (CPU)

Run these from `tests/python-reference/deepseek_v41`. `CKPT` is the checkpoint directory and `OUT` a results folder.

```bash
PY=~/dsv41-ref/.venv/bin/python
$PY dump_real_layers.py $CKPT $OUT/layers --layers 40 --mode structural --steps 16 --threads 8   # DSV41_ORACLE_DIR=$OUT/layers
$PY dump_real_dspark.py $CKPT $OUT/dspark --max-prompt-tokens 160 --threads 8                    # DSV41_DSPARK_ORACLE=$OUT/dspark
$PY dump_real_slices.py $CKPT $OUT/slices                                                        # DSV41_REAL_SLICES=$OUT/slices
$PY dump_real_vision.py $CKPT $OUT/vision --grids 28x28,17x23                                   # DSV41_VISION_ORACLE=$OUT/vision
$PY ../dump_deepseek_v41_shard3_ref.py --shard $CKPT/model-00003-of-00048.safetensors --out $OUT/shard3   # HARTSY_DSV41_SHARD3_FIXTURES=$OUT/shard3
```

Notes:

- `dump_real_layers.py` writes a `meta.json` naming the layers, mode, ids and the oracle's greedy tokens. The test reads
  it, so keep the output folder layout as written.
- The default prompt ids are `0,671,6102,294,8760,344`. The 160-token DSpark dump uses `--max-prompt-tokens 160`.
- On the home box the dumps ran under `systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0`. On a rented host,
  set the cap below physical RAM minus the working set.
- Earlier real-weight runs on the home box took 30 to 50 minutes each, with peak RSS near 30 GB. Re-measure on the host.
- After the shard-3 dump, confirm `$OUT/shard3/manifest.tsv` exists. The shard-3 test needs it.

## 6. Run the certification

```bash
export HARTSY_DSV41_FLASH_DIR=$CKPT
export HARTSY_DSV41_SHARD3_FIXTURES=$OUT/shard3
export DSV41_ORACLE_DIR=$OUT/layers DSV41_DSPARK_ORACLE=$OUT/dspark DSV41_REAL_SLICES=$OUT/slices DSV41_VISION_ORACLE=$OUT/vision
export DSV41_TOKENIZER_JSON=$CKPT/tokenizer.json DSV41_FULL_RUN=1

tests/dsv41-certification.sh cpu-oracle --dry-run   # prints the plan; runs nothing
tests/dsv41-certification.sh cpu-oracle             # green only when every class passed and nothing was skipped
```

GPU lanes take the same form: `tests/dsv41-certification.sh gpu-expert` or `gpu-native`. The runner refuses to start
while any compute process holds a device, so stop everything else on the GPU first. Lanes run one class at a time.

Exit codes: 0 green, 1 a class failed, 2 blocked or preflight failed, 64 usage error.

## 7. What to send back

Send the `Output/dsv41-certification/<stamp>/<lane>/` folder, which holds `summary.txt` and `logs/`. Also send:

- the commit SHA you ran;
- `nvidia-smi` output (GPU and driver), `free -g`, and `df -h` for the checkpoint volume;
- the shard byte total and the `fetch_upstream.py --verify-only` output.

Do not write parity rows yourself. The evidence goes into [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md) after it is
reviewed, as the plan requires.

## 8. Stop rules and cost control

- Stop the instance when a lane ends. Keep the checkpoint on a volume you can reattach, or accept a re-download.
- A failure that does not reproduce on a second run is a finding, not a pass.
- Stop if the driver does not match the PTX ISA. Stop if any class reports SKIPPED. With `HARTSY_REQUIRE_REAL_WEIGHTS=1`,
  missing assets fail the run.
- No throughput promises. Report measured numbers only, as the plan requires.

## 9. Decisions needed before renting

1. Which lane first. `cpu-oracle` gives new real-weight evidence. `gpu-expert` confirms the CUDA suites on another SKU.
2. How the weights reach the host: a download on the host, or a copy from the RAID.
3. The budget and the time cap for each session.

## Home box status (2026-10-09)

SwarmUI and RustDesk hold GPU memory on this box, so `gpu-expert` and `gpu-native` refuse to start here until they stop.
That is deliberate: the shared-GPU rule does not allow a suite to run beside Swarm. Nothing was stopped for this check.
