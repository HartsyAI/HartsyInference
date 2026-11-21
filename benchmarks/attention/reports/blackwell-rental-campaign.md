# Blackwell rental campaign and quote (draft, nothing provisioned)

The plan makes a consumer Blackwell card a blocking **final** CUDA and Vulkan gate and asks for "an exact rental
campaign and spending quote before requesting provisioning" (`proposed-queue.md`, evidence cadence). This is that
campaign as it can be written today. **No pod has been created and nothing has been spent.** Provisioning needs a
separate yes from the user, with the figures below read again at that time.

It is written before the series is finished, so the commit to run and the size of the corpus are not final. What is
fixed here is the order of the stages, what each must show, and the ceiling on time and money.

## Card and price

Read from the RunPod GraphQL `gpuTypes` list on 2026-11-21. These are list prices by the hour; they say nothing about
whether a card is free in a given data center at that moment.

| Card | Memory | Secure cloud | Community cloud |
| --- | --- | --- | --- |
| RTX 5090 | 32 GB | $1.39 | $0.69 |
| RTX 5080 | 16 GB | $0.59 | $0.39 |
| RTX 4090 (for comparison) | 24 GB | $0.89 | $0.34 |

Proposed: **one RTX 5090 on the secure cloud**. It is the consumer Blackwell card (SM 12.0) with the memory to run
every workload of the frozen protocol, and the secure cloud is where the existing network volume can be attached. The
RTX 5080 is cheaper and also SM 12.0, but its 16 GB would turn memory-bound cells into skips, and a skip is
unvalidated under the plan.

Not checked: whether an RTX 5090 is available in the data center of the existing volume (EU-RO-1). If it is not, the
choice is a new volume in another data center (the weights are copied again) or the community cloud without a volume.

## What is not known and decides the campaign

1. **Vulkan on a pod.** An earlier rental had none: `benchmarks/results/2026-10-11_moe_perf_a40.md` records "no
   Vulkan ICD in the container". The deleted runner (item 4) names the cause and the remedy: a container started
   without the graphics capability shows the loader no NVIDIA ICD, and the pod has to be created with
   `NVIDIA_DRIVER_CAPABILITIES=all`; it also installs `libvulkan1` and `vulkan-tools` when `vulkaninfo` is missing.
   So the remedy is recorded, but no report here records a Vulkan device working on a pod after it. Stage 1 creates
   the pod with that variable from the start. Only if `vulkaninfo --summary` still shows no NVIDIA device with it set
   does the campaign stop after stage 1, with the finding that the Vulkan half cannot be run at this provider. That
   is why stage 1 is separate and cheap.
2. **Driver.** Every nvcc-built PTX shipped here is ISA 9.0 and needs driver 580 or later
   (`benchmarks/CLOUD_GPU_RUNBOOK.md`). The host driver of a pod is not chosen by the renter.
3. **How long the frozen corpus takes.** No report in this repository records the wall time of a full campaign on
   the RTX 4090, so the hours below are a ceiling set here, not a measurement. The one duration behind them is an
   observation that no report records: the attention GPU test lane of the integration branch at `b4651cdb6` (345
   tests listed) took between 1.5 and 3 minutes in each of three runs on the RTX 3060 on 2026-11-21.
4. **The scripted runner.** The runbook names `tests/blackwell-run.sh` as the scripted form of the rental. That file
   was deleted in `e9474bb06` and is not on `main`. Either it is restored (it exists at `0a67f89e5` and `ab04134cf`)
   and rehearsed on the local card first, as the runbook asks, or the stages are driven by hand over SSH.
5. **Weights on the volume.** The protocol needs Qwen3-4B Q4_K_M and Krea2 Turbo. Whether they are on the volume is
   not checked. Copying or downloading them is a download and needs its own approval.

## Stages

Each stage has a hard time budget; a stage that reaches it stops and is recorded as incomplete, and the pod is stopped.
One GPU job at a time. Whoever drives the rental also checks the pod every 10 minutes and stops it when it sits idle;
that is a working rule of the driving session, not a script in this repository. The in-repository equivalents are the
`--budget-minutes` and `--auto-stop` options of the deleted runner, if it is restored.

| Stage | What it shows | Budget | Stop if |
| --- | --- | --- | --- |
| 1. Probe | Pod created with `NVIDIA_DRIVER_CAPABILITIES=all`; `libvulkan1` and `vulkan-tools` installed in the container if missing; `nvidia-smi` (card, driver at least 580), `vulkaninfo --summary` (an NVIDIA Vulkan device), `dotnet build`, the attention GPU lane on CUDA and Vulkan with the total compared against `--list-tests` | 30 min | no NVIDIA Vulkan device with the graphics capability requested, driver below 580, or the lane host crashes |
| 2. Semantics and routes | The complete CPU/CUDA/Vulkan attention suites and the route matrix on the final commit; every skip listed with its reason | 1 h | any failure that the recorded RTX 4090 run of the same commit does not show (that run exists before the rental; it is not repeated for the comparison) |
| 3. Operation cells | The frozen operation corpus and Kitchen comparison under `protocol.json` (3 sessions of 5 pairs and the confirmation session), CUDA then Vulkan | 3 h | an invalid device clock, or a cohort violation |
| 4. Primary workloads | Qwen and Krea primaries against Bfix under `protocol.json`, with real-generation correctness | 2 h | weights missing, or a required path proof absent |
| 5. Collect | Copy artifacts off the pod, verify hashes, stop the pod, confirm in the control plane that it is stopped | 30 min | |

## Quote

Ceiling of **7 pod-hours on one RTX 5090, secure cloud: $9.73** at the price above, plus the storage of the existing
volume, which is charged whether or not this runs. Stage 1 alone is at most **$0.70**.

Proposed as two approvals: stage 1 first (30 minutes, at most $0.70), then stages 2 to 5 only if stage 1 finds a
Vulkan device and a new enough driver, with the hours revised from what stage 1 measured. A second attempt after a
failed stage is a new request with its own figure, not an extension of this one.

Not in the quote: any rerun, a second card (the plan requires only one consumer Blackwell cohort), SM75 hardware
(stays unverified under the plan), and the three stopped pods already in the account, which this campaign does not
touch.

## Before asking for provisioning

- The series is merged and the final commit is named.
- The RTX 3060 and RTX 4090 have both run that commit, so a Blackwell failure can be told from a general one.
- The runner question above is settled and, if the script is restored, rehearsed locally.
- The weights question is answered by listing the volume from a running pod or the provider file browser.
- Prices and availability are read again; the figures above are a day old the day after they were written.
