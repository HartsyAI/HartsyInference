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

1. **Vulkan on a pod.** Nothing in this repository records a Vulkan device inside a RunPod container. The backend
   needs the NVIDIA Vulkan driver exposed to the container, not only CUDA. If `vulkaninfo --summary` shows no NVIDIA
   device on the pod, the Vulkan half of the gate cannot be run at this provider and the campaign stops after stage 1
   with that finding. That is why stage 1 is separate and cheap.
2. **Driver.** Every nvcc-built PTX shipped here is ISA 9.0 and needs driver 580 or later
   (`benchmarks/CLOUD_GPU_RUNBOOK.md`). The host driver of a pod is not chosen by the renter.
3. **How long the frozen corpus takes.** No report in this repository records the wall time of a full campaign on
   the RTX 4090, so the hours below are a ceiling set here, not a measurement. The one duration measured is the
   attention GPU test lane: 345 tests in 1.5 to 3 minutes on the RTX 3060.
4. **The scripted runner.** The runbook names `tests/blackwell-run.sh` as the scripted form of the rental. That file
   was deleted in `e9474bb06` and is not on `main`. Either it is restored (it exists at `0a67f89e5` and `ab04134cf`)
   and rehearsed on the local card first, as the runbook asks, or the stages are driven by hand over SSH.
5. **Weights on the volume.** The protocol needs Qwen3-4B Q4_K_M and Krea2 Turbo. Whether they are on the volume is
   not checked. Copying or downloading them is a download and needs its own approval.

## Stages

Each stage has a hard time budget; a stage that reaches it stops and is recorded as incomplete, and the pod is stopped.
One GPU job at a time. The pod idle check of this project (10 minutes) is armed at every start.

| Stage | What it shows | Budget | Stop if |
| --- | --- | --- | --- |
| 1. Probe | `nvidia-smi` (card, driver at least 580), `vulkaninfo --summary` (an NVIDIA Vulkan device), `dotnet build`, the attention GPU lane on CUDA and Vulkan with the total compared against `--list-tests` | 30 min | no Vulkan device, driver below 580, or the lane host crashes |
| 2. Semantics and routes | The complete CPU/CUDA/Vulkan attention suites and the route matrix on the final commit; every skip listed with its reason | 1 h | any failure that is not reproduced on the RTX 4090 at the same commit |
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
