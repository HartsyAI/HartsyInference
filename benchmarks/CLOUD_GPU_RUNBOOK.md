# Cloud GPU baseline runbook

Use [methodology](../docs/Research/PROFILING_METHODOLOGY.md) and the [benchmark harness](README.md). This is an execution checklist, not a claim that an unmeasured GPU is verified.

1. Define the hardware gap, workload, artifact destination, budget and shutdown time. Verify current availability/pricing when arranging a rental; old estimates are not a budget.
2. Pin the engine commit, reference revisions, checkpoint hashes and dependencies. Record nvidia-smi, driver, compiler, .NET, GPU topology and free memory. Ensure the driver can JIT the shipped PTX and required compute libraries resolve.
3. Build Release and prepare the baseline virtual environment using repository requirements. Preload necessary checkpoints; keep secrets out of logs and artifacts.
4. Run the smoke harness, inspect failures/skips, then the matched full workload. Record power/clock/thermal limits and other GPU processes; do not alter shared-host settings without authorization.
5. Verify raw C#/reference rows, matched shapes/dtypes, fingerprints, logs and quality results. Missing checkpoints or an aborted test run are incomplete coverage even if some output files exist.
6. Copy artifacts to the agreed durable destination, verify the copy, and update the canonical scoreboard with date/source. benchmarks/results is ignored locally; do not assume output was committed or uploaded.
7. Stop the workload and terminate the rented compute; check storage and other continuing charges. Confirm completion in the provider control plane.

The scripted form of steps 2–7 is `tests/blackwell-run.sh` (preflight → bootstrap → probe → gpu-tests → models →
collect, stage-resumable, `--budget-minutes` hard stop, `--auto-stop` for RunPod). Rehearse it with `--rehearsal` on the
local card before renting: if it completes there, the pod run differs only in hardware. Its preflight refuses a driver
below 580, because every nvcc-built PTX shipped here is ISA 9.0.

## Renting, in practice

- **Pick a lean image.** `nvidia/cuda:<ver>-devel-ubuntu24.04` with a start command that installs and runs `sshd`
  is usable about two minutes in; a provider's own PyTorch image pulled for 25 minutes without finishing, billed
  the whole time, and was thrown away. Set a minimum download bandwidth on the pod if the provider offers one.
- **Ask for graphics capability if Vulkan is in scope.** Without `NVIDIA_DRIVER_CAPABILITIES=all` the container
  gets no Vulkan ICD, `vulkaninfo` reports no device, and every cross-backend parity row fails for a reason that
  has nothing to do with the code.
- **Network volumes are region-locked.** Create the pod in the volume's data center or it starts with no storage.
- **Availability is a separate query from pricing.** On RunPod the REST API covers pods and volumes, but GPU
  availability per data center is GraphQL only (`gpuTypes`, `dataCenters.gpuAvailability`). Check it before
  reserving anything, and accept a substitute by compute capability rather than model name — an RTX PRO 6000 and
  an RTX 5090 are both CC 12.0 and load the same `sm_120a` PTX.
- **cuDNN is not in the CUDA devel images.** Convolution-heavy work takes the fallback path there, so say so
  beside any number measured that way.

Three things the 2026-09-25 run did not measure on Blackwell, each cheap to fold into a later one: the
`numerics.fp4Native` off/on comparison at one seed on an nvfp4 checkpoint (Klein 4B), which is the on-card quality
evidence that knob's default is waiting for (SSIM >= 0.90 between the two); the Vulkan stage, which needs the
driver-capability flag above; and one generation driven through the SwarmUI API, whose headless first-run setup is
worth working out on a local card beforehand rather than on rented time. **Delete each line once it is measured** —
this is the current coverage gap, not a log of past ones.

PTX load failures require checking the actual emitted ISA, target and installed driver. Do not “fix” the header manually or assume an old toolkit version supports every shipped artifact.
