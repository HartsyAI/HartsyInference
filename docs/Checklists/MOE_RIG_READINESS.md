# MoE runtime: GPU rig readiness

What must hold before GPU time is spent, what is already verified on CPU, and the order to run things on the rig. Status
lives here until a rig run replaces it with evidence.

## Verified on CPU (this machine, no GPU)

CPU-lane results per test project, with the GPU and real-weight tests excluded. The first baseline ran before the GPU labels in
#307 and #309; the table shows the results after them.

| Project | Result |
|---|---|
| Sixteen others (API, Audio, Core, Cpu, LLM, ModelAssets, Vision, Voice, and eight more) | all passing |
| Diffusion | 1869 passing after the GPU methods were labelled (#307) |
| Video | 114 passing after the GPU methods were labelled (#307) |
| Cuda | 115 CPU-only methods passing, no GPU method in the CPU lane (#307) |
| Vulkan | 12 passing, exit 0 after the device tests were labelled (#309); the run hung before |

Subsystems with CPU tests in place:

- Sparse topology contracts and fingerprint (#299).
- Scalar reference path with parity against `MoeFeedForward` on softmax and grouped sigmoid routing (#300).
- Expert identity with banks; cache keyed by (bank, layer) (#303).
- Residency queries and no-upload acquisition; allocation-free per call (#304).
- Expert pack: exact payload sizes, round trips, refusals and corruption checks (#305).
- Placement planner: residency decides placement, every routed pair counted once, planning changes no state (#306).

## Blockers on this machine (fix before renting or on the rig)

- **NVIDIA driver and library mismatch.** `nvidia-smi` reports "Driver/library version mismatch" (NVML library 595.99).
  No CUDA test can run until the driver matches the library. A reboot or a matching reinstall is the usual fix; this is
  a system change, so it was not made from here.
- **Driver and PTX.** The rig loads the shipped PTX from disk; `nvcc` is not in that path. The constraints are the deployment
  driver's PTX-ISA ceiling and the GPU's exact compute capability (`src/HartsyInference.Cuda/Kernels/README.md`). The
  committed PTX targets `sm_70`, `sm_75`, `sm_80`, and one `sm_120a` file. 281 CUDA tests failed here with a PTX JIT error
  ("SM version specified by .target is higher than default SM version"): that points at the driver on this machine, so check
  the driver's PTX-ISA support against those targets on the rig before the first GPU run.
  file. 281 CUDA tests fail with a PTX JIT error ("SM version specified by .target is higher than default SM version"),
  which points at the toolchain as much as the driver. Confirm the toolkit the rig uses against the PTX targets before the
  first GPU run.
- **Host memory.** 39 GB total. With the CPU lane running, 2 GB was free and 9 GB available. Check `free -g` before any
  large load (see the no-heavy-GPU-runs note).

## Rig run order

1. `nvidia-smi` shows the GPU and a driver that matches the library. No mismatch message.
2. GPU lane, one suite at a time, never alongside another run:
   `dotnet test tests/HartsyInference.Cuda.Tests --filter "Category=GpuIntegration"`. Record failures by test name. The GPU
   labels that this filter depends on land with #307; run the suites only from a checkout that includes it.
   The expert-cache suites are the first to run: `CudaExpertCacheTests`, `CudaExpertM1FixtureTests`,
   `CudaMoePrimitiveTests`, `CudaMoeTests`, `CudaQuantWorkspaceTests`, `CudaStreamingWeightCacheTests`.
3. Refactor A/B for the cache changes: `tests/regression-ab.sh --fresh --expect identical --backend cuda` on a generation case.
   `--fresh` rebuilds both arms with their PTX; without it the script reuses cached builds and results.
   The cache refactor (#304) must leave generations identical.
4. Vulkan: `Category=GpuIntegration` for the Vulkan MoE primitives (`VulkanMoePrimitiveTests`).

## Not yet possible, and what is missing

- **Packing a real checkpoint.** The pack and verifier are tested on random weights only. There is no command-line entry
  point yet (`hartsy moe pack|verify`), so verifying a real checkpoint needs a small harness.
- **GPU execution of the planned split.** The planner decides placement; the GPU and CPU executors that consume the plan
  are not built. Their kernels need the rig.
- **Measurements the plan asks for.** Instrumented now: per-op timing (`OpProfile`) and expert-cache counters (hits, misses,
  uploads, bytes). Not yet instrumented: time to first token, tokens per forward pass, host-to-device and device-to-host
  bandwidth, peer bandwidth, CPU and GPU utilization, KV bytes.

## Stop rules

- Stop if a GPU failure does not reproduce on a second run. Nondeterminism is a finding, not a pass.
- Stop if the driver mismatch returns. Do not measure through it.
