# MoE runtime: GPU rig readiness

What must hold before GPU time is spent, what is already verified on CPU, and the order to run things on the rig. Status
lives here until a rig run replaces it with evidence.

## Verified on CPU (this machine, no GPU)

CPU-lane baseline per test project, with the GPU and real-weight tests excluded. The baseline ran on `main` before the
GPU-trait labelling in #307; that labelling takes CUDA to zero CPU-lane tests and clears the Diffusion and Video failures.

| Project | Result |
|---|---|
| API, Audio, Audio.Phonemizer, BenchmarkRunner, Cli, Core, Cpu, Gpu, LLM, ModelAssets, ModelAssets.Tokenizers, ThreeD, Tools, Vision, Voice, World | all passing |
| Diffusion | 1869 passing, 4 GPU or real-weight failures (labelled in #307) |
| Video | 115 passing, 3 GPU failures (labelled in #307) |
| Cuda | 199 passing, 282 failing: the GPU tests ran in the CPU lane (labelled in #307) |
| Vulkan | not summarized by the baseline runner; checked separately |

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
- **Toolkit.** `nvcc` here is CUDA 11.5. The repo's committed PTX targets `sm_70`, `sm_75`, `sm_80`, and one `sm_120a`
  file. 281 CUDA tests fail with a PTX JIT error ("SM version specified by .target is higher than default SM version"),
  which points at the toolchain as much as the driver. Confirm the toolkit the rig uses against the PTX targets before the
  first GPU run.
- **Host memory.** 39 GB total. With the CPU lane running, 2 GB was free and 9 GB available. Check `free -g` before any
  large load (see the no-heavy-GPU-runs note).

## Rig run order

1. `nvidia-smi` shows the GPU and a driver that matches the library. No mismatch message.
2. GPU lane, one suite at a time, never alongside another run:
   `dotnet test tests/HartsyInference.Cuda.Tests --filter "Category=GpuIntegration"`. Record failures by test name.
   The expert-cache suites are the first to run: `CudaExpertCacheTests`, `CudaExpertM1FixtureTests`,
   `CudaMoePrimitiveTests`, `CudaMoeTests`, `CudaQuantWorkspaceTests`, `CudaStreamingWeightCacheTests`.
3. Refactor A/B for the cache changes: `tests/regression-ab.sh --expect identical --backend cuda` on a generation case.
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
