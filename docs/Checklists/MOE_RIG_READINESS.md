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
| Cuda | 157 CPU-only tests passing, none failing; GPU methods labelled (#307) |
| Vulkan | 12 passing, exit 0 after the device tests were labelled (#309); the run hung before |

Subsystems with CPU tests in place:

- Sparse topology contracts and fingerprint (#299).
- Scalar reference path with parity against `MoeFeedForward` on softmax and grouped sigmoid routing (#300).
- Expert identity with banks; cache keyed by (bank, layer) (#303).
- Residency queries and no-upload acquisition; allocation-free per call (#304).

Pending, not verified in this checkout:

- Expert pack (#305): exact payload sizes, round trips, refusals and corruption checks. No `ExpertPack` code or tests
  exist in this checkout.
- Placement planner (#306): residency decides placement, every routed pair counted once, planning changes no state. No
  planner code or tests exist in this checkout.

Do not count #305 or #306 as verified until those changes land on the branch being run.

## Blockers on this machine (fix before renting or on the rig)

- **NVIDIA driver and library mismatch.** `nvidia-smi` reports "Driver/library version mismatch" (NVML library 595.99).
  No CUDA test can run until the driver matches the library. A reboot or a matching reinstall is the usual fix; this is
  a system change, so it was not made from here.
- **Driver and PTX.** The rig loads the shipped PTX from disk; `nvcc` is not in that path. The constraints are the deployment
  driver's PTX-ISA ceiling and the GPU's exact compute capability (`src/HartsyInference.Cuda/Kernels/README.md`). The
  committed PTX targets `sm_70`, `sm_75`, `sm_80`, and one `sm_120a` file. 281 CUDA tests failed here with a PTX JIT error
  ("SM version specified by .target is higher than default SM version"): that points at the driver on this machine, so check
  the driver's PTX-ISA support against those targets on the rig before the first GPU run.
- **Host memory.** 39 GB total. With the CPU lane running, 2 GB was free and 9 GB available. Check `free -g` before any
  large load (see the no-heavy-GPU-runs note).

## Rig run order

1. `nvidia-smi` shows the GPU and a driver that matches the library. No mismatch message.
2. Real-weight assets, set before any GPU suite. With `HARTSY_REQUIRE_REAL_WEIGHTS=1` a missing asset fails the test
   instead of logging `SKIPPED`, so a run with missing assets is a failed run, not a green one:
   ```
   export HARTSY_REQUIRE_REAL_WEIGHTS=1
   export HARTSY_DSV41_FLASH_DIR=<dir holding model-00003-of-00048.safetensors>
   export HARTSY_DSV41_SHARD3_FIXTURES=<dir holding manifest.tsv>   # default: ~/dsv41-ref/shard3_fixtures
   ```
   Confirm both paths exist before starting. `CudaExpertM1FixtureTests` is the only suite that reads them.
3. Expert-cache suites first, one class per invocation, one at a time. xUnit does not run classes in a caller-chosen
   order, so the category filter cannot put these first. The trailing dot makes each filter match one class exactly:
   ```
   for c in CudaExpertCacheTests CudaExpertM1FixtureTests CudaMoePrimitiveTests CudaMoeTests \
            CudaQuantWorkspaceTests CudaStreamingWeightCacheTests; do
     dotnet test tests/HartsyInference.Cuda.Tests --filter "FullyQualifiedName~HartsyInference.Cuda.Tests.$c."
   done
   ```
   Before #307 lands, `CudaMoeTests` and `CudaStreamingWeightCacheTests` carry no `Category` trait, so the category filter
   does not select them; this explicit loop is how they run on the rig either way.
4. The rest of the GPU category, excluding the classes already run in step 3. Run only from a checkout that includes #307,
   which applies the GPU labels this filter depends on. Record failures by test name:
   ```
   dotnet test tests/HartsyInference.Cuda.Tests --filter "Category=GpuIntegration&FullyQualifiedName!~HartsyInference.Cuda.Tests.CudaExpertCacheTests.&FullyQualifiedName!~HartsyInference.Cuda.Tests.CudaExpertM1FixtureTests.&FullyQualifiedName!~HartsyInference.Cuda.Tests.CudaMoePrimitiveTests.&FullyQualifiedName!~HartsyInference.Cuda.Tests.CudaQuantWorkspaceTests."
   ```
5. **Deferred: refactor A/B for the cache changes.** Do not run `tests/regression-ab.sh` as evidence for #304 yet. No
   production path constructs `CudaExpertCache` (only the class and its tests do), and no core regression case drives it,
   so both arms would bypass the changed cache and could report identical output vacuously. The requirement that
   generations stay identical stays open. Evidence for the cache refactor on the rig is step 3. Reopen this step once an
   executor or a dedicated harness acquires and evicts experts through the cache; then run
   `tests/regression-ab.sh --fresh --expect identical --backend cuda` on that case.
6. Vulkan MoE primitives, after the CUDA lane has finished, with no other GPU run in progress:
   `dotnet test tests/HartsyInference.Vulkan.Tests --filter "FullyQualifiedName~HartsyInference.Vulkan.Tests.VulkanMoePrimitiveTests."`.
   This suite needs a local Vulkan device and no fixtures.

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
- Stop if any real-weight suite reports `SKIPPED` while `HARTSY_REQUIRE_REAL_WEIGHTS=1` is set, or if its assets are missing.
