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

## This machine (checked)

- **GPU:** NVIDIA GeForce RTX 2060 SUPER, Turing (sm_75), 8 GB, driver 595.91.07. CUDA runs here: `CudaContext.IsAvailable()`
  returns true. The `nvidia-smi` "driver/library version mismatch" message is an NVML tooling problem; it does not stop CUDA.
- **The 281 CUDA failures here are PTX JIT errors** ("SM version specified by .target is higher than default SM version"). The
  shipped PTX includes `sm_80` and `sm_120a` targets, which a sm_75 device cannot run. That is an architecture mismatch, not a
  code defect, and it is consistent with the rig being a newer GPU. It is not yet confirmed per test: check each failing
  test's PTX target on the rig, and confirm the rig's device covers the targets its suites load.
- **Host memory.** 39 GB total. With the CPU lane running, 2 GB was free and 9 GB available. Check `free -g` before any
  large load (see the no-heavy-GPU-runs note).
- **VRAM.** 8 GB limits what this card can run. Large-MoE and real-checkpoint validation needs the rented GPU.

## Rig run order

1. Preflight: CUDA must be usable, or every GPU suite would pass without running. This test fails when it is not:
   `dotnet test tests/HartsyInference.Cuda.Tests --filter "FullyQualifiedName~HartsyInference.Cuda.Tests.RigPreflightTests."`
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
   classes="CudaExpertCacheTests CudaExpertM1FixtureTests CudaMoePrimitiveTests"
   classes="$classes CudaMoeTests CudaQuantWorkspaceTests CudaStreamingWeightCacheTests"
   set -euo pipefail
   for c in $classes; do
     dotnet test tests/HartsyInference.Cuda.Tests --filter "FullyQualifiedName~HartsyInference.Cuda.Tests.$c."
   done
   ```
   `set -euo pipefail` stops the run at the first failing class, so a later pass cannot hide an earlier failure. A class
   whose tests return early without a device is caught by the preflight in step 1, not by a green count here.
   `CudaMoeTests` and `CudaStreamingWeightCacheTests` carry the GPU label from #307, so the category filter selects them in
   step 4 too; this explicit loop is how they run first.
4. The rest of the GPU category, excluding the classes already run in step 3. Run only from a checkout that includes #307,
   which applies the GPU labels this filter depends on. Build the filter in a variable so it has no embedded whitespace.
   Record failures by test name:
   ```
   filter="Category=GpuIntegration"
   for c in $classes; do filter="$filter&FullyQualifiedName!~HartsyInference.Cuda.Tests.$c."; done
   dotnet test tests/HartsyInference.Cuda.Tests --filter "$filter"
   ```
5. **Deferred: refactor A/B for the cache changes.** Do not run `tests/regression-ab.sh` as evidence for #304 yet. No
   production path constructs `CudaExpertCache` (only the class and its tests do), and no core regression case drives it,
   so both arms would bypass the changed cache and could report identical output vacuously. The requirement that
   generations stay identical stays open. Evidence for the cache refactor on the rig is step 3. Reopen this step once an
   executor or a dedicated harness acquires and evicts experts through the cache; then run
   `tests/regression-ab.sh --fresh --expect identical --backend cuda` on that case.
6. Vulkan MoE primitives, after the CUDA lane has finished, with no other GPU run in progress:
   `dotnet test tests/HartsyInference.Vulkan.Tests --filter "FullyQualifiedName~HartsyInference.Vulkan.Tests.VulkanMoePrimitiveTests."`
   (the trailing dot matches this class only).
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
